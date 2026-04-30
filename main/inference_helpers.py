from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
import os
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import torch
import torchaudio
import yaml
from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler
from torch import Tensor
from torch.nn import functional as F

from main.diffusion_module import (
    ConditionalModel,
    EmbeddingConditionalModel,
    Model,
    NSynthConditionalDatamodule,
)


@dataclass
class InferenceContext:
    config_path: str
    ckpt_path: str
    config: dict[str, Any]
    model: Model
    datamodule: Optional[NSynthConditionalDatamodule]
    class_names: list[str]
    conditioning_mode: str
    conditioning_dim: int
    sample_rate: int
    sample_length: int
    audio_channels: int
    is_conditional: bool
    is_embedding_model: bool
    metadata_path: Optional[str]


_TOKEN_PATTERN = re.compile(r"\$\{([^{}]+)\}")
_FULL_TOKEN_PATTERN = re.compile(r"^\$\{([^{}]+)\}$")


def _lookup_dot_path(root: dict[str, Any], key: str) -> Any:
    current: Any = root
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(key)
        current = current[part]
    return current


def _resolve_token(token: str, root: dict[str, Any], repo_root: str) -> Any:
    if token == "hydra:runtime.cwd":
        return repo_root

    if token.startswith("oc.env:"):
        raw = token[len("oc.env:") :]
        if "," in raw:
            env_name, default = raw.split(",", 1)
            env_name = env_name.strip()
            default = default.strip()
            return os.environ.get(env_name, default)
        return os.environ.get(raw.strip(), "")

    try:
        value = _lookup_dot_path(root, token)
    except KeyError:
        return "${" + token + "}"
    return value


def _resolve_simple_refs(value: Any, root: dict[str, Any], repo_root: str):
    if isinstance(value, dict):
        return {key: _resolve_simple_refs(inner, root, repo_root) for key, inner in value.items()}
    if isinstance(value, list):
        return [_resolve_simple_refs(item, root, repo_root) for item in value]
    if isinstance(value, str):
        full_match = _FULL_TOKEN_PATTERN.fullmatch(value)
        if full_match:
            resolved_value = _resolve_token(full_match.group(1), root, repo_root)
            if isinstance(resolved_value, str) and _FULL_TOKEN_PATTERN.fullmatch(resolved_value):
                return _resolve_simple_refs(resolved_value, root, repo_root)
            return resolved_value

        resolved = value
        for _ in range(6):
            updated = _TOKEN_PATTERN.sub(
                lambda match: str(_resolve_token(match.group(1), root, repo_root)),
                resolved,
            )
            if updated == resolved:
                break
            resolved = updated
        return resolved
    return value


def load_resolved_config(config_path: str) -> dict[str, Any]:
    config_path_obj = Path(config_path).expanduser().resolve()
    repo_root = config_path_obj.parent.parent
    base_config_path = repo_root / "config.yaml"

    merged_root: dict[str, Any] = {}
    if base_config_path.exists():
        with base_config_path.open("r", encoding="utf-8") as handle:
            base_raw = yaml.safe_load(handle) or {}
        if isinstance(base_raw, dict):
            merged_root.update(base_raw)

    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle) or {}

    if isinstance(raw_config, dict):
        merged_root.update(raw_config)

    resolved_root = _resolve_simple_refs(merged_root, merged_root, str(repo_root))
    if not isinstance(resolved_root, dict):
        raise ValueError(f"Resolved config is not a dict for: {config_path}")
    return resolved_root


def _build_core_model(model_configs: dict[str, Any]) -> DiffusionModel:
    core_kwargs = dict(
        net_t=UNetV0,
        in_channels=model_configs["in_channels"],
        channels=model_configs["channels"],
        factors=model_configs["factors"],
        items=model_configs["items"],
        attentions=model_configs["attentions"],
        attention_heads=model_configs["attention_heads"],
        attention_features=model_configs["attention_features"],
        diffusion_t=VDiffusion,
        sampler_t=VSampler,
    )
    if "cross_attentions" in model_configs:
        core_kwargs["cross_attentions"] = model_configs["cross_attentions"]
    if "embedding_features" in model_configs:
        core_kwargs["embedding_features"] = model_configs["embedding_features"]
    return DiffusionModel(**core_kwargs)


def _class_names_from_context(
    datamodule: Optional[NSynthConditionalDatamodule],
    metadata_path: Optional[str],
    conditioning_dim: int,
) -> list[str]:
    if datamodule is not None and datamodule.class_to_index:
        return [
            name
            for name, _ in sorted(datamodule.class_to_index.items(), key=lambda item: item[1])
        ]

    if metadata_path is not None:
        metadata_file = Path(metadata_path).expanduser().resolve()
        if metadata_file.exists():
            classes: set[str] = set()
            with metadata_file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    class_name = row.get("class")
                    if class_name is not None:
                        classes.add(str(class_name))
            if classes:
                return sorted(classes)

    if conditioning_dim > 0:
        raise ValueError(
            "Unable to infer class names from metadata. "
            "Provide a valid metadata jsonl via datamodule.metadata_path or metadata_path_override."
        )

    return []


def load_inference_context(
    config_path: str,
    ckpt_path: str,
    *,
    conditioning_mode_override: Optional[str] = None,
    metadata_path_override: Optional[str] = None,
    device: Optional[str] = None,
) -> InferenceContext:
    config = load_resolved_config(config_path)
    pl_configs = config["model"]
    model_configs = config["model"]["model"]
    datamodule_cfg = config.get("datamodule", {})
    model_target = str(config["model"].get("_target_", ""))
    is_embedding_model = "EmbeddingConditionalModel" in model_target
    is_conditional = ("ConditionalModel" in model_target) or is_embedding_model

    if conditioning_mode_override is not None:
        pl_configs["conditioning_mode"] = conditioning_mode_override

    conditioning_mode = pl_configs.get("conditioning_mode", "onehot")
    conditioning_dim = int(pl_configs.get("conditioning_dim", config.get("conditioning_dim", 3)))
    audio_channels = int(config.get("audio_channels", model_configs.get("in_channels", 1)))
    sample_rate = int(config.get("sampling_rate", 16000))
    sample_length = int(config.get("length", 16384))
    class_names: list[str] = []

    resolved_metadata_path: Optional[str] = None
    if is_conditional and str(datamodule_cfg.get("_target_", "")).endswith("NSynthConditionalDatamodule"):
        resolved_metadata_path = metadata_path_override or datamodule_cfg["metadata_path"]
        metadata_class_names = _class_names_from_context(None, resolved_metadata_path, 0)
        if metadata_class_names:
            inferred_dim = len(metadata_class_names)
            conditioning_dim = inferred_dim
            config["conditioning_dim"] = inferred_dim
            pl_configs["conditioning_dim"] = inferred_dim
            pl_configs["num_classes"] = inferred_dim

            if is_embedding_model:
                model_configs["embedding_features"] = inferred_dim
            else:
                model_configs["in_channels"] = audio_channels + inferred_dim

    core_model = _build_core_model(model_configs)

    if is_conditional:
        if is_embedding_model:
            model: Model = EmbeddingConditionalModel(
                lr=pl_configs["lr"],
                lr_beta1=pl_configs["lr_beta1"],
                lr_beta2=pl_configs["lr_beta2"],
                lr_eps=pl_configs["lr_eps"],
                lr_weight_decay=pl_configs["lr_weight_decay"],
                ema_beta=pl_configs["ema_beta"],
                ema_power=pl_configs["ema_power"],
                model=core_model,
                conditioning_dim=conditioning_dim,
                conditioning_mode=conditioning_mode,
                conditioning_dropout=pl_configs.get("conditioning_dropout", 0.0),
                num_classes=pl_configs.get("num_classes", conditioning_dim),
                label_embedding_dim=pl_configs.get("label_embedding_dim", 32),
                use_contrastive_loss=pl_configs.get("use_contrastive_loss", False),
                contrastive_weight=pl_configs.get("contrastive_weight", 0.1),
                contrastive_temperature=pl_configs.get("contrastive_temperature", 0.1),
                contrastive_projection_dim=pl_configs.get("contrastive_projection_dim", 64),
            )
        else:
            model = ConditionalModel(
                lr=pl_configs["lr"],
                lr_beta1=pl_configs["lr_beta1"],
                lr_beta2=pl_configs["lr_beta2"],
                lr_eps=pl_configs["lr_eps"],
                lr_weight_decay=pl_configs["lr_weight_decay"],
                ema_beta=pl_configs["ema_beta"],
                ema_power=pl_configs["ema_power"],
                model=core_model,
                audio_channels=audio_channels,
                conditioning_dim=conditioning_dim,
                conditioning_mode=conditioning_mode,
                conditioning_dropout=pl_configs.get("conditioning_dropout", 0.0),
                num_classes=pl_configs.get("num_classes", conditioning_dim),
                label_embedding_dim=pl_configs.get("label_embedding_dim", 32),
                use_contrastive_loss=pl_configs.get("use_contrastive_loss", False),
                contrastive_weight=pl_configs.get("contrastive_weight", 0.1),
                contrastive_temperature=pl_configs.get("contrastive_temperature", 0.1),
                contrastive_projection_dim=pl_configs.get("contrastive_projection_dim", 64),
            )
    else:
        model = Model(
            lr=pl_configs["lr"],
            lr_beta1=pl_configs["lr_beta1"],
            lr_beta2=pl_configs["lr_beta2"],
            lr_eps=pl_configs["lr_eps"],
            lr_weight_decay=pl_configs["lr_weight_decay"],
            ema_beta=pl_configs["ema_beta"],
            ema_power=pl_configs["ema_power"],
            model=core_model,
        )

    if torch.cuda.is_available():
        model = model.to(device or "cuda")

    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    datamodule = None
    if is_conditional:
        if str(datamodule_cfg.get("_target_", "")).endswith("NSynthConditionalDatamodule"):
            datamodule = NSynthConditionalDatamodule(
                metadata_path=resolved_metadata_path,
                batch_size=datamodule_cfg["batch_size"],
                num_workers=datamodule_cfg["num_workers"],
                pin_memory=datamodule_cfg.get("pin_memory", False),
                include_spectrogram=datamodule_cfg.get("include_spectrogram", True),
                sample_rate=datamodule_cfg.get("sample_rate", sample_rate),
                n_fft=datamodule_cfg.get("n_fft", 1024),
                hop_length=datamodule_cfg.get("hop_length", 256),
                n_mels=datamodule_cfg.get("n_mels", 128),
                lowpass_hz=datamodule_cfg.get("lowpass_hz", None),
                highpass_hz=datamodule_cfg.get("highpass_hz", None),
                target_length=datamodule_cfg.get("target_length", sample_length),
            )
            datamodule.setup()

    class_names = _class_names_from_context(datamodule, resolved_metadata_path, conditioning_dim)

    return InferenceContext(
        config_path=config_path,
        ckpt_path=ckpt_path,
        config=config,
        model=model,
        datamodule=datamodule,
        class_names=class_names,
        conditioning_mode=conditioning_mode,
        conditioning_dim=conditioning_dim,
        sample_rate=sample_rate,
        sample_length=sample_length,
        audio_channels=audio_channels,
        is_conditional=is_conditional,
        is_embedding_model=is_embedding_model,
        metadata_path=resolved_metadata_path,
    )


def class_conditioning(class_name: str, class_names: list[str], device: torch.device | str):
    if class_name not in class_names:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of {class_names}.")
    class_id = torch.tensor([class_names.index(class_name)], dtype=torch.long, device=device)
    conditioning = F.one_hot(class_id, num_classes=len(class_names)).float()
    return class_id, conditioning


def prepare_conditioning_inputs(context: InferenceContext, class_name: str):
    class_id, onehot_conditioning = class_conditioning(
        class_name,
        context.class_names,
        context.model.device,
    )
    conditioning_for_model = onehot_conditioning if context.conditioning_mode == "onehot" else None
    return class_id, conditioning_for_model


def sample_reference_item(
    datamodule: NSynthConditionalDatamodule,
    class_name: str,
    *,
    split: str = "test",
    index: Optional[int] = None,
    seed: Optional[int] = None,
):
    if datamodule is None:
        raise ValueError("This inference context does not include a conditional datamodule.")

    dataset = getattr(datamodule, f"data_{split}")
    if dataset is None:
        raise ValueError(f"Split '{split}' is not available.")

    matching = [i for i, row in enumerate(dataset.rows) if row["class"] == class_name]
    if not matching:
        raise ValueError(f"No samples found for class '{class_name}' in split '{split}'.")

    if index is not None:
        if index not in matching:
            raise ValueError(
                f"Requested index {index} does not belong to class '{class_name}' in split '{split}'."
            )
        chosen_index = index
    else:
        rng = random.Random(seed)
        chosen_index = rng.choice(matching)

    return chosen_index, dataset[chosen_index]


def resolve_reference_waveform(
    context: InferenceContext,
    *,
    class_name: str,
    seed: int,
    reference_split: str,
    reference_index: Optional[int],
    reference_input_path: Optional[str] = None,
    reference_pt_path: Optional[str] = None,
):
    override_path = reference_input_path if reference_input_path is not None else reference_pt_path

    if override_path is not None:
        waveform = load_waveform_file(
            override_path,
            target_length=context.sample_length,
            target_sample_rate=context.sample_rate,
            target_channels=context.audio_channels,
        )
        return waveform, override_path, None

    if context.datamodule is None:
        raise ValueError("No datamodule available to resolve a dataset reference waveform.")

    reference_sample_index, reference_item = sample_reference_item(
        context.datamodule,
        class_name,
        split=reference_split,
        index=reference_index,
        seed=seed,
    )
    waveform = reference_item["Clean Waveform"]
    label = f"{reference_split}/{reference_sample_index}"
    return waveform, label, reference_item


def load_waveform_pt(
    pt_path: str,
    *,
    target_length: int,
    target_sample_rate: Optional[int] = None,
    source_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
) -> Tensor:
    loaded = torch.load(pt_path, map_location="cpu")
    if isinstance(loaded, dict):
        waveform = loaded.get("waveform")
        if waveform is None:
            waveform = loaded.get("audio")
        if waveform is None:
            waveform = loaded.get("data")
        if waveform is None:
            raise ValueError(
                f"PT file {pt_path} must contain a tensor or one of: waveform, audio, data."
            )
        source_sample_rate = loaded.get("sample_rate", source_sample_rate)
    else:
        waveform = loaded

    return _normalize_waveform(
        waveform,
        target_length=target_length,
        target_sample_rate=target_sample_rate,
        source_sample_rate=source_sample_rate,
        target_channels=target_channels,
    )


def load_waveform_file(
    path: str,
    *,
    target_length: int,
    target_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
) -> Tensor:
    resolved = Path(path).expanduser().resolve()
    suffix = resolved.suffix.lower()

    if suffix in {".pt", ".pth"}:
        return load_waveform_pt(
            str(resolved),
            target_length=target_length,
            target_sample_rate=target_sample_rate,
            target_channels=target_channels,
        )

    if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}:
        try:
            waveform, source_sample_rate = torchaudio.load(str(resolved))
        except Exception:
            import soundfile as sf

            audio_np, source_sample_rate = sf.read(str(resolved), always_2d=True, dtype="float32")
            waveform = torch.from_numpy(audio_np).transpose(0, 1)
        return _normalize_waveform(
            waveform,
            target_length=target_length,
            target_sample_rate=target_sample_rate,
            source_sample_rate=source_sample_rate,
            target_channels=target_channels,
        )

    raise ValueError(
        f"Unsupported reference file type for {resolved}. Use an audio file (e.g., .wav) or .pt/.pth tensor file."
    )


def _normalize_waveform(
    waveform: Tensor,
    *,
    target_length: int,
    target_sample_rate: Optional[int] = None,
    source_sample_rate: Optional[int] = None,
    target_channels: Optional[int] = None,
) -> Tensor:
    audio = torch.as_tensor(waveform).detach().float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim > 2:
        audio = audio.view(audio.shape[0], -1)

    if target_channels is not None:
        if audio.shape[0] != target_channels:
            if target_channels == 1:
                audio = audio.mean(dim=0, keepdim=True)
            elif audio.shape[0] == 1:
                audio = audio.repeat(target_channels, 1)
            else:
                audio = audio[:target_channels]

    if (
        target_sample_rate is not None
        and source_sample_rate is not None
        and source_sample_rate != target_sample_rate
    ):
        audio = torchaudio.functional.resample(
            audio,
            source_sample_rate,
            target_sample_rate,
        )

    if audio.shape[-1] > target_length:
        audio = audio[..., :target_length]
    elif audio.shape[-1] < target_length:
        audio = torch.nn.functional.pad(audio, (0, target_length - audio.shape[-1]))

    return audio.clamp(-1.0, 1.0)


def prepare_audio_for_display(waveform: Tensor) -> Tensor:
    audio = waveform.detach().float().cpu()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.squeeze(0)
    return audio.clamp(-1.0, 1.0)


def generate_unconditional_sample(
    context: InferenceContext,
    *,
    num_steps: int,
    seed: int,
    unconditional_noise_scale: float,
) -> Tensor:
    generator = torch.Generator(device=context.model.device).manual_seed(seed)
    noise = torch.randn(
        (1, context.audio_channels, context.sample_length),
        generator=generator,
        device=context.model.device,
    ) * unconditional_noise_scale

    if hasattr(context.model, "sample_conditioned"):
        zero_conditioning = torch.zeros(
            (1, context.conditioning_dim),
            dtype=torch.float32,
            device=context.model.device,
        )
        sample = context.model.sample_conditioned(
            noise=noise,
            num_steps=num_steps,
            conditioning=zero_conditioning,
            class_ids=None,
            use_ema_model=True,
        )
    else:
        sample = context.model.model_ema.ema_model.sample(noise, num_steps=num_steps)

    return sample.squeeze(0).detach().cpu()


def generate_unconditional_samples(
    context: InferenceContext,
    *,
    num_samples: int,
    num_steps: int,
    seed: int,
    noise_scale: float,
) -> list[Tensor]:
    outputs: list[Tensor] = []
    for offset in range(num_samples):
        outputs.append(
            generate_unconditional_sample(
                context,
                num_steps=num_steps,
                seed=seed + offset,
                unconditional_noise_scale=noise_scale,
            )
        )
    return outputs


def build_noised_model_input(
    reference_waveform: Tensor,
    *,
    model_device: torch.device | str,
    seed: int,
    noise_scale: float,
) -> Tensor:
    base_input = prepare_audio_for_display(reference_waveform).unsqueeze(0).to(model_device)
    generator = torch.Generator(device=model_device).manual_seed(seed)
    input_noise = torch.randn(
        base_input.shape,
        generator=generator,
        device=model_device,
        dtype=base_input.dtype,
    ) * noise_scale
    return (base_input + input_noise).clamp(-1.0, 1.0)


def generate_conditional_sample_from_input(
    context: InferenceContext,
    *,
    model_input: Tensor,
    class_id: Tensor,
    conditioning_for_model: Optional[Tensor],
    num_steps: int,
) -> Tensor:
    sample = context.model.sample_conditioned(
        noise=model_input,
        num_steps=num_steps,
        conditioning=conditioning_for_model,
        class_ids=class_id,
        use_ema_model=True,
    )
    return sample.squeeze(0).detach().cpu()


def generate_conditional_samples_from_reference(
    context: InferenceContext,
    *,
    class_id: Tensor,
    conditioning_for_model: Optional[Tensor],
    reference_waveform: Tensor,
    num_samples: int,
    num_steps: int,
    seed: int,
    noise_scale: float,
) -> list[dict[str, Tensor]]:
    results: list[dict[str, Tensor]] = []
    for offset in range(num_samples):
        model_input = build_noised_model_input(
            reference_waveform,
            model_device=context.model.device,
            seed=seed + offset,
            noise_scale=noise_scale,
        )
        generated = generate_conditional_sample_from_input(
            context,
            model_input=model_input,
            class_id=class_id,
            conditioning_for_model=conditioning_for_model,
            num_steps=num_steps,
        )
        results.append(
            {
                "model_input": model_input.squeeze(0).detach().cpu(),
                "generated": generated,
            }
        )
    return results


def generate_class_conditioned_samples(
    context: InferenceContext,
    class_name: str,
    *,
    num_samples: int,
    num_steps: int,
    seed: int,
    noise_scale: float,
    seed_waveform: Optional[Tensor] = None,
):
    class_id, conditioning = class_conditioning(class_name, context.class_names, context.model.device)
    generated = []

    base_waveform = None
    if seed_waveform is not None:
        base_waveform = prepare_audio_for_display(seed_waveform)
        if base_waveform.ndim == 2:
            base_waveform = base_waveform.unsqueeze(0)

    for offset in range(num_samples):
        generator = torch.Generator(device=context.model.device).manual_seed(seed + offset)
        if base_waveform is None:
            noise = torch.randn(
                (1, context.audio_channels, context.sample_length),
                generator=generator,
                device=context.model.device,
            ) * noise_scale
        else:
            noise = base_waveform.to(context.model.device)
            if noise_scale > 0:
                extra_noise = torch.randn(
                    noise.shape,
                    generator=generator,
                    device=context.model.device,
                    dtype=noise.dtype,
                )
                noise = noise + (extra_noise * noise_scale)
        if hasattr(context.model, "sample_conditioned"):
            sample = context.model.sample_conditioned(
                noise=noise,
                num_steps=num_steps,
                conditioning=conditioning,
                class_ids=class_id,
                use_ema_model=True,
            )
        else:
            sample = context.model.model_ema.ema_model.sample(noise, num_steps=num_steps)
        generated.append(sample.squeeze(0).detach().cpu())

    return generated, {"class_ids": class_id, "conditioning": conditioning}


def plot_mel_spectrogram(sample: Tensor, *, sample_rate: int):
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )
    mono = sample.mean(dim=0) if sample.ndim == 2 else sample
    spectrogram = transform(mono)
    spectrogram = torchaudio.functional.amplitude_to_DB(spectrogram, 1.0, 1e-10, 80.0)

    fig = plt.figure(figsize=(7, 4))
    plt.imshow(spectrogram, aspect="auto", origin="lower")
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Frame")
    plt.ylabel("Mel Bin")
    plt.tight_layout()
    return fig