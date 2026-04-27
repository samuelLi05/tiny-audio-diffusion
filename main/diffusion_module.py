# This code has been adapted from Flavio Schneider's work with Archinet.
# (https://github.com/archinetai/audio-diffusion-pytorch-trainer)

from audio_data_pytorch.utils import fractional_random_split
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from audio_diffusion_pytorch import UNetV0, VDiffusion, VSampler, LTPlugin

import random
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import plotly.graph_objs as go
import pytorch_lightning as pl
import soundfile as sf
import torch
import torchaudio
import wandb

from einops import rearrange
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from scipy.signal import resample_poly
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from main.nsynth_waveform_dataset import NSynthWaveformDataset


""" Model """

# Option to use learned transform to downsample (by stride length) input data (not recommended).
# Can reduce computational load, but introduces undesirable high freq artifacts.
UNetT_LT = lambda: LTPlugin(UNetV0, num_filters=32, window_length=16, stride=16)

UNetT = lambda: UNetV0 # define Unet to be used (from audio_diffusion_pytorch)
DiffusionT = VDiffusion # define diffusion method to be used (from audio_diffusion_pytorch)
SamplerT = VSampler # define diffusion sampler to be used (from audio_diffusion_pytorch)

def dropout(proba: float):
    return random.random() < proba

class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.model = model
        self.model_ema = EMA(self.model, beta=ema_beta, power=ema_power)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        wave = batch
        loss = self.model(wave)
        self.log("train_loss", loss, sync_dist=True)
        
        # Update EMA model and log decay
        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        wave = batch
        loss = self.model_ema(wave)
        self.log("valid_loss", loss, sync_dist=True)
        return loss


class ConditionalModel(Model):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
        audio_channels: int,
        conditioning_dim: int,
        conditioning_mode: str = "onehot",
        conditioning_dropout: float = 0.0,
        num_classes: Optional[int] = None,
        label_embedding_dim: int = 32,
        use_contrastive_loss: bool = False,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        contrastive_projection_dim: int = 64,
    ):
        super().__init__(
            lr=lr,
            lr_beta1=lr_beta1,
            lr_beta2=lr_beta2,
            lr_eps=lr_eps,
            lr_weight_decay=lr_weight_decay,
            ema_beta=ema_beta,
            ema_power=ema_power,
            model=model,
        )
        if conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be > 0 for ConditionalModel")

        self.audio_channels = audio_channels
        self.conditioning_dim = conditioning_dim
        self.conditioning_mode = conditioning_mode
        self.conditioning_dropout = conditioning_dropout
        self.num_classes = num_classes if num_classes is not None else conditioning_dim

        self.use_label_embedding = conditioning_mode == "label_embedding"
        if self.use_label_embedding:
            self.class_embedding = nn.Embedding(self.num_classes, label_embedding_dim)
            self.embedding_to_conditioning = nn.Linear(
                label_embedding_dim, conditioning_dim
            )

        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        if self.use_contrastive_loss:
            self.audio_to_latent = nn.Linear(audio_channels, contrastive_projection_dim)
            self.text_embedding = nn.Embedding(self.num_classes, label_embedding_dim)
            self.text_to_latent = nn.Linear(
                label_embedding_dim, contrastive_projection_dim
            )

    def _encode_conditioning(
        self,
        conditioning: Optional[Tensor],
        class_ids: Optional[Tensor],
    ) -> Tensor:
        if self.conditioning_mode == "onehot":
            if conditioning is not None:
                cond = conditioning
            elif class_ids is not None:
                cond = F.one_hot(class_ids, num_classes=self.conditioning_dim).float()
            else:
                raise ValueError(
                    "Need conditioning tensor or class ids for onehot conditioning."
                )
        elif self.conditioning_mode == "label_embedding":
            if conditioning is not None:
                cond = conditioning
            elif class_ids is not None:
                embed = self.class_embedding(class_ids)
                cond = self.embedding_to_conditioning(embed)
            else:
                raise ValueError(
                    "Need conditioning tensor or class ids for label_embedding conditioning."
                )
        else:
            raise ValueError(f"Unsupported conditioning_mode: {self.conditioning_mode}")

        if self.training and self.conditioning_dropout > 0.0:
            if dropout(self.conditioning_dropout):
                cond = torch.zeros_like(cond)

        return cond

    def _build_conditioned_wave(self, wave: Tensor, cond: Tensor) -> Tensor:
        cond_map = cond.unsqueeze(-1).expand(-1, -1, wave.shape[-1])
        return torch.cat([wave, cond_map], dim=1)

    def _compute_contrastive_loss(self, clean_wave: Tensor, class_ids: Tensor) -> Tensor:
        audio_summary = clean_wave.mean(dim=-1)
        audio_latent = F.normalize(self.audio_to_latent(audio_summary), dim=-1)

        text_embed = self.text_embedding(class_ids)
        text_latent = F.normalize(self.text_to_latent(text_embed), dim=-1)

        logits = torch.matmul(audio_latent, text_latent.transpose(0, 1))
        logits = logits / self.contrastive_temperature
        targets = torch.arange(logits.shape[0], device=logits.device)

        loss_audio_to_text = F.cross_entropy(logits, targets)
        loss_text_to_audio = F.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (loss_audio_to_text + loss_text_to_audio)

    def _unpack_conditional_batch(self, batch: dict[str, Any]):
        clean_wave = batch["Clean Waveform"]
        conditioning = batch.get("Conditioning")
        class_ids = batch.get("Class Id")

        clean_wave = clean_wave.to(self.device)
        conditioning = conditioning.to(self.device) if conditioning is not None else None
        class_ids = class_ids.to(self.device) if class_ids is not None else None
        return clean_wave, conditioning, class_ids

    def training_step(self, batch, batch_idx):
        clean_wave, conditioning, class_ids = self._unpack_conditional_batch(batch)
        cond = self._encode_conditioning(conditioning, class_ids)
        conditioned_wave = self._build_conditioned_wave(clean_wave, cond)

        diffusion_loss = self.model(conditioned_wave)
        loss = diffusion_loss
        self.log("train_loss", diffusion_loss, sync_dist=True)

        if self.use_contrastive_loss and class_ids is not None:
            contrastive_loss = self._compute_contrastive_loss(clean_wave, class_ids)
            loss = loss + (self.contrastive_weight * contrastive_loss)
            self.log("train_contrastive_loss", contrastive_loss, sync_dist=True)

        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean_wave, conditioning, class_ids = self._unpack_conditional_batch(batch)
        cond = self._encode_conditioning(conditioning, class_ids)
        conditioned_wave = self._build_conditioned_wave(clean_wave, cond)

        loss = self.model_ema(conditioned_wave)
        self.log("valid_loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_conditioned(
        self,
        noise: Tensor,
        num_steps: int,
        conditioning: Optional[Tensor] = None,
        class_ids: Optional[Tensor] = None,
        use_ema_model: bool = True,
    ) -> Tensor:
        cond = self._encode_conditioning(conditioning, class_ids)
        cond = cond.to(noise.device)
        conditioned_noise = self._build_conditioned_wave(noise, cond)

        model = self.model_ema.ema_model if use_ema_model else self.model
        samples = model.sample(conditioned_noise, num_steps=num_steps)
        return samples[:, : self.audio_channels]


class EmbeddingConditionalModel(Model):
    """Conditional diffusion model that injects class conditioning via embedding kwargs.

    This avoids concatenating conditioning channels to the diffused waveform tensor,
    which can improve conditioning stability because only audio channels are diffused.
    """

    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
        conditioning_dim: int,
        conditioning_mode: str = "onehot",
        conditioning_dropout: float = 0.0,
        num_classes: Optional[int] = None,
        label_embedding_dim: int = 32,
        use_contrastive_loss: bool = False,
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.1,
        contrastive_projection_dim: int = 64,
    ):
        super().__init__(
            lr=lr,
            lr_beta1=lr_beta1,
            lr_beta2=lr_beta2,
            lr_eps=lr_eps,
            lr_weight_decay=lr_weight_decay,
            ema_beta=ema_beta,
            ema_power=ema_power,
            model=model,
        )
        if conditioning_dim <= 0:
            raise ValueError("conditioning_dim must be > 0 for EmbeddingConditionalModel")

        self.conditioning_dim = conditioning_dim
        self.conditioning_mode = conditioning_mode
        self.conditioning_dropout = conditioning_dropout
        self.num_classes = num_classes if num_classes is not None else conditioning_dim

        self.use_label_embedding = conditioning_mode == "label_embedding"
        if self.use_label_embedding:
            self.class_embedding = nn.Embedding(self.num_classes, label_embedding_dim)
            self.embedding_to_conditioning = nn.Linear(
                label_embedding_dim, conditioning_dim
            )

        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        if self.use_contrastive_loss:
            self.audio_to_latent = nn.Linear(1, contrastive_projection_dim)
            self.text_embedding = nn.Embedding(self.num_classes, label_embedding_dim)
            self.text_to_latent = nn.Linear(
                label_embedding_dim, contrastive_projection_dim
            )

    def _encode_conditioning(
        self,
        conditioning: Optional[Tensor],
        class_ids: Optional[Tensor],
    ) -> Tensor:
        if self.conditioning_mode == "onehot":
            if conditioning is not None:
                cond = conditioning
            elif class_ids is not None:
                cond = F.one_hot(class_ids, num_classes=self.conditioning_dim).float()
            else:
                raise ValueError(
                    "Need conditioning tensor or class ids for onehot conditioning."
                )
        elif self.conditioning_mode == "label_embedding":
            if conditioning is not None:
                cond = conditioning
            elif class_ids is not None:
                embed = self.class_embedding(class_ids)
                cond = self.embedding_to_conditioning(embed)
            else:
                raise ValueError(
                    "Need conditioning tensor or class ids for label_embedding conditioning."
                )
        else:
            raise ValueError(f"Unsupported conditioning_mode: {self.conditioning_mode}")

        if self.training and self.conditioning_dropout > 0.0:
            if dropout(self.conditioning_dropout):
                cond = torch.zeros_like(cond)

        return cond

    def _compute_contrastive_loss(self, clean_wave: Tensor, class_ids: Tensor) -> Tensor:
        audio_summary = clean_wave.mean(dim=-1)
        audio_latent = F.normalize(self.audio_to_latent(audio_summary), dim=-1)

        text_embed = self.text_embedding(class_ids)
        text_latent = F.normalize(self.text_to_latent(text_embed), dim=-1)

        logits = torch.matmul(audio_latent, text_latent.transpose(0, 1))
        logits = logits / self.contrastive_temperature
        targets = torch.arange(logits.shape[0], device=logits.device)

        loss_audio_to_text = F.cross_entropy(logits, targets)
        loss_text_to_audio = F.cross_entropy(logits.transpose(0, 1), targets)
        return 0.5 * (loss_audio_to_text + loss_text_to_audio)

    def _format_embedding(self, cond: Tensor) -> Tensor:
        # Cross-attention expects [batch, sequence, features].
        if cond.ndim == 2:
            return cond.unsqueeze(1)
        return cond

    def _unpack_conditional_batch(self, batch: dict[str, Any]):
        clean_wave = batch["Clean Waveform"]
        conditioning = batch.get("Conditioning")
        class_ids = batch.get("Class Id")

        clean_wave = clean_wave.to(self.device)
        conditioning = conditioning.to(self.device) if conditioning is not None else None
        class_ids = class_ids.to(self.device) if class_ids is not None else None
        return clean_wave, conditioning, class_ids

    def training_step(self, batch, batch_idx):
        clean_wave, conditioning, class_ids = self._unpack_conditional_batch(batch)
        cond = self._encode_conditioning(conditioning, class_ids)
        cond = self._format_embedding(cond)

        diffusion_loss = self.model(clean_wave, embedding=cond)
        loss = diffusion_loss
        self.log("train_loss", diffusion_loss, sync_dist=True)

        if self.use_contrastive_loss and class_ids is not None:
            contrastive_loss = self._compute_contrastive_loss(clean_wave, class_ids)
            loss = loss + (self.contrastive_weight * contrastive_loss)
            self.log("train_contrastive_loss", contrastive_loss, sync_dist=True)

        self.model_ema.update()
        self.log("ema_decay", self.model_ema.get_current_decay(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        clean_wave, conditioning, class_ids = self._unpack_conditional_batch(batch)
        cond = self._encode_conditioning(conditioning, class_ids)
        cond = self._format_embedding(cond)

        loss = self.model_ema(clean_wave, embedding=cond)
        self.log("valid_loss", loss, sync_dist=True)
        return loss

    @torch.no_grad()
    def sample_conditioned(
        self,
        noise: Tensor,
        num_steps: int,
        conditioning: Optional[Tensor] = None,
        class_ids: Optional[Tensor] = None,
        use_ema_model: bool = True,
    ) -> Tensor:
        cond = self._encode_conditioning(conditioning, class_ids)
        cond = self._format_embedding(cond).to(noise.device)
        model = self.model_ema.ema_model if use_ema_model else self.model
        return model.sample(noise, num_steps=num_steps, embedding=cond)


""" Datamodule """

class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        *,
        val_split: float,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        **kwargs: int,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.data_train: Any = None
        self.data_val: Any = None

    def setup(self, stage: Any = None) -> None:
        split = [1.0 - self.val_split, self.val_split]
        self.data_train, self.data_val = fractional_random_split(self.dataset, split)

    def get_dataloader(self, dataset) -> DataLoader:
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": True,
        }
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 2
            dataloader_kwargs["persistent_workers"] = True

        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.data_train)

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader(self.data_val)


class NSynthConditionalDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        metadata_path: str,
        *,
        batch_size: int,
        num_workers: int,
        pin_memory: bool = False,
        include_spectrogram: bool = True,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        lowpass_hz: Optional[float] = None,
        highpass_hz: Optional[float] = None,
        target_length: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.metadata_path = metadata_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.include_spectrogram = include_spectrogram
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.lowpass_hz = lowpass_hz
        self.highpass_hz = highpass_hz

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.class_to_index: dict[str, int] = {}

    def setup(self, stage: Any = None) -> None:
        self.data_train = NSynthWaveformDataset(
            self.metadata_path,
            split="train",
            target_sample_rate=self.sample_rate,
            target_length=self.target_length,
        )
        self.data_val = NSynthWaveformDataset(
            self.metadata_path,
            split="valid",
            target_sample_rate=self.sample_rate,
            target_length=self.target_length,
        )
        self.data_test = NSynthWaveformDataset(
            self.metadata_path,
            split="test",
            target_sample_rate=self.sample_rate,
            target_length=self.target_length,
        )

        classes = sorted(
            {
                row["class"]
                for row in self.data_train.rows + self.data_val.rows + self.data_test.rows
            }
        )
        self.class_to_index = {name: idx for idx, name in enumerate(classes)}

    def _apply_filters(self, wave: Tensor) -> Tensor:
        filtered = wave
        if self.highpass_hz is not None:
            filtered = torchaudio.functional.highpass_biquad(
                filtered,
                sample_rate=self.sample_rate,
                cutoff_freq=float(self.highpass_hz),
            )
        if self.lowpass_hz is not None:
            filtered = torchaudio.functional.lowpass_biquad(
                filtered,
                sample_rate=self.sample_rate,
                cutoff_freq=float(self.lowpass_hz),
            )
        return filtered

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        clean_wave = torch.stack([item["Clean Waveform"] for item in batch])
        noisy_wave = torch.stack([item["Noisy Waveform"] for item in batch])

        clean_wave = self._apply_filters(clean_wave)
        noisy_wave = self._apply_filters(noisy_wave)

        conditioning_values = [item["Conditioning"] for item in batch]
        conditioning = None
        if all(value is not None for value in conditioning_values):
            conditioning = torch.stack(conditioning_values)

        class_names = [item["Class"] for item in batch]
        class_ids = torch.tensor(
            [self.class_to_index[name] for name in class_names], dtype=torch.long
        )

        spectrogram = None
        if self.include_spectrogram:
            spectrogram = self.mel_transform(clean_wave[:, :1, :])

        return {
            "Sample_id": [item["Sample_id"] for item in batch],
            "Name": [item["Name"] for item in batch],
            "Split": [item["Split"] for item in batch],
            "Class": class_names,
            "Noise Type": [item["Noise Type"] for item in batch],
            "Class Id": class_ids,
            "Conditioning": conditioning,
            "Clean Waveform": clean_wave,
            "Noisy Waveform": noisy_wave,
            "Spectrogram": spectrogram,
        }

    def _get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        dataloader_kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "shuffle": shuffle,
            "collate_fn": self._collate_fn,
        }
        if self.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = 2
            dataloader_kwargs["persistent_workers"] = True

        return DataLoader(**dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.data_train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.data_val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.data_test, shuffle=False)


class NsynthWavDataset(Dataset):
    """Simple local WAV dataset for Nsynth subsets using soundfile decoding.

    This avoids torchaudio's decoding backend, which can fail in environments
    missing torchcodec/ffmpeg runtime libs.
    """

    def __init__(
        self,
        path: str,
        sample_rate: int,
        crop_size: int,
        stereo: bool = True,
        recursive: bool = True,
    ) -> None:
        super().__init__()
        self.path = Path(path)
        self.sample_rate = sample_rate
        self.crop_size = crop_size
        self.stereo = stereo

        if recursive:
            self.file_paths = sorted(self.path.rglob("*.wav"))
        else:
            self.file_paths = sorted(self.path.glob("*.wav"))

        if not self.file_paths:
            raise RuntimeError(f"No .wav files found in {self.path}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        if source_rate == self.sample_rate:
            return audio

        gcd = np.gcd(source_rate, self.sample_rate)
        up = self.sample_rate // gcd
        down = source_rate // gcd
        return resample_poly(audio, up=up, down=down, axis=1)

    def _to_channels(self, audio: np.ndarray) -> np.ndarray:
        channels = 2 if self.stereo else 1
        if audio.shape[0] == channels:
            return audio
        if channels == 1:
            return np.mean(audio, axis=0, keepdims=True)
        if audio.shape[0] == 1:
            return np.repeat(audio, repeats=2, axis=0)
        return audio[:2]

    def _crop_or_pad(self, audio: np.ndarray) -> np.ndarray:
        length = audio.shape[1]
        if length > self.crop_size:
            start = random.randint(0, length - self.crop_size)
            return audio[:, start : start + self.crop_size]
        if length < self.crop_size:
            pad_width = self.crop_size - length
            return np.pad(audio, ((0, 0), (0, pad_width)), mode="constant")
        return audio

    def __getitem__(self, index: int) -> Tensor:
        file_path = self.file_paths[index]
        audio, source_rate = sf.read(str(file_path), dtype="float32", always_2d=True)
        audio = audio.T
        audio = self._to_channels(audio)
        audio = self._resample(audio, source_rate)
        audio = self._crop_or_pad(audio)
        audio = np.clip(audio, -1.0, 1.0)
        return torch.from_numpy(audio).float()


""" Callbacks """

def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    print("WandbLogger not found.")
    return None


def log_wandb_audio_batch(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = rearrange(samples, "b c t -> b t c").detach().cpu().numpy()
    logger.log(
        {
            f"sample_{idx}_{id}": wandb.Audio(
                samples[idx],
                caption=caption,
                sample_rate=sampling_rate,
            )
            for idx in range(num_items)
        }
    )


def log_wandb_audio_spectrogram(
    logger: WandbLogger, id: str, samples: Tensor, sampling_rate: int, caption: str = ""
):
    num_items = samples.shape[0]
    samples = samples.detach().cpu()
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=80,
        center=True,
        norm="slaney",
    )

    def get_spectrogram_image(x):
        spectrogram = transform(x[0])
        image = torchaudio.functional.amplitude_to_DB(spectrogram, 1.0, 1e-10, 80.0)
        trace = [go.Heatmap(z=image, colorscale="viridis")]
        layout = go.Layout(
            yaxis=dict(title="Mel Bin (Log Frequency)"),
            xaxis=dict(title="Frame"),
            title_font_size=10,
        )
        fig = go.Figure(data=trace, layout=layout)
        return fig

    logger.log(
        {
            f"mel_spectrogram_{idx}_{id}": get_spectrogram_image(samples[idx])
            for idx in range(num_items)
        }
    )


class SampleLogger(Callback):
    def __init__(
        self,
        num_items: int,
        channels: int,
        sampling_rate: int,
        sampling_steps: List[int],
        use_ema_model: bool,
        length: int,
    ) -> None:
        self.num_items = num_items
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.sampling_steps = sampling_steps
        self.use_ema_model = use_ema_model
        self.log_next = False
        self.length = length


    def on_validation_epoch_start(self, trainer, pl_module):
        self.log_next = True

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx
    ):
        # Skip expensive sample generation during Lightning sanity checking.
        if getattr(trainer, "sanity_checking", False):
            return

        if self.log_next and trainer.logger: # only log if logger present in config
            self.log_sample(trainer, pl_module, batch)
            self.log_next = False

    @torch.no_grad()
    def log_sample(self, trainer, pl_module, batch):
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        # Get wandb logger
        wandb_pl_logger = get_wandb_logger(trainer)
        if wandb_pl_logger is None:
            if is_train:
                pl_module.train()
            return
        wandb_logger = wandb_pl_logger.experiment

        model = pl_module.model

        if self.use_ema_model:
            model = pl_module.model_ema.ema_model


        # Get noise for diffusion inference
        noise = torch.randn(
            (self.num_items, self.channels, self.length), device=pl_module.device
        )

        conditioning = None
        class_ids = None
        if isinstance(batch, dict):
            conditioning = batch.get("Conditioning")
            class_ids = batch.get("Class Id")
            if conditioning is not None:
                conditioning = conditioning[: self.num_items].to(pl_module.device)
            if class_ids is not None:
                class_ids = class_ids[: self.num_items].to(pl_module.device)

        for steps in self.sampling_steps:
            if hasattr(pl_module, "sample_conditioned"):
                samples = pl_module.sample_conditioned(
                    noise=noise,
                    num_steps=steps,
                    conditioning=conditioning,
                    class_ids=class_ids,
                    use_ema_model=self.use_ema_model,
                )
            else:
                samples = model.sample(
                    noise,
                    num_steps=steps,
                )
            log_wandb_audio_batch(
                logger=wandb_logger,
                id="sample",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )
            log_wandb_audio_spectrogram(
                logger=wandb_logger,
                id="sample",
                samples=samples,
                sampling_rate=self.sampling_rate,
                caption=f"Sampled in {steps} steps",
            )

        if is_train:
            pl_module.train()
