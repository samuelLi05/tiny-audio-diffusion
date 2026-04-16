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
from torch.utils.data import DataLoader, Dataset


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
            list(self.model.parameters()),
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

        for steps in self.sampling_steps:
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
