import copy
import math
import random

import numpy as np
import torch
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

import commons
import modules
from commons import get_padding, init_weights
from DeepSpeaker.deep_speaker.audio import read_mfcc
from DeepSpeaker.deep_speaker.batcher import sample_from_mfcc
from DeepSpeaker.deep_speaker.constants import NUM_FRAMES, SAMPLE_RATE
from DeepSpeaker.deep_speaker.conv_models import DeepSpeakerModel

# Load the Deep Speaker model
deep_speaker_model = DeepSpeakerModel()
deep_speaker_model.m.load_weights(
    "checkpoints/ResCNN_triplet_training_checkpoint_265.h5", by_name=True
)


def extract_speaker_embedding(
    wav_file, device="cuda" if torch.cuda.is_available() else "cpu"
):
    """Extract speaker embedding from a WAV file using Deep Speaker."""
    mfcc = sample_from_mfcc(read_mfcc(wav_file, SAMPLE_RATE), NUM_FRAMES)
    embedding = deep_speaker_model.m.predict(
        np.expand_dims(mfcc, axis=0)
    )  # Shape: (1, 512)
    embedding = torch.tensor(embedding, dtype=torch.float32).to(
        device
    )  # Move to correct device
    return embedding  # Shape: (1, 512)


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super().__init__()
        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )

        self.ups = nn.ModuleList(
            [
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
                for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
            ]
        )

        self.resblocks = nn.ModuleList(
            [
                modules.ResBlock1(upsample_initial_channel // (2 ** (i + 1)), k, d)
                for i in range(len(self.ups))
                for j, (k, d) in enumerate(
                    zip(resblock_kernel_sizes, resblock_dilation_sizes)
                )
            ]
        )

        self.conv_post = Conv1d(
            upsample_initial_channel // (2 ** len(self.ups)),
            1,
            7,
            1,
            padding=3,
            bias=False,
        )
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = (
                sum(
                    self.resblocks[i * self.num_kernels + j](x)
                    for j in range(self.num_kernels)
                )
                / self.num_kernels
            )
            x = xs

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        return torch.tanh(x)


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training using Deep Speaker.
    """

    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
        use_spk,
        **kwargs
    ):

        super().__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.gin_channels = 512

        self.enc_p = Encoder(ssl_dim, inter_channels, hidden_channels, 5, 1, 16)
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )
        self.enc_q = Encoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=self.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=self.gin_channels
        )

    def forward(self, c, spec, wav_file=None, c_lengths=None, spec_lengths=None):
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(c.device)

        if wav_file is not None:
            g = extract_speaker_embedding(wav_file)  # Shape: (1, 512)
            g = g.unsqueeze(-1)

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)
        z_p = self.flow(z, spec_mask, g=g)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, spec_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c, wav_file=None, c_lengths=None):
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        if wav_file is not None:
            g = extract_speaker_embedding(wav_file)
            g = g.unsqueeze(-1)

        z_p, _, _, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        return self.dec(z * c_mask, g=g)
