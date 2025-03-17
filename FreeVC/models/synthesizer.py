import torch
import torch.nn as nn

from commons import rand_slice_segments

from .deep_speaker import extract_speaker_embedding
from .encoder import Encoder
from .generator import Generator
from .residual_coupling_block import ResidualCouplingBlock


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
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels,
        ssl_dim,
    ):
        super().__init__()

        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.segment_size = segment_size
        self.gin_channels = gin_channels

        self.enc_p = Encoder(
            ssl_dim,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
        )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = Encoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels,
            hidden_channels,
            5,
            1,
            4,
            gin_channels=gin_channels,
        )

    def forward(self, c, spec, filenames=None, c_lengths=None, spec_lengths=None):
        device = c.device

        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(device)

        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

        g = self._extract_speaker_embeddings(filenames) if filenames else None

        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)

        z, m_q, logs_q, spec_mask = self.enc_q(spec, spec_lengths, g=g)

        z_p = self.flow(z, spec_mask, g=g)

        z_slice, ids_slice = rand_slice_segments(z, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    def infer(self, c, filenames=None, c_lengths=None):
        device = c.device

        c_lengths = c_lengths or torch.full((c.size(0),), c.size(-1), device=device)
        g = self._extract_speaker_embeddings(filenames) if filenames else None

        z_p, _, _, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)

        return self.dec(z * c_mask, g=g)

    def _extract_speaker_embeddings(self, filenames):
        """
        Extracts speaker embeddings from filenames.
        """
        g_list = [extract_speaker_embedding(filename) for filename in filenames]
        g = torch.stack(g_list, dim=0).permute(0, 2, 1)  # Shape: (batch_size, 1, 512)
        return g
