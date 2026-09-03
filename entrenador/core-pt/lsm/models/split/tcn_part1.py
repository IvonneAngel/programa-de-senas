"""Modelos PyTorch v2 para reconocimiento eficiente de LSM."""
from __future__ import annotations

import math
from typing import Iterable

try:
    import torch
    from torch import Tensor, nn
    from torch.autograd import Function
    import torch.nn.functional as F
except ModuleNotFoundError:  # permite importar documentación sin PyTorch
    torch = None
    Tensor = object
    nn = None
    F = None
    Function = object

_BaseModule = nn.Module if nn is not None else object


STATIC_LETTERS = 21
DYNAMIC_LETTERS = 6


def _require_torch():
    if torch is None or nn is None:
        raise ImportError("PyTorch es necesario para construir los modelos v2.")



class _ReverseGradient(Function):
    """Identidad en avance; invierte y escala la señal hacia el backbone."""

    @staticmethod
    def forward(ctx, values: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.scale * gradient, None

class StaticLettersMLP(_BaseModule):
    def __init__(self, feature_dim: int = 93, classes: int = STATIC_LETTERS, dropout: float = 0.20):
        _require_torch()
        super().__init__()
        self.frames = 1
        self.features = feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 96),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(96, classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        return self.net(x[:, 0, :])

class TCNResidualBlock(_BaseModule):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)

class PartialTemporalResidualBlock(TCNResidualBlock):
    """Bloque residual que no trata frames sin detección como landmarks observados."""

    def forward_with_mask(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        y, mask1 = partial_temporal_convolution(x, mask, self.conv1)
        y = F.gelu(self.norm1(y)) * mask1
        y = self.dropout(y)
        y, mask2 = partial_temporal_convolution(y, mask1, self.conv2)
        y = self.norm2(y) * mask2
        return F.gelu(x + y), mask2

class DeformableTemporalOffsetResidualBlock(TCNResidualBlock):
    """Bloque TCN con desplazamiento local aprendible e identidad inicial."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__(channels, dilation, dropout)
        self.offset1 = nn.Conv1d(channels, 1, kernel_size=3, padding=dilation, dilation=dilation)
        self.offset2 = nn.Conv1d(channels, 1, kernel_size=3, padding=dilation, dilation=dilation)
        nn.init.zeros_(self.offset1.weight)
        nn.init.zeros_(self.offset1.bias)
        nn.init.zeros_(self.offset2.weight)
        nn.init.zeros_(self.offset2.bias)

    @staticmethod
    def interpolate_temporally(x: Tensor, raw_offsets: Tensor) -> Tensor:
        if x.ndim != 3 or raw_offsets.shape != (x.shape[0], 1, x.shape[2]):
            raise ValueError("Offsets temporales incompatibles con la secuencia")
        frames = x.shape[2]
        base = torch.arange(frames, device=x.device, dtype=x.dtype).view(1, 1, frames)
        positions = (base + torch.tanh(raw_offsets)).clamp(0.0, float(frames - 1))
        left = positions.floor()
        right = (left + 1.0).clamp(max=float(frames - 1))
        fraction = positions - left
        channels = x.shape[1]
        left_values = x.gather(2, left.to(dtype=torch.long).expand(-1, channels, -1))
        right_values = x.gather(2, right.to(dtype=torch.long).expand(-1, channels, -1))
        return left_values * (1.0 - fraction) + right_values * fraction

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv1(self.interpolate_temporally(x, self.offset1(x)))
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.norm2(self.conv2(self.interpolate_temporally(y, self.offset2(y))))
        return F.gelu(x + y)

class ParallelReceptiveFieldTCNResidualBlock(_BaseModule):
    """W64: dos receptivos en paralelo, con igual presupuesto convolucional que W33."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        if channels % 2 != 0 or dilation < 1:
            raise ValueError("W64 requiere canales pares y dilatación positiva")
        branch_channels = channels // 2
        self.dilation = int(dilation)
        self.branch_dilations = (self.dilation, 2 * self.dilation)
        self.conv1_short = nn.Conv1d(channels, branch_channels, 3, padding=self.dilation, dilation=self.dilation, bias=False)
        self.conv1_long = nn.Conv1d(channels, branch_channels, 3, padding=2 * self.dilation, dilation=2 * self.dilation, bias=False)
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.conv2_short = nn.Conv1d(channels, branch_channels, 3, padding=self.dilation, dilation=self.dilation, bias=False)
        self.conv2_long = nn.Conv1d(channels, branch_channels, 3, padding=2 * self.dilation, dilation=2 * self.dilation, bias=False)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.dropout = nn.Dropout(dropout)

    def split_transform_merge(self, x: Tensor, stage: int) -> Tensor:
        if stage == 1:
            return torch.cat([self.conv1_short(x), self.conv1_long(x)], dim=1)
        if stage == 2:
            return torch.cat([self.conv2_short(x), self.conv2_long(x)], dim=1)
        raise ValueError("W64 stage debe ser 1 o 2")

    def forward(self, x: Tensor) -> Tensor:
        y = F.gelu(self.norm1(self.split_transform_merge(x, 1)))
        y = self.dropout(y)
        y = self.norm2(self.split_transform_merge(y, 2))
        return F.gelu(x + y)

class TemporalTCN(_BaseModule):
    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        super().__init__()
        self.frames = frames
        self.features = feature_dim
        self.stem = nn.Conv1d(feature_dim, channels, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[TCNResidualBlock(channels, dilation, dropout) for dilation in dilations])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def forward_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        x = x.transpose(1, 2)
        x = F.gelu(self.stem(x))
        return self.blocks(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.forward_features(x))

    def forward_with_embedding(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Devuelve logits y embedding de 128 canales ya existente antes del clasificador."""
        temporal = self.forward_features(x)
        pooled = self.head[1](self.head[0](temporal))
        embedding = self.head[3](self.head[2](pooled))
        logits = self.head[5](self.head[4](embedding))
        return logits, embedding

class TemporalCovariancePoolingTCN(TemporalTCN):
    """TCN con media y covarianza temporal compacta de sus estados profundos."""

    COVARIANCE_CHANNELS = 16

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if frames < 2:
            raise ValueError("TemporalCovariancePoolingTCN requiere al menos dos frames")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.covariance_projection = nn.Conv1d(channels, self.COVARIANCE_CHANNELS, kernel_size=1)
        indices = torch.triu_indices(self.COVARIANCE_CHANNELS, self.COVARIANCE_CHANNELS)
        self.register_buffer("covariance_upper_indices", indices, persistent=True)
        descriptor_dim = channels + self.COVARIANCE_CHANNELS * (self.COVARIANCE_CHANNELS + 1) // 2
        self.descriptor_dim = descriptor_dim
        self.head = nn.Sequential(
            nn.LayerNorm(descriptor_dim),
            nn.Linear(descriptor_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def temporal_covariance(self, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1] != self.covariance_projection.in_channels or temporal.shape[2] != self.frames:
            raise ValueError("Estados temporales incompatibles con pooling de covarianza")
        projected = F.gelu(self.covariance_projection(temporal))
        centered = projected - projected.mean(dim=2, keepdim=True)
        return centered.matmul(centered.transpose(1, 2)) / float(self.frames - 1)

    def descriptor(self, temporal: Tensor) -> Tensor:
        covariance = self.temporal_covariance(temporal)
        upper = covariance[:, self.covariance_upper_indices[0], self.covariance_upper_indices[1]]
        descriptor = torch.cat([temporal.mean(dim=2), upper], dim=1)
        if descriptor.shape[1] != self.descriptor_dim or not torch.isfinite(descriptor).all():
            raise FloatingPointError("Descriptor temporal de covarianza inválido")
        return descriptor

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.descriptor(self.forward_features(x)))

class SignerAdversarialTCN(TemporalTCN):
    """TCN léxica con cabeza GRL de firmante, prohibida fuera de entrenamiento."""

    SIGNER_CLASSES = 7

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("SignerAdversarialTCN requiere entrada recuperada (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.signer_head = nn.Linear(128, self.SIGNER_CLASSES)

    def forward_with_signer(self, x: Tensor, gradient_scale: float) -> tuple[Tensor, Tensor]:
        """Devuelve logits léxicos y de firmante; la GRL actúa solo en esta ruta de."""
        logits, embedding = self.forward_with_embedding(x)
        return logits, self.signer_head(reverse_gradient(embedding, gradient_scale))

class LowRankBidirectionalSSM(nn.Module):
    """Filtro temporal bidireccional estable de estado bajo, sin atención."""

    def __init__(self, channels: int, state_dim: int = 16):
        super().__init__()
        if channels < 1 or state_dim < 1:
            raise ValueError("channels y state_dim deben ser positivos")
        self.channels = channels
        self.state_dim = state_dim
        self.input_projection = nn.Conv1d(channels, state_dim, kernel_size=1, bias=False)
        self.output_projection = nn.Conv1d(state_dim, channels, kernel_size=1, bias=False)
        initial_ratio = torch.tensor(0.90 / 0.98, dtype=torch.float32)
        self.transition_logits = nn.Parameter(torch.logit(initial_ratio).repeat(state_dim))
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def transition(self) -> Tensor:
        """Devuelve coeficientes diagonalmente estables en el intervalo abierto `(0,0.98)`."""
        return 0.98 * torch.sigmoid(self.transition_logits)

    def _scan(self, projected: Tensor, reverse: bool) -> Tensor:
        if projected.ndim != 3:
            raise ValueError("La proyección SSM debe tener forma (B,H,T)")
        batch, states, frames = projected.shape
        state = projected.new_zeros((batch, states))
        ordered = range(frames - 1, -1, -1) if reverse else range(frames)
        outputs: list[Tensor] = []
        decay = self.transition().to(dtype=projected.dtype).view(1, states)
        for index in ordered:
            state = decay * state + projected[:, :, index]
            outputs.append(state)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=2)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"LowRankBidirectionalSSM esperaba (B,{self.channels},T), recibió {tuple(x.shape)}")
        projected = self.input_projection(x)
        temporal_state = self._scan(projected, reverse=False) + self._scan(projected, reverse=True)
        return x + self.residual_scale * self.output_projection(temporal_state)

class LowRankBiSSMTemporalTCN(TemporalTCN):
    """Control TCN con residual bidireccional de espacio de estados post-TCN."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20, state_dim: int = 16):
        if feature_dim != 126:
            raise ValueError("LowRankBiSSMTemporalTCN requiere entrada recuperada (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.temporal_ssm = LowRankBidirectionalSSM(channels=channels, state_dim=state_dim)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.temporal_ssm(super().forward_features(x))

class DeterministicEchoReservoir(nn.Module):
    """Reservorio fijo reproducible de tres fugas; solo su proyección residual se aprende."""

    def __init__(self, channels: int, state_dim: int = 8, leaks: tuple[float, ...] = (0.15, 0.50, 0.85), spectral_radius: float = 0.90):
        super().__init__()
        if channels < 1 or state_dim < 1 or not leaks:
            raise ValueError("El reservorio requiere canales, estado y al menos una fuga positivos")
        if any(not 0.0 < leak < 1.0 for leak in leaks) or not 0.0 < spectral_radius < 1.0:
            raise ValueError("Las fugas y el radio espectral deben pertenecer a (0,1)")
        self.channels = channels
        self.state_dim = state_dim
        self.leaks = tuple(float(leak) for leak in leaks)
        self.spectral_radius = float(spectral_radius)
        input_weights: list[Tensor] = []
        recurrent_weights: list[Tensor] = []
        for seed in (104_729, 130_363, 155_921)[:len(leaks)]:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            input_weight = torch.empty((state_dim, channels), dtype=torch.float32).uniform_(-0.5, 0.5, generator=generator)
            recurrent_weight = torch.empty((state_dim, state_dim), dtype=torch.float32).uniform_(-0.5, 0.5, generator=generator)
            current_radius = torch.linalg.eigvals(recurrent_weight).abs().max().real
            if current_radius <= 0.0:
                raise AssertionError("El reservorio determinista no puede tener radio nulo")
            input_weights.append(input_weight)
            recurrent_weights.append(recurrent_weight * (spectral_radius / current_radius))
        self.register_buffer('input_weights', torch.stack(input_weights))
        self.register_buffer('recurrent_weights', torch.stack(recurrent_weights))
        self.register_buffer('leak_values', torch.tensor(self.leaks, dtype=torch.float32))
        self.output_projection = nn.Conv1d(len(leaks) * state_dim, channels, kernel_size=1, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def max_spectral_radius(self) -> Tensor:
        radii = torch.linalg.eigvals(self.recurrent_weights).abs().amax(dim=1).real
        return radii.max()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"DeterministicEchoReservoir esperaba (B,{self.channels},T), recibió {tuple(x.shape)}")
        batch, _, frames = x.shape
        branches = self.input_weights.shape[0]
        states = x.new_zeros((batch, branches, self.state_dim))
        leak_values = self.leak_values.to(dtype=x.dtype).view(1, branches, 1)
        outputs: list[Tensor] = []
        for index in range(frames):
            drive = torch.einsum('rhc,bc->brh', self.input_weights.to(dtype=x.dtype), x[:, :, index])
            recurrence = torch.einsum('rij,brj->bri', self.recurrent_weights.to(dtype=x.dtype), states)
            states = (1.0 - leak_values) * states + leak_values * torch.tanh(drive + recurrence)
            outputs.append(states.reshape(batch, branches * self.state_dim))
        reservoir_sequence = torch.stack(outputs, dim=2)
        return x + self.residual_scale * self.output_projection(reservoir_sequence)

class DeterministicEchoReservoirTCN(TemporalTCN):
    """TCN de control más una única rama de reservorio temporal fijo y residual nulo."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("DeterministicEchoReservoirTCN requiere entrada recuperada (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.echo_reservoir = DeterministicEchoReservoir(channels=channels, state_dim=8, leaks=(0.15, 0.50, 0.85), spectral_radius=0.90)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.echo_reservoir(super().forward_features(x))

class LowRankTemporalRelation(nn.Module):
    """Una relación temporal contenido–contenido de bajo rango con sesgo por distancia."""

    def __init__(self, channels: int, rank: int = 16, value_dim: int = 32, frames: int = 30):
        super().__init__()
        if channels < 1 or rank < 1 or value_dim < 1 or frames < 1:
            raise ValueError("canales, rango, valor y frames deben ser positivos")
        self.channels = channels
        self.rank = rank
        self.value_dim = value_dim
        self.frames = frames
        self.normalization = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, rank)
        self.key = nn.Linear(channels, rank)
        self.value = nn.Linear(channels, value_dim)
        self.output = nn.Linear(value_dim, channels)
        self.relative_bias = nn.Parameter(torch.zeros(frames, dtype=torch.float32))
        self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def distance_indices(self, frames: int, device: torch.device) -> Tensor:
        if frames != self.frames:
            raise ValueError(f"La relación temporal espera {self.frames} frames, recibió {frames}")
        positions = torch.arange(frames, device=device)
        return (positions[:, None] - positions[None, :]).abs()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(f"LowRankTemporalRelation esperaba (B,{self.channels},T), recibió {tuple(x.shape)}")
        sequence = x.transpose(1, 2)
        normalized = self.normalization(sequence)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)
        scores = torch.matmul(query, key.transpose(1, 2)) / float(self.rank) ** 0.5
        scores = scores + self.relative_bias[self.distance_indices(sequence.shape[1], sequence.device)].unsqueeze(0)
        attention = torch.softmax(scores, dim=-1)
        related = self.output(torch.matmul(attention, value)).transpose(1, 2)
        return x + self.residual_scale * related

class LowRankTemporalRelationTCN(TemporalTCN):
    """TCN de control más una relación temporal de bajo rango y residual inicialmente nulo."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("LowRankTemporalRelationTCN requiere entrada recuperada (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.temporal_relation = LowRankTemporalRelation(channels=channels, rank=16, value_dim=32, frames=frames)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.temporal_relation(super().forward_features(x))

