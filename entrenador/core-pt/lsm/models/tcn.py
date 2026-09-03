"""Modelos PyTorch v2 para reconocimiento eficiente de LSM.

Las entradas usan siempre (batch, frames, features) para que el contrato sea
idéntico en entrenamiento, ONNX y runtime. Los modelos devuelven logits, no
probabilidades; CrossEntropyLoss aplica la normalización durante entrenamiento.
"""
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


def _check_sequence(x: Tensor, frames: int, features: int) -> None:
    if x.ndim != 3 or x.shape[1] != frames or x.shape[2] != features:
        raise ValueError(f"Se esperaba (batch,{frames},{features}), se recibió {tuple(x.shape)}")


class _ReverseGradient(Function):
    """Identidad en avance; invierte y escala la señal hacia el backbone."""

    @staticmethod
    def forward(ctx, values: Tensor, scale: float) -> Tensor:
        ctx.scale = float(scale)
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return -ctx.scale * gradient, None


def reverse_gradient(values: Tensor, scale: float) -> Tensor:
    if scale < 0:
        raise ValueError("La escala de reversión de gradiente debe ser no negativa")
    return _ReverseGradient.apply(values, float(scale))


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


def partial_temporal_convolution(x: Tensor, mask: Tensor, conv: nn.Conv1d) -> tuple[Tensor, Tensor]:
    """Convoluciona únicamente soporte temporal observado y propaga su máscara."""
    if mask.shape != (x.shape[0], 1, x.shape[2]):
        raise ValueError("La máscara temporal debe tener forma (batch,1,frames)")
    if conv.stride != (1,):
        raise ValueError("La convolución parcial solo admite stride temporal uno")
    support_kernel = torch.ones((1, 1, conv.kernel_size[0]), device=x.device, dtype=x.dtype)
    kwargs = {"stride": conv.stride, "padding": conv.padding, "dilation": conv.dilation}
    observed = F.conv1d(mask, support_kernel, **kwargs)
    expected = F.conv1d(torch.ones_like(mask), support_kernel, **kwargs)
    scale = torch.where(observed > 0.0, expected / observed.clamp_min(1.0), torch.zeros_like(observed))
    weighted = F.conv1d(x * mask, conv.weight, bias=None, **kwargs)
    output = weighted * scale
    if conv.bias is not None:
        output = output + conv.bias.view(1, -1, 1)
    return output, (observed > 0.0).to(dtype=x.dtype)


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
        """Devuelve logits léxicos y de firmante; la GRL actúa solo en esta ruta de train."""
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


class PartialTemporalConvTCN(TemporalTCN):
    """TCN sucesora con convoluciones temporales parciales sin parámetros nuevos."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("PartialTemporalConvTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.blocks = nn.ModuleList([PartialTemporalResidualBlock(channels, dilation, dropout) for dilation in dilations])

    def forward_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        temporal_mask = (x.abs().sum(dim=2, keepdim=False) > 0.0).to(dtype=x.dtype).unsqueeze(1)
        y, temporal_mask = partial_temporal_convolution(x.transpose(1, 2), temporal_mask, self.stem)
        y = F.gelu(y) * temporal_mask
        for block in self.blocks:
            y, temporal_mask = block.forward_with_mask(y, temporal_mask)
        return y


class DeformableTemporalOffsetsTCN(TemporalTCN):
    """TCN sucesora con offsets temporales locales aprendidos, iniciada como control."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("DeformableTemporalOffsetsTCN requiere entrada (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.blocks = nn.Sequential(*[DeformableTemporalOffsetResidualBlock(channels, dilation, dropout) for dilation in dilations])


def hadamard_codebook(order: int = 256) -> Tensor:
    """Construye códigos ortogonales ±1 de orden potencia de dos, como 0/1."""
    _require_torch()
    if order < 2 or order & (order - 1):
        raise ValueError("El orden Hadamard debe ser una potencia de dos >= 2")
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat([torch.cat([matrix, matrix], dim=1), torch.cat([matrix, -matrix], dim=1)], dim=0)
    return 0.5 * (matrix + 1.0)


class ECOCAuxiliaryTCN(TemporalTCN):
    """TCN sucesora con código Hadamard auxiliar exclusivo de entrenamiento."""

    CODE_LENGTH = 256

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("ECOCAuxiliaryTCN requiere landmarks manuales (30,126)")
        if classes > self.CODE_LENGTH:
            raise ValueError("ECOCAuxiliaryTCN requiere clases no mayores a 256")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.code_head = nn.Linear(channels, self.CODE_LENGTH)
        self.register_buffer("ecoc_codebook", hadamard_codebook(self.CODE_LENGTH), persistent=True)

    def ecoc_targets(self, targets: Tensor) -> Tensor:
        if targets.ndim != 1 or targets.numel() == 0:
            raise ValueError("Los objetivos ECOC deben ser un vector no vacío")
        if int(targets.min()) < 0 or int(targets.max()) >= self.head[-1].out_features:
            raise ValueError("Objetivo fuera del rango de clases ECOC")
        return self.ecoc_codebook.index_select(0, targets)

    def forward_with_ecoc(self, x: Tensor) -> tuple[Tensor, Tensor]:
        temporal = self.forward_features(x)
        return self.head(temporal), self.code_head(temporal.mean(dim=-1))


class SharedBilateralTCN(_BaseModule):
    """TCN sucesora con un codificador temporal compartido entre manos.

    La fusión conserva simetría, diferencia firmada y presencia para distinguir
    una mano ausente de una configuración geométrica con valores pequeños.
    """

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, dropout: float = 0.20):
        _require_torch()
        super().__init__()
        if feature_dim != 126:
            raise ValueError("SharedBilateralTCN requiere landmarks manuales (30,126)")
        self.frames = frames
        self.features = feature_dim
        self.hand_stem = nn.Conv1d(63, 32, kernel_size=3, padding=1)
        self.hand_blocks = nn.Sequential(*[TCNResidualBlock(32, dilation, dropout) for dilation in (1, 2, 4, 8)])
        # media (32), diferencia absoluta (32), diferencia firmada (32), dos máscaras.
        self.fusion_stem = nn.Conv1d(98, 64, kernel_size=1)
        self.fusion_blocks = nn.Sequential(*[TCNResidualBlock(64, dilation, dropout) for dilation in (1, 2)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def encode_hand(self, hand: Tensor) -> Tensor:
        if hand.ndim != 3 or hand.shape[1] != self.frames or hand.shape[2] != 63:
            raise ValueError(f"Se esperaba mano (batch,{self.frames},63), se recibió {tuple(hand.shape)}")
        return self.hand_blocks(F.gelu(self.hand_stem(hand.transpose(1, 2))))

    @staticmethod
    def hand_presence(hand: Tensor) -> Tensor:
        return (hand.abs().sum(dim=2, keepdim=False) > 0.0).to(dtype=hand.dtype).unsqueeze(1)

    def fuse_bilateral(self, left: Tensor, right: Tensor) -> Tensor:
        left_features = self.encode_hand(left)
        right_features = self.encode_hand(right)
        left_present = self.hand_presence(left)
        right_present = self.hand_presence(right)
        return torch.cat(
            [
                0.5 * (left_features + right_features),
                (left_features - right_features).abs(),
                left_features - right_features,
                left_present,
                right_present,
            ],
            dim=1,
        )

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        bilateral = self.fuse_bilateral(x[:, :, :63], x[:, :, 63:])
        return self.fusion_blocks(F.gelu(self.fusion_stem(bilateral)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.temporal_features(x))


def fixed_hand_adjacency() -> Tensor:
    """Matriz de 21 joints: cinco cadenas de dedos conectadas a muñeca."""
    _require_torch()
    adjacency = torch.zeros((21, 21), dtype=torch.float32)
    chains = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
    for chain in chains:
        for source, target in zip(chain, chain[1:]):
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0
    adjacency.fill_diagonal_(1.0)
    return adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)


class FixedHandGraphTCN(_BaseModule):
    """Grafo espacial fijo por mano seguido de TCN; no aprende adyacencia."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, dropout: float = 0.20):
        _require_torch()
        super().__init__()
        if feature_dim != 126:
            raise ValueError("FixedHandGraphTCN requiere landmarks manuales (30,126)")
        self.frames = frames
        self.features = feature_dim
        self.register_buffer("adjacency", fixed_hand_adjacency(), persistent=True)
        self.graph1 = nn.Linear(3, 16, bias=False)
        self.graph2 = nn.Linear(16, 32, bias=False)
        self.stem = nn.Conv1d(66, 64, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[TCNResidualBlock(64, dilation, dropout) for dilation in (1, 2, 4, 8)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def graph_encode_hand(self, hand: Tensor) -> Tensor:
        if hand.ndim != 3 or hand.shape[1] != self.frames or hand.shape[2] != 63:
            raise ValueError(f"Se esperaba mano (batch,{self.frames},63), se recibió {tuple(hand.shape)}")
        joints = hand.reshape(hand.shape[0], self.frames, 21, 3)
        present = (joints.abs().sum(dim=(2, 3), keepdim=False) > 0.0).to(dtype=hand.dtype)
        spatial = torch.einsum("ij,btjf->btif", self.adjacency, joints)
        spatial = F.gelu(self.graph1(spatial))
        spatial = torch.einsum("ij,btjf->btif", self.adjacency, spatial)
        spatial = F.gelu(self.graph2(spatial)).mean(dim=2)
        return spatial * present.unsqueeze(-1)

    @staticmethod
    def hand_presence(hand: Tensor) -> Tensor:
        return (hand.abs().sum(dim=2, keepdim=False) > 0.0).to(dtype=hand.dtype).unsqueeze(-1)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        left, right = x[:, :, :63], x[:, :, 63:]
        features = torch.cat([self.graph_encode_hand(left), self.graph_encode_hand(right), self.hand_presence(left), self.hand_presence(right)], dim=2)
        return self.blocks(F.gelu(self.stem(features.transpose(1, 2))))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.temporal_features(x))


def arc_length_frame_reindex(x: Tensor, frames: int = 30, features: int = 126, epsilon: float = 1e-12) -> Tensor:
    """Selecciona frames observados a cuantiles uniformes de longitud acumulada."""
    _check_sequence(x, frames, features)
    steps = torch.linalg.vector_norm(x[:, 1:] - x[:, :-1], dim=2)
    cumulative = torch.cat([torch.zeros_like(steps[:, :1]), torch.cumsum(steps, dim=1)], dim=1)
    total = cumulative[:, -1:]
    normalized = cumulative / total.clamp_min(float(epsilon))
    targets = torch.linspace(0.0, 1.0, frames, dtype=x.dtype, device=x.device).unsqueeze(0).expand(x.shape[0], -1)
    indices = torch.searchsorted(normalized.contiguous(), targets.contiguous(), right=False).clamp_max(frames - 1)
    gathered = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, features))
    return torch.where((total > float(epsilon)).unsqueeze(-1), gathered, x)


class ArcLengthFrameReindexingTCN(TemporalTCN):
    """TCN de control precedida por selección discreta de frames por arco."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("ArcLengthFrameReindexingTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)

    def forward_features(self, x: Tensor) -> Tensor:
        return super().forward_features(arc_length_frame_reindex(x, self.frames, self.features))


class BidirectionalGRUClassifier(_BaseModule):
    """Codificador recurrente de bajo presupuesto que conserva orden completo."""

    HIDDEN_SIZE = 64

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, dropout: float = 0.20):
        _require_torch()
        super().__init__()
        if feature_dim != 126:
            raise ValueError("BidirectionalGRUClassifier requiere landmarks manuales (30,126)")
        self.frames = frames
        self.features = feature_dim
        self.gru = nn.GRU(feature_dim, self.HIDDEN_SIZE, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * self.HIDDEN_SIZE, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def temporal_context(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        _, hidden = self.gru(x)
        return torch.cat([hidden[-2], hidden[-1]], dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.temporal_context(x))


class CosineClassifierTCN(TemporalTCN):
    """TCN de control con decisión angular y escala positiva aprendible."""

    INITIAL_SCALE = 8.0

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("CosineClassifierTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.head = nn.Identity()
        self.projection = nn.Sequential(nn.Linear(channels, 128), nn.GELU(), nn.Dropout(dropout))
        self.class_weights = nn.Parameter(torch.empty(classes, 128))
        nn.init.xavier_uniform_(self.class_weights)
        self.scale_raw = nn.Parameter(torch.tensor(self.INITIAL_SCALE, dtype=torch.float32))

    def normalized_features(self, x: Tensor) -> Tensor:
        temporal = self.forward_features(x)
        pooled = F.adaptive_avg_pool1d(temporal, 1).flatten(1)
        return F.normalize(self.projection(pooled), p=2.0, dim=1)

    def scale(self) -> Tensor:
        return F.softplus(self.scale_raw)

    def forward(self, x: Tensor) -> Tensor:
        return self.scale() * F.linear(self.normalized_features(x), F.normalize(self.class_weights, p=2.0, dim=1))


class SpectralTemporalTCN(TemporalTCN):
    """TCN de control con norma espectral en todas las capas lineales."""

    def __init__(self, *args, **kwargs):
        _require_torch()
        super().__init__(*args, **kwargs)
        spectral_norm = torch.nn.utils.parametrizations.spectral_norm
        for module in list(self.modules()):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                spectral_norm(module, name="weight", n_power_iterations=1)


class MaskedHandReconstructionTCN(TemporalTCN):
    """TCN sucesora con decodificador auxiliar de landmarks manuales.

    La predicción auxiliar solo se invoca durante el entrenamiento desde el
    bucle correspondiente. ``forward`` conserva el contrato puro de logits
    para validation, test, exportación y despliegue.
    """

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("MaskedHandReconstructionTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(channels, feature_dim, kernel_size=1),
        )

    def forward_with_reconstruction(self, x: Tensor) -> tuple[Tensor, Tensor]:
        temporal = self.forward_features(x)
        return self.head(temporal), self.reconstruction_head(temporal).transpose(1, 2)


class IntraClipStyleNormalizationTCN(TemporalTCN):
    """TCN sucesora con corrección residual de traslación y escala intra-clip.

    La referencia es la mediana de escala medida exclusivamente en S01--S07.
    Cada clip se normaliza de forma independiente y los landmarks no detectados
    continúan siendo exactamente cero. La mezcla inicia en identidad.
    """

    TRAIN_REFERENCE_SCALE = 1.1359084844589233
    EPSILON = 1e-6

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("IntraClipStyleNormalizationTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.style_alpha = nn.Parameter(torch.zeros(()))

    @classmethod
    def normalized_view(cls, x: Tensor) -> Tensor:
        _check_sequence(x, 30, 126)
        hands = x.reshape(x.shape[0], 30, 2, 21, 3)
        valid = torch.any(torch.abs(hands) > 1e-8, dim=(-1, -2))
        weights = valid[:, :, :, None, None].to(dtype=x.dtype)
        support = (valid.sum(dim=(1, 2), keepdim=True).to(dtype=x.dtype) * 21.0).clamp_min(1.0)
        center = (hands * weights).sum(dim=(1, 2, 3), keepdim=True) / support[:, :, :, None, None]
        centered = (hands - center) * weights
        coordinate_support = (support * 3.0).reshape(x.shape[0], 1, 1, 1, 1)
        scale = torch.sqrt(centered.square().sum(dim=(1, 2, 3, 4), keepdim=True) / coordinate_support).clamp_min(cls.EPSILON)
        normalized = centered * (cls.TRAIN_REFERENCE_SCALE / scale)
        return normalized.reshape_as(x)

    def style_corrected_input(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        normalized = self.normalized_view(x)
        return x + torch.tanh(self.style_alpha) * (normalized - x)

    def forward_features(self, x: Tensor) -> Tensor:
        return super().forward_features(self.style_corrected_input(x))


class FixedTemporalPyramidPool(_BaseModule):
    """Concatena promedios temporales de la secuencia, mitades y tercios."""

    LEVELS = (1, 2, 3)

    def __init__(self, channels: int):
        super().__init__()
        self.channels = int(channels)

    @property
    def output_features(self) -> int:
        return self.channels * sum(self.LEVELS)

    def forward(self, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1] != self.channels:
            raise ValueError(f"La pirámide temporal requiere (batch,{self.channels},frames), se recibió {tuple(temporal.shape)}")
        summaries = [F.adaptive_avg_pool1d(temporal, level) for level in self.LEVELS]
        return torch.cat(summaries, dim=2).flatten(start_dim=1)


class TemporalPyramidPoolingTCN(TemporalTCN):
    """TCN sucesora que conserva fases temporales mediante pooling 1/2/3."""

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("TemporalPyramidPoolingTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.temporal_pyramid = FixedTemporalPyramidPool(channels)
        self.head = nn.Sequential(
            nn.Linear(self.temporal_pyramid.output_features, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.temporal_pyramid(self.forward_features(x)))


class LogAvgExpTemporalPoolingTCN(TemporalTCN):
    """TCN sucesora con mezcla residual entre promedio y LogAvgExp temporal."""

    TEMPERATURE = 1.0

    def __init__(self, feature_dim: int, classes: int, frames: int = 30, channels: int = 64, dilations: Iterable[int] = (1, 2, 4, 8), dropout: float = 0.20):
        if feature_dim != 126:
            raise ValueError("LogAvgExpTemporalPoolingTCN requiere landmarks manuales (30,126)")
        super().__init__(feature_dim, classes, frames=frames, channels=channels, dilations=dilations, dropout=dropout)
        self.logavgexp_alpha = nn.Parameter(torch.zeros(()))

    @classmethod
    def logavgexp_pool(cls, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3:
            raise ValueError("LogAvgExp requiere rasgos temporales (batch,canales,frames)")
        frames = temporal.shape[-1]
        if frames < 1:
            raise ValueError("LogAvgExp requiere al menos un frame")
        temperature = temporal.new_tensor(cls.TEMPERATURE)
        return temperature * (torch.logsumexp(temporal / temperature, dim=-1) - math.log(frames))

    def pooled_features(self, temporal: Tensor) -> Tensor:
        average = temporal.mean(dim=-1)
        logavgexp = self.logavgexp_pool(temporal)
        return average + torch.tanh(self.logavgexp_alpha) * (logavgexp - average)

    def forward(self, x: Tensor) -> Tensor:
        pooled = self.pooled_features(self.forward_features(x))
        return self.head(pooled.unsqueeze(-1))


class ChronologicalRolePool(_BaseModule):
    """Resume una secuencia conservando roles de inicio, núcleo y cierre.

    A diferencia de ``AdaptiveAvgPool1d``, la inversión temporal puede cambiar
    la salida porque cada rol tiene un peso aprendido independiente. Añade solo
    cuatro pesos de mezcla y una atención posicional ligera, no otra red TCN.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.position = nn.Sequential(nn.Linear(1, channels), nn.Tanh())
        self.score = nn.Conv1d(channels, 1, kernel_size=1)
        self.role_logits = nn.Parameter(torch.zeros(4))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"Se esperaba (batch,channels,frames), se recibió {tuple(features.shape)}")
        batch, channels, frames = features.shape
        indices = torch.arange(frames, device=features.device, dtype=features.dtype)
        denominator = torch.as_tensor(max(1, frames - 1), device=features.device, dtype=features.dtype)
        time = (2.0 * indices / denominator - 1.0).view(1, frames, 1)
        position = self.position(time).transpose(1, 2).expand(batch, -1, -1)
        positioned = features + position
        attention = torch.softmax(self.score(positioned), dim=-1)
        attended = (features * attention).sum(dim=-1)
        roles = torch.stack([attended, features[:, :, 0], features[:, :, frames // 2], features[:, :, -1]], dim=1)
        weights = torch.softmax(self.role_logits, dim=0).view(1, 4, 1)
        return (roles * weights).sum(dim=1)


class ResidualChronologicalPool(_BaseModule):
    """Promedio temporal más una corrección cronológica con compuerta residual.

    La proyección correctiva inicia en cero, de manera que el modelo es
    exactamente equivalente al promedio temporal de W3 al primer paso. La
    compuerta queda activa para que la corrección aprenda sin inyectar ruido
    aleatorio en la primera actualización.
    """

    def __init__(self, channels: int, phase_frames: tuple[int, int, int] = (6, 18, 6)):
        super().__init__()
        if sum(phase_frames) < 3 or min(phase_frames) < 1:
            raise ValueError("phase_frames debe tener tres segmentos positivos")
        self.phase_frames = phase_frames
        self.correction = nn.Sequential(
            nn.Linear(channels * 3, channels, bias=False),
            nn.GELU(),
            nn.Linear(channels, channels, bias=False),
        )
        nn.init.zeros_(self.correction[-1].weight)
        self.gate = nn.Parameter(torch.ones(()))

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"Se esperaba (batch,channels,frames), se recibió {tuple(features.shape)}")
        _, _, frames = features.shape
        a, b, c = self.phase_frames
        if frames != a + b + c:
            boundaries = (max(1, round(frames * 0.2)), max(1, round(frames * 0.6)))
            a, b = boundaries
            c = frames - a - b
        start = features[:, :, :a].mean(dim=-1)
        middle = features[:, :, a:a + b].mean(dim=-1)
        end = features[:, :, a + b:a + b + c].mean(dim=-1)
        average = features.mean(dim=-1)
        phase_delta = torch.cat([start - average, middle - average, end - average], dim=1)
        return average + self.gate * self.correction(phase_delta)


class DynamicLettersTCN(TemporalTCN):
    def __init__(self, frames: int = 30, feature_dim: int = 93, classes: int = DYNAMIC_LETTERS, dropout: float = 0.20):
        super().__init__(feature_dim, classes, frames=frames, channels=64, dropout=dropout)


class IsolatedWordsTCN(_BaseModule):
    def __init__(self, frames: int = 30, feature_dim: int = 226, classes: int = 200, dropout: float = 0.25):
        _require_torch()
        super().__init__()
        if feature_dim not in {226, 352, 353, 354, 478}:
            raise ValueError("IsolatedWordsTCN requiere feature_dim=226, 352 (PTA-LSM), 353 (W60/W61), 354 (W62) o 478 (PTA-LSM+aceleración)")
        self.frames = frames
        self.features = feature_dim
        self.base_features = feature_dim
        self.hand_input_dim = {226: 126, 352: 252, 353: 253, 354: 254, 478: 378}[feature_dim]
        self.hands = nn.Sequential(nn.Conv1d(self.hand_input_dim, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU())
        self.pose = nn.Sequential(nn.Conv1d(52, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.face = nn.Sequential(nn.Conv1d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.fusion = nn.Conv1d(192, 128, 1)
        self.blocks = nn.Sequential(*[TCNResidualBlock(128, dilation, dropout) for dilation in (1, 2, 4, 8)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, classes),
        )

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.base_features)
        x = x.transpose(1, 2)
        hand_block = x[:, :126, :]
        if self.base_features >= 352:
            hand_block = torch.cat([hand_block, x[:, 226:352, :]], dim=1)
        if self.base_features == 353:
            hand_block = torch.cat([hand_block, x[:, 352:353, :]], dim=1)
        if self.base_features == 354:
            hand_block = torch.cat([hand_block, x[:, 352:354, :]], dim=1)
        if self.base_features == 478:
            hand_block = torch.cat([hand_block, x[:, 352:478, :]], dim=1)
        hands = self.hands(hand_block)
        pose = self.pose(x[:, 126:178, :])
        face = self.face(x[:, 178:226, :])
        return self.blocks(F.gelu(self.fusion(torch.cat([hands, pose, face], dim=1))))

    def logits_from_temporal_features(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 128:
            raise ValueError(f"Se esperaba rasgos temporales (batch,128,frames), se recibió {tuple(features.shape)}")
        return self.head(features)

    def forward(self, x: Tensor) -> Tensor:
        return self.logits_from_temporal_features(self.temporal_features(x))


class BimanualCouplingResidualTCN(IsolatedWordsTCN):
    """W94: W33 más corrección residual desde cuatro estadísticas bimanuales factuales."""

    def __init__(self, frames: int = 30, feature_dim: int = 356, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 356:
            raise ValueError("BimanualCouplingResidualTCN requiere W94 (30,356)")
        super().__init__(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.features = 356
        self.coupling_residual = nn.Sequential(
            nn.Conv1d(4, 16, 1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 128, 1, bias=False),
        )
        nn.init.zeros_(self.coupling_residual[-1].weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        base = super().temporal_features(x[:, :, :352])
        correction = self.coupling_residual(x[:, :, 352:356].transpose(1, 2))
        return base + correction


class TrajectoryTopologyResidualTCN(IsolatedWordsTCN):
    """W95: W33 más una corrección residual de topología H0-MST temporal."""

    def __init__(self, frames: int = 30, feature_dim: int = 368, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 368:
            raise ValueError("TrajectoryTopologyResidualTCN requiere W95 (30,368)")
        super().__init__(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.features = 368
        self.topology_residual = nn.Sequential(
            nn.Conv1d(16, 32, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 128, 1, bias=False),
        )
        nn.init.zeros_(self.topology_residual[-1].weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        base = super().temporal_features(x[:, :, :352])
        correction = self.topology_residual(x[:, :, 352:368].transpose(1, 2))
        return base + correction


class TemporalSelfSimilarityResidualTCN(IsolatedWordsTCN):
    """W96: W33 más corrección residual de auto-similitud de configuración temporal."""

    def __init__(self, frames: int = 30, feature_dim: int = 380, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 380:
            raise ValueError("TemporalSelfSimilarityResidualTCN requiere W96 (30,380)")
        super().__init__(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.features = 380
        self.self_similarity_residual = nn.Sequential(
            nn.Conv1d(28, 32, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 128, 1, bias=False),
        )
        nn.init.zeros_(self.self_similarity_residual[-1].weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        base = super().temporal_features(x[:, :, :352])
        correction = self.self_similarity_residual(x[:, :, 352:380].transpose(1, 2))
        return base + correction


class FingertipTurningCurvatureResidualTCN(IsolatedWordsTCN):
    """W97: W33 más corrección residual desde curvatura interna de puntas."""

    def __init__(self, frames: int = 30, feature_dim: int = 376, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 376:
            raise ValueError("FingertipTurningCurvatureResidualTCN requiere W97 (30,376)")
        super().__init__(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.features = 376
        self.turning_residual = nn.Sequential(
            nn.Conv1d(24, 32, 1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 128, 1, bias=False),
        )
        nn.init.zeros_(self.turning_residual[-1].weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        return super().temporal_features(x[:, :, :352]) + self.turning_residual(x[:, :, 352:376].transpose(1, 2))


class FingerExtensionPermutationEntropyResidualTCN(IsolatedWordsTCN):
    """W98: W33 más residual de complejidad ordinal de extensiones."""

    def __init__(self, frames: int = 30, feature_dim: int = 362, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 362:
            raise ValueError("FingerExtensionPermutationEntropyResidualTCN requiere W98 (30,362)")
        super().__init__(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.features = 362
        self.permutation_entropy_residual = nn.Sequential(
            nn.Conv1d(10, 16, 1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 128, 1, bias=False),
        )
        nn.init.zeros_(self.permutation_entropy_residual[-1].weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        return super().temporal_features(x[:, :, :352]) + self.permutation_entropy_residual(x[:, :, 352:362].transpose(1, 2))


class AdaptiveTemporalPrototypeWordsTCN(IsolatedWordsTCN):
    """W73: W33 más clasificación directa contra prototipos temporales entrenables.

    La ruta base permanece exacta mientras ``prototype_alpha`` vale cero. Los prototipos
    son parámetros del checkpoint, no medias externas ni soportes consultados en inferencia.
    """

    PROTOTYPE_FRAMES = 6
    SOFT_DTW_GAMMA = 0.10

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 352:
            raise ValueError("AdaptiveTemporalPrototypeWordsTCN requiere W33 (30,352)")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.prototypes = nn.Parameter(torch.empty(classes, self.PROTOTYPE_FRAMES, 128))
        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)
        self.prototype_alpha = nn.Parameter(torch.zeros(()))

    def soft_dtw_distances(self, temporal: Tensor) -> Tensor:
        """Devuelve distancias Soft-DTW (batch,clases) con costo coseno, sin datos externos."""
        if temporal.ndim != 3 or temporal.shape[1:] != (128, self.frames):
            raise ValueError("W73 requiere temporales W33 (batch,128,30)")
        sequence = F.normalize(temporal.transpose(1, 2), p=2, dim=-1)
        prototypes = F.normalize(self.prototypes, p=2, dim=-1)
        costs = 1.0 - torch.einsum("btd,kpd->bktp", sequence, prototypes)
        batch, classes, frames, prototype_frames = costs.shape
        infinity = costs.new_full((batch, classes), float("inf"))
        previous = [costs.new_zeros((batch, classes))] + [infinity] * prototype_frames
        for frame_index in range(frames):
            current = [infinity]
            for prototype_index in range(prototype_frames):
                candidates = torch.stack([
                    previous[prototype_index],
                    previous[prototype_index + 1],
                    current[prototype_index],
                ], dim=0)
                soft_minimum = -self.SOFT_DTW_GAMMA * torch.logsumexp(-candidates / self.SOFT_DTW_GAMMA, dim=0)
                current.append(costs[:, :, frame_index, prototype_index] + soft_minimum)
            previous = current
        distances = previous[-1]
        if not torch.isfinite(distances).all():
            raise FloatingPointError("W73 produjo distancias Soft-DTW no finitas")
        return distances

    def prototype_logits_from_temporal_features(self, temporal: Tensor) -> Tensor:
        distances = self.soft_dtw_distances(temporal)
        raw_logits = -distances
        return (raw_logits - raw_logits.mean(dim=1, keepdim=True)) / (raw_logits.std(dim=1, keepdim=True, unbiased=False) + 1e-6)

    def forward_with_prototype_logits(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        temporal = self.temporal_features(x)
        base_logits = self.logits_from_temporal_features(temporal)
        prototype_logits = self.prototype_logits_from_temporal_features(temporal)
        return base_logits + self.prototype_alpha * prototype_logits, base_logits, prototype_logits

    def forward(self, x: Tensor) -> Tensor:
        logits, _base_logits, _prototype_logits = self.forward_with_prototype_logits(x)
        return logits


class CompositionalTemporalPrototypeWordsTCN(IsolatedWordsTCN):
    """W75: Soft-DTW contra prototipos compuestos por átomos compartidos.

    Reduce la capacidad de W73: cada una de las clases usa una combinación convexa de
    pocos átomos temporales globales. La rama inicia anulada para preservar W33 exacto.
    """

    PROTOTYPE_FRAMES = 6
    PROTOTYPE_ATOMS = 16
    SOFT_DTW_GAMMA = 0.10

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 352:
            raise ValueError("CompositionalTemporalPrototypeWordsTCN requiere W33 (30,352)")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.prototype_atoms = nn.Parameter(torch.empty(self.PROTOTYPE_ATOMS, self.PROTOTYPE_FRAMES, 128))
        self.class_atom_logits = nn.Parameter(torch.empty(classes, self.PROTOTYPE_ATOMS))
        nn.init.normal_(self.prototype_atoms, mean=0.0, std=0.02)
        nn.init.normal_(self.class_atom_logits, mean=0.0, std=0.02)
        self.prototype_alpha = nn.Parameter(torch.zeros(()))

    def class_atom_weights(self) -> Tensor:
        return torch.softmax(self.class_atom_logits, dim=1)

    def composed_prototypes(self) -> Tensor:
        return torch.einsum("km,mpd->kpd", self.class_atom_weights(), self.prototype_atoms)

    def prototype_parameter_count(self) -> int:
        return int(self.prototype_atoms.numel() + self.class_atom_logits.numel())

    def soft_dtw_distances(self, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1:] != (128, self.frames):
            raise ValueError("W75 requiere temporales W33 (batch,128,30)")
        sequence = F.normalize(temporal.transpose(1, 2), p=2, dim=-1)
        prototypes = F.normalize(self.composed_prototypes(), p=2, dim=-1)
        costs = 1.0 - torch.einsum("btd,kpd->bktp", sequence, prototypes)
        batch, classes, frames, prototype_frames = costs.shape
        infinity = costs.new_full((batch, classes), float("inf"))
        previous = [costs.new_zeros((batch, classes))] + [infinity] * prototype_frames
        for frame_index in range(frames):
            current = [infinity]
            for prototype_index in range(prototype_frames):
                candidates = torch.stack([
                    previous[prototype_index],
                    previous[prototype_index + 1],
                    current[prototype_index],
                ], dim=0)
                soft_minimum = -self.SOFT_DTW_GAMMA * torch.logsumexp(-candidates / self.SOFT_DTW_GAMMA, dim=0)
                current.append(costs[:, :, frame_index, prototype_index] + soft_minimum)
            previous = current
        distances = previous[-1]
        if not torch.isfinite(distances).all():
            raise FloatingPointError("W75 produjo distancias Soft-DTW no finitas")
        return distances

    def prototype_logits_from_temporal_features(self, temporal: Tensor) -> Tensor:
        raw_logits = -self.soft_dtw_distances(temporal)
        return (raw_logits - raw_logits.mean(dim=1, keepdim=True)) / (raw_logits.std(dim=1, keepdim=True, unbiased=False) + 1e-6)

    def forward_with_prototype_logits(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        temporal = self.temporal_features(x)
        base_logits = self.logits_from_temporal_features(temporal)
        prototype_logits = self.prototype_logits_from_temporal_features(temporal)
        return base_logits + self.prototype_alpha * prototype_logits, base_logits, prototype_logits

    def forward(self, x: Tensor) -> Tensor:
        logits, _base_logits, _prototype_logits = self.forward_with_prototype_logits(x)
        return logits


class ShapeMotionBilinearWordsTCN(IsolatedWordsTCN):
    """W78: residual de rango bajo para interacción explícita forma×velocidad W33."""

    INTERACTION_RANK = 16
    SHAPE_SLICE = slice(126, 226)
    VELOCITY_SLICE = slice(266, 352)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if frames != 30 or feature_dim != 352:
            raise ValueError("ShapeMotionBilinearWordsTCN requiere W33 (30,352)")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.shape_projection = nn.Linear(100, self.INTERACTION_RANK, bias=False)
        self.velocity_projection = nn.Linear(86, self.INTERACTION_RANK, bias=False)
        self.interaction_projection = nn.Linear(self.INTERACTION_RANK, 128, bias=True)
        self.interaction_alpha = nn.Parameter(torch.zeros(()))

    def interaction_features(self, x: Tensor) -> Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.frames, 352):
            raise ValueError("W78 requiere entrada W33 (batch,30,352)")
        shape = x[:, :, self.SHAPE_SLICE]
        velocity = x[:, :, self.VELOCITY_SLICE]
        interaction = self.shape_projection(shape) * self.velocity_projection(velocity)
        features = self.interaction_projection(interaction).transpose(1, 2)
        if not torch.isfinite(features).all():
            raise FloatingPointError("W78 produjo interacción no finita")
        return features

    def interaction_parameter_count(self) -> int:
        return sum(parameter.numel() for module in (self.shape_projection, self.velocity_projection, self.interaction_projection) for parameter in module.parameters())

    def temporal_features(self, x: Tensor) -> Tensor:
        base_temporal = super().temporal_features(x)
        return base_temporal + self.interaction_alpha * self.interaction_features(x)


class ParallelReceptiveFieldWordsTCN(IsolatedWordsTCN):
    """W64: reemplaza solo los bloques temporales W33 por ramas d y 2d."""

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("ParallelReceptiveFieldWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.blocks = nn.Sequential(*[
            ParallelReceptiveFieldTCNResidualBlock(128, dilation, dropout)
            for dilation in self.DILATIONS
        ])


class ParameterFreeTemporalShiftWordsTCN(IsolatedWordsTCN):
    """W65: mezcla de vecinos temporales sin parámetros antes de cada bloque W33."""

    FOLD = 16

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("ParameterFreeTemporalShiftWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)

    @classmethod
    def temporal_shift(cls, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1] != 128:
            raise ValueError("W65 requiere rasgos temporales (batch,128,frames)")
        if cls.FOLD * 2 >= temporal.shape[1]:
            raise ValueError("W65 requiere al menos dos folds y canales no desplazados")
        shifted = torch.zeros_like(temporal)
        shifted[:, :cls.FOLD, 1:] = temporal[:, :cls.FOLD, :-1]
        shifted[:, cls.FOLD:2 * cls.FOLD, :-1] = temporal[:, cls.FOLD:2 * cls.FOLD, 1:]
        shifted[:, 2 * cls.FOLD:, :] = temporal[:, 2 * cls.FOLD:, :]
        return shifted

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        channels = x.transpose(1, 2)
        hand_block = torch.cat([channels[:, :126, :], channels[:, 226:352, :]], dim=1)
        hands = self.hands(hand_block)
        pose = self.pose(channels[:, 126:178, :])
        face = self.face(channels[:, 178:226, :])
        temporal = F.gelu(self.fusion(torch.cat([hands, pose, face], dim=1)))
        for block in self.blocks:
            temporal = block(self.temporal_shift(temporal))
        return temporal


class WeightStandardizedTCNResidualBlock(TCNResidualBlock):
    """W66: bloque W33 con filtros Conv1d estandarizados en cada forward."""

    EPSILON = 1e-5

    @classmethod
    def standardized_weight(cls, weight: Tensor) -> Tensor:
        if weight.ndim != 3:
            raise ValueError("W66 requiere pesos Conv1d (out,in,kernel)")
        mean = weight.mean(dim=(1, 2), keepdim=True)
        variance = weight.var(dim=(1, 2), unbiased=False, keepdim=True)
        return (weight - mean) / torch.sqrt(variance + cls.EPSILON)

    @classmethod
    def standardized_conv(cls, x: Tensor, convolution: nn.Conv1d) -> Tensor:
        return F.conv1d(
            x,
            cls.standardized_weight(convolution.weight),
            convolution.bias,
            convolution.stride,
            convolution.padding,
            convolution.dilation,
            convolution.groups,
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.standardized_conv(x, self.conv1)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        y = self.norm2(self.standardized_conv(y, self.conv2))
        return F.gelu(x + y)


class TemporalWeightStandardizedWordsTCN(IsolatedWordsTCN):
    """W66: W33 con Weight Standardization limitada a las ocho convs temporales."""

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("TemporalWeightStandardizedWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.blocks = nn.Sequential(*[
            WeightStandardizedTCNResidualBlock(128, dilation, dropout)
            for dilation in self.DILATIONS
        ])


class DepthwiseSeparableTCNResidualBlock(_BaseModule):
    """W67: bloque residual temporal con filtrado depthwise y mezcla pointwise."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        if dilation < 1:
            raise ValueError("W67 requiere dilatación positiva")
        self.dilation = int(dilation)
        self.depthwise1 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False)
        self.pointwise1 = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.depthwise2 = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, groups=channels, bias=False)
        self.pointwise2 = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.dropout = nn.Dropout(dropout)

    def separable_transform(self, x: Tensor, stage: int) -> Tensor:
        if stage == 1:
            return self.pointwise1(self.depthwise1(x))
        if stage == 2:
            return self.pointwise2(self.depthwise2(x))
        raise ValueError("W67 stage debe ser 1 o 2")

    def forward(self, x: Tensor) -> Tensor:
        y = F.gelu(self.norm1(self.separable_transform(x, 1)))
        y = self.dropout(y)
        y = self.norm2(self.separable_transform(y, 2))
        return F.gelu(x + y)


class DepthwiseSeparableTemporalWordsTCN(IsolatedWordsTCN):
    """W67: W33 con bloques temporales depthwise-separable y mismos canales."""

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("DepthwiseSeparableTemporalWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.blocks = nn.Sequential(*[
            DepthwiseSeparableTCNResidualBlock(128, dilation, dropout)
            for dilation in self.DILATIONS
        ])


class StochasticDepthTCNResidualBlock(TCNResidualBlock):
    """W68: rama residual W33 con supervivencia estocástica solo en train."""

    def __init__(self, channels: int, dilation: int, dropout: float, survival_probability: float):
        if not 0.0 < survival_probability <= 1.0:
            raise ValueError("W68 requiere supervivencia en (0,1]")
        super().__init__(channels, dilation, dropout)
        self.survival_probability = float(survival_probability)

    def residual_branch(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        return self.norm2(self.conv2(y))

    def stochastic_mask(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError("W68 requiere rasgos temporales (batch,channels,frames)")
        return (torch.rand((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) < self.survival_probability).to(x.dtype)

    def forward_with_mask(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        residual = self.residual_branch(x)
        if self.training:
            if mask is None:
                mask = self.stochastic_mask(x)
            if mask.shape != (x.shape[0], 1, 1):
                raise ValueError("W68 requiere máscara (batch,1,1)")
            if not torch.logical_or(mask == 0, mask == 1).all():
                raise ValueError("W68 requiere máscara Bernoulli de ceros y unos")
            residual = residual * mask / self.survival_probability
        return F.gelu(x + residual)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_with_mask(x)


class LinearStochasticDepthWordsTCN(IsolatedWordsTCN):
    """W68: W33 con cuatro supervivencias lineales, sin aleatoriedad en eval."""

    DILATIONS = (1, 2, 4, 8)
    SURVIVAL_PROBABILITIES = (0.95, 0.90, 0.85, 0.80)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("LinearStochasticDepthWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.blocks = nn.Sequential(*[
            StochasticDepthTCNResidualBlock(128, dilation, dropout, probability)
            for dilation, probability in zip(self.DILATIONS, self.SURVIVAL_PROBABILITIES)
        ])


class ReZeroTCNResidualBlock(TCNResidualBlock):
    """W69: bloque W33 con una escala residual escalar inicializada en cero."""

    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__(channels, dilation, dropout)
        self.alpha = nn.Parameter(torch.zeros(()))

    def residual_branch(self, x: Tensor) -> Tensor:
        y = self.conv1(x)
        y = F.gelu(self.norm1(y))
        y = self.dropout(y)
        return self.norm2(self.conv2(y))

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.alpha * self.residual_branch(x))


class ReZeroTemporalWordsTCN(IsolatedWordsTCN):
    """W69: W33 con cuatro escalas ReZero y las mismas convoluciones temporales."""

    DILATIONS = (1, 2, 4, 8)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("ReZeroTemporalWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.blocks = nn.Sequential(*[
            ReZeroTCNResidualBlock(128, dilation, dropout)
            for dilation in self.DILATIONS
        ])


class EnergyPhaseResidualTCN(IsolatedWordsTCN):
    """W53: resume inicio, núcleo y final con fronteras de energía canónica.

    La corrección final inicia en cero y conserva de forma exacta el pooling
    promedio W33 antes de cualquier actualización de parámetros.
    """

    VELOCITY_SLICE = slice(266, 352)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("EnergyPhaseResidualTCN requiere feature_dim=352")
        if frames < 3:
            raise ValueError("EnergyPhaseResidualTCN requiere al menos tres frames")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.phase_encoder = nn.Sequential(nn.Linear(128 * 3, 64), nn.GELU())
        self.phase_projection = nn.Linear(64, 128, bias=False)
        nn.init.zeros_(self.phase_projection.weight)

    def phase_boundaries(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        with torch.no_grad():
            energy = x[:, :, self.VELOCITY_SLICE].abs().mean(dim=2)
            total = energy.sum(dim=1, keepdim=True)
            cumulative = energy.cumsum(dim=1)
            fallback = torch.tensor([self.frames // 3, (2 * self.frames) // 3], device=x.device, dtype=torch.long)
            first = torch.argmax((cumulative >= total / 3.0).to(torch.int64), dim=1).to(torch.long) + 1
            second = torch.argmax((cumulative >= (2.0 * total / 3.0)).to(torch.int64), dim=1).to(torch.long) + 1
            first = first.clamp(1, self.frames - 2)
            second = second.clamp(2, self.frames - 1)
            second = torch.maximum(second, first + 1)
            boundaries = torch.stack([first, second], dim=1)
            static = total.squeeze(1) < 1e-6
            return torch.where(static[:, None], fallback[None, :], boundaries)

    def phase_means(self, features: Tensor, boundaries: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 128 or features.shape[2] != self.frames:
            raise ValueError("W53 requiere rasgos temporales (batch,128,frames) compatibles")
        if boundaries.shape != (features.shape[0], 2):
            raise ValueError("W53 requiere fronteras (batch,2)")
        summaries = []
        for index in range(features.shape[0]):
            first, second = (int(value) for value in boundaries[index].tolist())
            if not (1 <= first < second <= self.frames - 1):
                raise ValueError("Las tres fases W53 deben ser no vacías")
            summaries.append(torch.cat([
                features[index, :, :first].mean(dim=-1),
                features[index, :, first:second].mean(dim=-1),
                features[index, :, second:].mean(dim=-1),
            ], dim=0))
        return torch.stack(summaries, dim=0)

    def pooled_features(self, x: Tensor) -> Tensor:
        temporal = super().temporal_features(x)
        phase_summary = self.phase_means(temporal, self.phase_boundaries(x))
        return temporal.mean(dim=-1) + self.phase_projection(self.phase_encoder(phase_summary))

    def forward(self, x: Tensor) -> Tensor:
        return self.head[1:](self.pooled_features(x).unsqueeze(-1))


class ChannelRecalibrationTCN(IsolatedWordsTCN):
    """W54: recalibración SE compacta de canales post-TCN, inicializada en identidad."""

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("ChannelRecalibrationTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.channel_gate = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 128))
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)

    def channel_factors(self, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1] != 128 or temporal.shape[2] != self.frames:
            raise ValueError("W54 requiere temporal W33 con forma (batch,128,frames)")
        return 2.0 * torch.sigmoid(self.channel_gate(temporal.mean(dim=-1)))

    def recalibrated_temporal_features(self, x: Tensor) -> Tensor:
        temporal = super().temporal_features(x)
        return temporal * self.channel_factors(temporal).unsqueeze(-1)

    def forward(self, x: Tensor) -> Tensor:
        return self.logits_from_temporal_features(self.recalibrated_temporal_features(x))


class ManualFactorialLDAMWordsTCN(_BaseModule):
    """W34: factoriza grupos manuales disjuntos sin usar etiquetas fonológicas.

    La clasificación sigue siendo única. Las proyecciones de factores solo dan
    una señal auxiliar de decorrelación durante entrenamiento; no son salidas
    lingüísticas ni intervienen en inferencia.
    """

    FACTOR_SLICES = ((0, 126), (126, 186), (186, 226), (226, 352))

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        _require_torch()
        super().__init__()
        if feature_dim != 352:
            raise ValueError("ManualFactorialLDAMWordsTCN requiere feature_dim=352")
        self.frames = frames
        self.features = feature_dim
        self.factor_channels = 48
        self.factor_projection_channels = 32

        def stem(channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(channels, self.factor_channels, kernel_size=3, padding=1),
                nn.GroupNorm(8, self.factor_channels),
                nn.GELU(),
            )

        self.geometry_stem = stem(126)
        self.shape_stem = stem(60)
        self.relation_stem = stem(40)
        self.motion_stem = stem(126)
        self.factor_projections = nn.ModuleList([
            nn.Conv1d(self.factor_channels, self.factor_projection_channels, kernel_size=1)
            for _ in self.FACTOR_SLICES
        ])
        self.fusion = nn.Conv1d(self.factor_channels * len(self.FACTOR_SLICES), 128, kernel_size=1)
        self.blocks = nn.Sequential(*[TCNResidualBlock(128, dilation, dropout) for dilation in (1, 2, 4, 8)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, classes),
        )

    def factor_stems(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        _check_sequence(x, self.frames, self.features)
        channels = x.transpose(1, 2)
        return (
            self.geometry_stem(channels[:, 0:126, :]),
            self.shape_stem(channels[:, 126:186, :]),
            self.relation_stem(channels[:, 186:226, :]),
            self.motion_stem(channels[:, 226:352, :]),
        )

    def temporal_features_from_stems(self, stems: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        return self.blocks(F.gelu(self.fusion(torch.cat(stems, dim=1))))

    def temporal_features(self, x: Tensor) -> Tensor:
        return self.temporal_features_from_stems(self.factor_stems(x))

    def logits_from_temporal_features(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 128:
            raise ValueError(f"Se esperaba rasgos temporales (batch,128,frames), se recibió {tuple(features.shape)}")
        return self.head(features)

    def factor_embeddings(self, stems: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        embeddings = [
            F.normalize(projection(stem).mean(dim=-1), p=2, dim=1)
            for projection, stem in zip(self.factor_projections, stems)
        ]
        return torch.stack(embeddings, dim=1)

    def forward_with_factor_embeddings(self, x: Tensor) -> tuple[Tensor, Tensor]:
        stems = self.factor_stems(x)
        temporal = self.temporal_features_from_stems(stems)
        return self.logits_from_temporal_features(temporal), self.factor_embeddings(stems)

    def forward(self, x: Tensor) -> Tensor:
        return self.logits_from_temporal_features(self.temporal_features(x))


class HopfieldPrototypeWordsTCN(IsolatedWordsTCN):
    """W22: memoria de prototipos auxiliares, nunca consultada en inferencia."""

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, slots_per_class: int = 2, beta: float = 8.0, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("HopfieldPrototypeWordsTCN requiere feature_dim=352")
        if slots_per_class < 1 or beta <= 0:
            raise ValueError("La memoria Hopfield requiere slots positivos y beta positiva")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.slots_per_class = int(slots_per_class)
        self.hopfield_beta = float(beta)
        self.prototype_memory = nn.Parameter(torch.empty(classes, self.slots_per_class, 128))
        nn.init.xavier_uniform_(self.prototype_memory)

    def memory_logits_from_temporal_features(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != 128:
            raise ValueError(f"Se esperaba rasgos temporales (batch,128,frames), se recibió {tuple(features.shape)}")
        query = F.normalize(features.mean(dim=-1), p=2, dim=1)
        memory = F.normalize(self.prototype_memory, p=2, dim=-1)
        similarities = torch.einsum("bd,ckd->bck", query, memory)
        return torch.logsumexp(self.hopfield_beta * similarities, dim=-1)

    def forward_with_memory(self, x: Tensor) -> tuple[Tensor, Tensor]:
        temporal = self.temporal_features(x)
        return self.logits_from_temporal_features(temporal), self.memory_logits_from_temporal_features(temporal)


class SignerInvariantWordsTCN(IsolatedWordsTCN):
    """W9: adversaria de firmante solo durante el entrenamiento."""

    def __init__(self, frames: int = 30, feature_dim: int = 226, classes: int = 200, signer_classes: int = 2, adversarial_scale: float = 0.10, dropout: float = 0.25):
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        if signer_classes < 2:
            raise ValueError("SignerInvariantWordsTCN requiere al menos dos firmantes")
        if adversarial_scale <= 0:
            raise ValueError("La escala adversaria debe ser positiva")
        self.signer_classes = signer_classes
        self.adversarial_scale = float(adversarial_scale)
        self.signer_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, signer_classes),
        )

    def forward_with_signer(self, x: Tensor, scale: float | None = None) -> tuple[Tensor, Tensor]:
        temporal = self.temporal_features(x)
        logits = self.head(temporal)
        pooled = temporal.mean(dim=-1)
        signer_logits = self.signer_head(reverse_gradient(pooled, self.adversarial_scale if scale is None else scale))
        return logits, signer_logits


class TrajectoryResidualWordsTCN(IsolatedWordsTCN):
    """W12: conserva W3 y suma una corrección nula de trayectoria robusta."""

    def __init__(self, frames: int = 30, feature_dim: int = 364, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 364:
            raise ValueError("TrajectoryResidualWordsTCN requiere feature_dim=364")
        super().__init__(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.features = 364
        self.trajectory_residual = nn.Conv1d(12, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.trajectory_residual.weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x[:, :, :352])
        trajectory = x[:, :, 352:].transpose(1, 2)
        return canonical + self.trajectory_residual(trajectory)


class BodyAnchorResidualTCN(IsolatedWordsTCN):
    """W35: ubicación manual respecto a hombros como corrección nula sobre W33.

    Los 352 canales canónicos se procesan por la misma ruta W3/W33. La única
    señal añadida son 51 canales cuerpo-relativos ya cacheados: posiciones XY,
    velocidades XY y máscaras de presencia. La proyección final no tiene sesgo
    y comienza en cero; por tanto, con la misma semilla los logits iniciales
    son exactamente los de ``IsolatedWordsTCN(feature_dim=352)``.
    """

    CANONICAL_FEATURES = 352
    BODY_ANCHOR_FEATURES = 51
    FEATURE_DIM = CANONICAL_FEATURES + BODY_ANCHOR_FEATURES

    def __init__(self, frames: int = 30, feature_dim: int = FEATURE_DIM, classes: int = 200, dropout: float = 0.25):
        if feature_dim != self.FEATURE_DIM:
            raise ValueError("BodyAnchorResidualTCN requiere feature_dim=403")
        super().__init__(frames=frames, feature_dim=self.CANONICAL_FEATURES, classes=classes, dropout=dropout)
        self.features = self.FEATURE_DIM
        self.body_stem = nn.Sequential(
            nn.Conv1d(self.BODY_ANCHOR_FEATURES, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.body_residual = nn.Conv1d(48, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.body_residual.weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x[:, :, :self.CANONICAL_FEATURES])
        body_anchor = x[:, :, self.CANONICAL_FEATURES:].transpose(1, 2)
        return canonical + self.body_residual(self.body_stem(body_anchor))


class WorldHandGeometryResidualTCN(IsolatedWordsTCN):
    """W83: corrección nula W33 con geometría 3D mundial intramanual."""

    CANONICAL_FEATURES = 352
    WORLD_HAND_FEATURES = 126
    FEATURE_DIM = CANONICAL_FEATURES + WORLD_HAND_FEATURES

    def __init__(self, frames: int = 30, feature_dim: int = FEATURE_DIM, classes: int = 200, dropout: float = 0.25):
        if feature_dim != self.FEATURE_DIM:
            raise ValueError("WorldHandGeometryResidualTCN requiere feature_dim=478")
        super().__init__(frames=frames, feature_dim=self.CANONICAL_FEATURES, classes=classes, dropout=dropout)
        self.features = self.FEATURE_DIM
        self.world_stem = nn.Sequential(
            nn.Conv1d(self.WORLD_HAND_FEATURES, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.world_residual = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.world_residual.weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x[:, :, :self.CANONICAL_FEATURES])
        world_hand = x[:, :, self.CANONICAL_FEATURES:].transpose(1, 2)
        return canonical + self.world_residual(self.world_stem(world_hand))


class QualityGatedBodyAnchorResidualTCN(BodyAnchorResidualTCN):
    """W38: W35 cuya corrección corporal se anula sin evidencia mano–pose.

    Los tres canales de máscara ya cacheados ocupan los índices globales
    400:403. La compuerta es determinista: pose × promedio de manos con pose.
    No añade un clasificador, no depende de clases/firmantes y conserva la
    equivalencia inicial con W33 por la proyección residual nula heredada.
    """

    MASK_START = BodyAnchorResidualTCN.CANONICAL_FEATURES + 48

    def quality_gate(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        masks = x[:, :, self.MASK_START:self.MASK_START + 3]
        if masks.shape[-1] != 3:
            raise ValueError("W38 requiere las tres máscaras body_anchor")
        right, left, pose = masks.unbind(dim=2)
        return (pose * (right + left) / 2.0).unsqueeze(1)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x[:, :, :self.CANONICAL_FEATURES])
        body_anchor = x[:, :, self.CANONICAL_FEATURES:].transpose(1, 2)
        residual = self.body_residual(self.body_stem(body_anchor))
        return canonical + self.quality_gate(x) * residual


class PalmAxisResidualTCN(IsolatedWordsTCN):
    """W39: forma XY de mano orientada al eje palma–dedo medio sobre W33."""

    CANONICAL_FEATURES = 352
    PALM_AXIS_FEATURES = 84
    FEATURE_DIM = CANONICAL_FEATURES + PALM_AXIS_FEATURES

    def __init__(self, frames: int = 30, feature_dim: int = FEATURE_DIM, classes: int = 200, dropout: float = 0.25):
        if feature_dim != self.FEATURE_DIM:
            raise ValueError("PalmAxisResidualTCN requiere feature_dim=436")
        super().__init__(frames=frames, feature_dim=self.CANONICAL_FEATURES, classes=classes, dropout=dropout)
        self.features = self.FEATURE_DIM
        self.palm_axis_stem = nn.Sequential(
            nn.Conv1d(self.PALM_AXIS_FEATURES, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.palm_axis_residual = nn.Conv1d(48, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.palm_axis_residual.weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x[:, :, :self.CANONICAL_FEATURES])
        palm_axis = x[:, :, self.CANONICAL_FEATURES:].transpose(1, 2)
        return canonical + self.palm_axis_residual(self.palm_axis_stem(palm_axis))


class DenseHandSpectralResidualTCN(IsolatedWordsTCN):
    """W79: firma Laplaciana densa de ambas manos sobre el prefijo W33."""

    CANONICAL_FEATURES = 352
    SPECTRAL_FEATURES = 40
    FEATURE_DIM = CANONICAL_FEATURES + SPECTRAL_FEATURES

    def __init__(self, frames: int = 30, feature_dim: int = FEATURE_DIM, classes: int = 200, dropout: float = 0.25):
        if feature_dim != self.FEATURE_DIM:
            raise ValueError("DenseHandSpectralResidualTCN requiere feature_dim=392")
        super().__init__(frames=frames, feature_dim=self.CANONICAL_FEATURES, classes=classes, dropout=dropout)
        self.features = self.FEATURE_DIM
        self.spectral_stem = nn.Sequential(
            nn.Conv1d(self.SPECTRAL_FEATURES, 48, kernel_size=3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
        )
        self.spectral_residual = nn.Conv1d(48, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.spectral_residual.weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x[:, :, :self.CANONICAL_FEATURES])
        spectral = x[:, :, self.CANONICAL_FEATURES:].transpose(1, 2)
        return canonical + self.spectral_residual(self.spectral_stem(spectral))


class LatentStyleMixWordsTCN(IsolatedWordsTCN):
    """W41: mezcla estadística temporal solo en entrenamiento sobre W33.

    No añade parámetros, canales ni salidas. Al evaluar, ``temporal_features``
    coincide con W33 bit a bit bajo la misma inicialización. La mezcla se
    realiza después de la fusión W33 y antes de los bloques TCN, por lo que
    altera estilo latente temporal sin leer etiquetas, firmantes o test.
    """

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25, probability: float = 0.50, alpha: float = 0.10, epsilon: float = 1e-6):
        if feature_dim != 352:
            raise ValueError("LatentStyleMixWordsTCN requiere feature_dim=352")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("W41 probability debe estar en [0,1]")
        if alpha <= 0.0:
            raise ValueError("W41 alpha debe ser positivo")
        if epsilon <= 0.0:
            raise ValueError("W41 epsilon debe ser positivo")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.probability = float(probability)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

    def mix_temporal_statistics(self, features: Tensor, permutation: Tensor | None = None, lambdas: Tensor | None = None) -> Tensor:
        """Mezcla media y escala por canal; la semilla Torch controla aleatoriedad."""
        if features.ndim != 3 or features.shape[1] != 128 or features.shape[2] != self.frames:
            raise ValueError(f"W41 espera rasgos temporales (batch,128,{self.frames})")
        if not self.training or self.probability == 0.0 or features.shape[0] < 2:
            return features
        batch = features.shape[0]
        if permutation is None:
            permutation = torch.randperm(batch, device=features.device)
        if permutation.shape != (batch,) or permutation.min() < 0 or permutation.max() >= batch:
            raise ValueError("W41 recibió permutación de batch inválida")
        if lambdas is None:
            distribution = torch.distributions.Beta(self.alpha, self.alpha)
            lambdas = distribution.sample((batch,)).to(device=features.device, dtype=features.dtype)
        if lambdas.shape != (batch,) or not torch.isfinite(lambdas).all() or (lambdas < 0).any() or (lambdas > 1).any():
            raise ValueError("W41 recibió lambdas inválidos")
        mean = features.mean(dim=-1, keepdim=True)
        scale = torch.sqrt(features.var(dim=-1, keepdim=True, unbiased=False) + self.epsilon)
        mixing = lambdas.view(batch, 1, 1)
        mixed_mean = mixing * mean + (1.0 - mixing) * mean[permutation]
        mixed_scale = mixing * scale + (1.0 - mixing) * scale[permutation]
        return mixed_scale * (features - mean) / scale + mixed_mean

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        channels = x.transpose(1, 2)
        hand_block = torch.cat([channels[:, :126, :], channels[:, 226:352, :]], dim=1)
        hands = self.hands(hand_block)
        pose = self.pose(channels[:, 126:178, :])
        face = self.face(channels[:, 178:226, :])
        fused = F.gelu(self.fusion(torch.cat([hands, pose, face], dim=1)))
        return self.blocks(self.mix_temporal_statistics(fused))


class MotionBiasedAttentiveWordsTCN(IsolatedWordsTCN):
    """W44: reemplaza el promedio W33 por atención temporal de un solo modelo.

    El sesgo de energía lee exclusivamente velocidad manual `266:352`; no es una
    compuerta de calidad ni rechaza frames. La forma manual `126:226` conserva
    su ruta W33 hacia el TCN y nunca entra al cálculo del sesgo.
    """

    MOTION_START = 266
    MOTION_END = 352

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25, motion_bias_scale: float = 0.25, motion_epsilon: float = 1e-6):
        if feature_dim != 352:
            raise ValueError("MotionBiasedAttentiveWordsTCN requiere feature_dim=352")
        if motion_bias_scale < 0.0 or motion_epsilon <= 0.0:
            raise ValueError("W44 requiere escala de sesgo no negativa y epsilon positivo")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.motion_bias_scale = float(motion_bias_scale)
        self.motion_epsilon = float(motion_epsilon)
        self.attention_score = nn.Sequential(
            nn.Conv1d(128, 32, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, classes),
        )

    def motion_energy_bias(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        energy = x[:, :, self.MOTION_START:self.MOTION_END].abs().mean(dim=2)
        centered = energy - energy.mean(dim=1, keepdim=True)
        scale = torch.sqrt(energy.var(dim=1, keepdim=True, unbiased=False) + self.motion_epsilon)
        return self.motion_bias_scale * centered / scale

    def attention_weights(self, temporal: Tensor, x: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1:] != (128, self.frames):
            raise ValueError(f"W44 espera rasgos temporales (batch,128,{self.frames})")
        if temporal.shape[0] != x.shape[0]:
            raise ValueError("W44 requiere batch compatible entre temporal e input")
        learned = self.attention_score(temporal).squeeze(1)
        return torch.softmax(learned + self.motion_energy_bias(x), dim=1)

    def forward(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        temporal = self.temporal_features(x)
        weights = self.attention_weights(temporal, x).unsqueeze(1)
        pooled = (temporal * weights).sum(dim=-1)
        return self.head(pooled)


class MultiscaleTemporalDifferenceResidualTCN(IsolatedWordsTCN):
    """W45: corrección nula W33 con diferencias absolutas post-TCN a tres escalas."""

    LAGS = (1, 2, 4)

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("MultiscaleTemporalDifferenceResidualTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.difference_stem = nn.Sequential(
            nn.Conv1d(128 * len(self.LAGS), 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.difference_residual = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.difference_residual.weight)

    def multiscale_differences(self, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1:] != (128, self.frames):
            raise ValueError(f"W45 espera rasgos temporales (batch,128,{self.frames})")
        details = []
        for lag in self.LAGS:
            if lag >= self.frames:
                raise ValueError("W45 requiere frames mayores que todos los retardos")
            delayed = F.pad(temporal[:, :, :-lag], (lag, 0), mode="replicate")
            details.append((temporal - delayed).abs())
        return torch.cat(details, dim=1)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x)
        residual = self.difference_residual(self.difference_stem(self.multiscale_differences(canonical)))
        return canonical + residual


class DCTSpectralResidualWordsTCN(IsolatedWordsTCN):
    """W46: corrección nula W33 desde banda DCT-II fija de rasgos post-TCN."""

    BAND_START = 1
    BAND_END = 9

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("DCTSpectralResidualWordsTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        times = torch.arange(frames, dtype=torch.float64).view(1, frames)
        frequencies = torch.arange(frames, dtype=torch.float64).view(frames, 1)
        matrix = math.sqrt(2.0 / frames) * torch.cos(math.pi * (times + 0.5) * frequencies / frames)
        matrix[0] = 1.0 / math.sqrt(frames)
        self.register_buffer("dct_matrix", matrix, persistent=True)
        self.spectral_stem = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.spectral_residual = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.spectral_residual.weight)

    def spectral_band(self, temporal: Tensor) -> Tensor:
        if temporal.ndim != 3 or temporal.shape[1:] != (128, self.frames):
            raise ValueError(f"W46 espera rasgos temporales (batch,128,{self.frames})")
        dct = self.dct_matrix.to(device=temporal.device, dtype=temporal.dtype)
        coefficients = temporal @ dct.transpose(0, 1)
        selected = torch.zeros_like(coefficients)
        selected[:, :, self.BAND_START:self.BAND_END] = coefficients[:, :, self.BAND_START:self.BAND_END]
        return selected @ dct

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x)
        residual = self.spectral_residual(self.spectral_stem(self.spectral_band(canonical)))
        return canonical + residual


class FixedHandGraphResidualTCN(IsolatedWordsTCN):
    """W49: grafo anatómico manual fijo como corrección residual nula sobre W33."""

    NODES_PER_HAND = 21
    TOTAL_NODES = 42

    @classmethod
    def anatomical_adjacency(cls) -> Tensor:
        """Devuelve el grafo manual simétrico con autoaristas, antes de normalizar."""
        adjacency = torch.zeros((cls.TOTAL_NODES, cls.TOTAL_NODES), dtype=torch.float32)
        edges = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        )
        for offset in (0, cls.NODES_PER_HAND):
            for first, second in edges:
                adjacency[offset + first, offset + second] = 1.0
                adjacency[offset + second, offset + first] = 1.0
        adjacency += torch.eye(cls.TOTAL_NODES, dtype=adjacency.dtype)
        return adjacency

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25):
        if feature_dim != 352:
            raise ValueError("FixedHandGraphResidualTCN requiere feature_dim=352")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        adjacency = self.anatomical_adjacency()
        self.register_buffer("hand_adjacency", adjacency / adjacency.sum(dim=1, keepdim=True), persistent=True)
        self.graph_stem = nn.Sequential(
            nn.Conv1d(252, 64, kernel_size=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.graph_blocks = nn.Sequential(
            TCNResidualBlock(64, dilation=1, dropout=dropout),
            TCNResidualBlock(64, dilation=2, dropout=dropout),
        )
        self.graph_residual = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.graph_residual.weight)

    def graph_node_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        positions = x[:, :, :126].reshape(x.shape[0], self.frames, self.TOTAL_NODES, 3)
        adjacency = self.hand_adjacency.to(device=x.device, dtype=x.dtype)
        neighbours = torch.einsum("vw,btwc->btvc", adjacency, positions)
        return torch.cat([positions, neighbours], dim=-1)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        canonical = IsolatedWordsTCN.temporal_features(self, x)
        graph_channels = self.graph_node_features(x).reshape(x.shape[0], self.frames, 252).transpose(1, 2)
        residual = self.graph_residual(self.graph_blocks(self.graph_stem(graph_channels)))
        return canonical + residual


class CosineClassifierWordsTCN(IsolatedWordsTCN):
    """W47: cabeza coseno normalizada sobre la representación temporal W33."""

    def __init__(self, frames: int = 30, feature_dim: int = 352, classes: int = 200, dropout: float = 0.25, normalization_epsilon: float = 1e-6):
        if feature_dim != 352:
            raise ValueError("CosineClassifierWordsTCN requiere feature_dim=352")
        if normalization_epsilon <= 0.0:
            raise ValueError("W47 requiere epsilon de normalización positivo")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.normalization_epsilon = float(normalization_epsilon)
        self.embedding_projection = nn.Linear(128, 160)
        self.embedding_dropout = nn.Dropout(dropout)
        self.class_weights = nn.Parameter(torch.empty(classes, 160))
        nn.init.xavier_uniform_(self.class_weights)
        # Reemplaza y descarta los parámetros de la cabeza lineal W33; W47 usa un único
        # clasificador coseno sin sesgo ni escala aprendida.
        self.head = nn.Identity()

    def cosine_logits_from_embedding(self, embedding: Tensor) -> Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != 160:
            raise ValueError("W47 espera embeddings (batch,160)")
        normalized_embedding = F.normalize(embedding, p=2, dim=1, eps=self.normalization_epsilon)
        normalized_weights = F.normalize(self.class_weights, p=2, dim=1, eps=self.normalization_epsilon)
        return normalized_embedding @ normalized_weights.transpose(0, 1)

    def forward(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        temporal = self.temporal_features(x)
        embedding = self.embedding_dropout(F.gelu(self.embedding_projection(temporal.mean(dim=-1))))
        return self.cosine_logits_from_embedding(embedding)


class PathSignatureEarlyFusionTCN(_BaseModule):
    """W40: clasificador único con firma manual de trayectoria antes del TCN.

    La firma de orden dos se calcula fuera del modelo en una caché determinista.
    Esta clase no usa residual tardío, cuerpo, máscaras, fusión de modelos ni
    pesos W33: aprende desde cero una representación temprana de `(30,424)`.
    """

    CANONICAL_FEATURES = 352
    PATH_SIGNATURE_FEATURES = 72
    FEATURE_DIM = CANONICAL_FEATURES + PATH_SIGNATURE_FEATURES

    def __init__(self, frames: int = 30, feature_dim: int = FEATURE_DIM, classes: int = 200, dropout: float = 0.25):
        _require_torch()
        super().__init__()
        if feature_dim != self.FEATURE_DIM:
            raise ValueError("PathSignatureEarlyFusionTCN requiere feature_dim=424")
        self.frames = frames
        self.features = self.FEATURE_DIM
        self.hand_stem = nn.Sequential(nn.Conv1d(252, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU())
        self.shape_relation_stem = nn.Sequential(nn.Conv1d(100, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.path_signature_stem = nn.Sequential(nn.Conv1d(self.PATH_SIGNATURE_FEATURES, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU())
        self.fusion = nn.Conv1d(192, 128, 1)
        self.blocks = nn.Sequential(*[TCNResidualBlock(128, dilation, dropout) for dilation in (1, 2, 4, 8)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, classes),
        )

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        channels = x.transpose(1, 2)
        hands = torch.cat([channels[:, :126, :], channels[:, 226:352, :]], dim=1)
        shape_relations = channels[:, 126:226, :]
        signature = channels[:, 352:424, :]
        fused = torch.cat([self.hand_stem(hands), self.shape_relation_stem(shape_relations), self.path_signature_stem(signature)], dim=1)
        return self.blocks(F.gelu(self.fusion(fused)))

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.temporal_features(x))


class KinematicChannelDropoutWordsTCN(IsolatedWordsTCN):
    """W17: regulariza solo la aceleración manual de W15 durante entrenamiento."""

    def __init__(self, frames: int = 30, feature_dim: int = 478, classes: int = 200, dropout: float = 0.25, acceleration_dropout: float = 0.20):
        if feature_dim != 478:
            raise ValueError("KinematicChannelDropoutWordsTCN requiere feature_dim=478")
        if not 0.0 <= acceleration_dropout < 1.0:
            raise ValueError("acceleration_dropout debe estar en [0,1)")
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.acceleration_dropout = nn.Dropout1d(acceleration_dropout)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        if self.training:
            x = x.clone()
            acceleration = x[:, :, 352:478].transpose(1, 2)
            x[:, :, 352:478] = self.acceleration_dropout(acceleration).transpose(1, 2)
        return super().temporal_features(x)


class DualViewResidualWordsTCN(_BaseModule):
    """W13: W3 primario más una segunda vista temporal introducida de modo residual."""

    def __init__(self, frames: int = 30, feature_dim: int = 704, classes: int = 200, dropout: float = 0.25):
        _require_torch()
        super().__init__()
        if feature_dim != 704:
            raise ValueError("DualViewResidualWordsTCN requiere feature_dim=704")
        self.frames = frames
        self.features = feature_dim
        self.primary = IsolatedWordsTCN(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.secondary = IsolatedWordsTCN(frames=frames, feature_dim=352, classes=classes, dropout=dropout)
        self.secondary_residual = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.secondary_residual.weight)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        primary = self.primary.temporal_features(x[:, :, :352])
        secondary = self.secondary.temporal_features(x[:, :, 352:])
        return primary + self.secondary_residual(secondary)

    def forward(self, x: Tensor) -> Tensor:
        return self.primary.head(self.temporal_features(x))


class GatedDualViewResidualWordsTCN(DualViewResidualWordsTCN):
    """W14: residual de la segunda vista regulado por una compuerta temporal de W3."""

    def __init__(self, frames: int = 30, feature_dim: int = 704, classes: int = 200, dropout: float = 0.25):
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.temporal_gate = nn.Conv1d(128, 128, kernel_size=1)

    def temporal_features(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        primary = self.primary.temporal_features(x[:, :, :352])
        secondary = self.secondary.temporal_features(x[:, :, 352:])
        gate = torch.sigmoid(self.temporal_gate(primary))
        return primary + gate * self.secondary_residual(secondary)


class IsolatedWordsChronologicalTCN(IsolatedWordsTCN):
    """Control W4: misma red de palabras, con pool cronológico en el cierre."""

    def __init__(self, frames: int = 30, feature_dim: int = 226, classes: int = 200, dropout: float = 0.25):
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.pool = ChronologicalRolePool(128)
        self.head = nn.Sequential(
            nn.Linear(128, 160),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(160, classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        x = x.transpose(1, 2)
        hand_block = x[:, :126, :]
        if self.features == 352:
            hand_block = torch.cat([hand_block, x[:, 226:352, :]], dim=1)
        hands = self.hands(hand_block)
        pose = self.pose(x[:, 126:178, :])
        face = self.face(x[:, 178:226, :])
        features = self.blocks(F.gelu(self.fusion(torch.cat([hands, pose, face], dim=1))))
        return self.head(self.pool(features))


class IsolatedWordsResidualChronologicalTCN(IsolatedWordsTCN):
    """Control W5: cronología por fases como corrección residual de W3."""

    def __init__(self, frames: int = 30, feature_dim: int = 226, classes: int = 200, dropout: float = 0.25):
        super().__init__(frames=frames, feature_dim=feature_dim, classes=classes, dropout=dropout)
        self.pool = ResidualChronologicalPool(128)

    def forward(self, x: Tensor) -> Tensor:
        _check_sequence(x, self.frames, self.features)
        x = x.transpose(1, 2)
        hand_block = x[:, :126, :]
        if self.features == 352:
            hand_block = torch.cat([hand_block, x[:, 226:352, :]], dim=1)
        hands = self.hands(hand_block)
        pose = self.pose(x[:, 126:178, :])
        face = self.face(x[:, 178:226, :])
        features = self.blocks(F.gelu(self.fusion(torch.cat([hands, pose, face], dim=1))))
        # Reutiliza la cabeza creada por IsolatedWordsTCN. Con gate=0, alimenta
        # exactamente el mismo promedio temporal y conserva los mismos pesos.
        return self.head[1:](self.pool(features).unsqueeze(-1))


def build_model(task: str, **kwargs):
    _require_torch()
    if task == "static_letter":
        return StaticLettersMLP(**kwargs)
    if task == "dynamic_letter":
        return DynamicLettersTCN(**kwargs)
    if task == "recovery_positions":
        return TemporalTCN(**kwargs)
    if task == "successor_positions126":
        return TemporalTCN(**kwargs)
    if task == "successor_positions126_train_augmented":
        return TemporalTCN(**kwargs)
    if task == "successor_episodic_real_reference":
        return TemporalTCN(**kwargs)
    if task == "successor_temporal_relation_pairs":
        return TemporalTCN(**kwargs)
    if task == "successor_selective_core_relation":
        return TemporalTCN(**kwargs)
    if task == "successor_intramanual_bone166":
        return TemporalTCN(**kwargs)
    if task == "successor_intramanual_kinematic196":
        return TemporalTCN(**kwargs)
    if task == "successor_signer_stratified_batch":
        return TemporalTCN(**kwargs)
    if task == "successor_soft_presence_weight":
        return TemporalTCN(**kwargs)
    if task == "successor_masked_hand_reconstruction":
        return MaskedHandReconstructionTCN(**kwargs)
    if task == "successor_intraclip_style_normalization":
        return IntraClipStyleNormalizationTCN(**kwargs)
    if task == "successor_temporal_pyramid_pooling":
        return TemporalPyramidPoolingTCN(**kwargs)
    if task == "successor_logavgexp_temporal_pooling":
        return LogAvgExpTemporalPoolingTCN(**kwargs)
    if task == "successor_uniform_label_smoothing":
        return TemporalTCN(**kwargs)
    if task == "successor_train_only_swa":
        return TemporalTCN(**kwargs)
    if task == "successor_shared_bilateral_tcn":
        return SharedBilateralTCN(**kwargs)
    if task == "successor_ecoc_auxiliary_head":
        return ECOCAuxiliaryTCN(**kwargs)
    if task == "successor_signer_vrex":
        return TemporalTCN(**kwargs)
    if task == "successor_fixed_hand_graph_tcn":
        return FixedHandGraphTCN(**kwargs)
    if task == "successor_arc_length_frame_reindexing":
        return ArcLengthFrameReindexingTCN(**kwargs)
    if task == "successor_bidirectional_gru":
        return BidirectionalGRUClassifier(**kwargs)
    if task == "successor_cosine_classifier":
        return CosineClassifierTCN(**kwargs)
    if task == "successor_spectral_tcn":
        return SpectralTemporalTCN(**kwargs)
    if task == "successor_deformable_temporal_offsets":
        return DeformableTemporalOffsetsTCN(**kwargs)
    if task == "successor_partial_temporal_conv":
        return PartialTemporalConvTCN(**kwargs)
    if task == "successor_global_wrist132":
        return TemporalTCN(**kwargs)
    if task == "successor_wrist_velocity132":
        return TemporalTCN(**kwargs)
    if task == "recovery_path_signature":
        return TemporalTCN(**kwargs)
    if task == "dynamic_alphabet_zenodo":
        return TemporalTCN(**kwargs)
    if task == "ickmejia_jkq":
        return DynamicLettersTCN(**kwargs)
    if task == "isolated_word":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_chronological":
        return IsolatedWordsChronologicalTCN(**kwargs)
    if task == "isolated_word_residual_chronological":
        return IsolatedWordsResidualChronologicalTCN(**kwargs)
    if task == "isolated_word_zero_init_residual":
        return IsolatedWordsResidualChronologicalTCN(**kwargs)
    if task == "isolated_word_signer_invariant":
        return SignerInvariantWordsTCN(**kwargs)
    if task == "isolated_word_trajectory_residual":
        return TrajectoryResidualWordsTCN(**kwargs)
    if task == "isolated_word_kinematic_channel_dropout":
        return KinematicChannelDropoutWordsTCN(**kwargs)
    if task == "isolated_word_temporal_prediction_consistency":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_cross_signer_feature_mixup":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_cross_signer_temporal_consistency":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_contrastive_soft_dtw_alignment":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_hopfield_prototype_memory":
        return HopfieldPrototypeWordsTCN(**kwargs)
    if task == "isolated_word_log_euclidean_covariance_consistency":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_uncertainty_balanced_covariance":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_sharpness_aware_w3":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_group_dro_signer":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_w85_train_signer_group_dro_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_w89_class_conditional_hand_quality_curriculum_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_w93_motion_adaptive_temporal_coherence_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_masked_temporal_pretraining":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_pretrained_temporal_consistency":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_temporal_order_pretraining":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_mean_teacher_temporal_consistency":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_position_velocity_representation_consistency":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_ema_weight_average_inference":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_ldam_deferred_reweighting":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_manual_factorial_ldam":
        return ManualFactorialLDAMWordsTCN(**kwargs)
    if task == "isolated_word_body_anchor_residual_ldam":
        return BodyAnchorResidualTCN(**kwargs)
    if task == "isolated_word_w94_bimanual_temporal_coupling_residual_ldam":
        return BimanualCouplingResidualTCN(**kwargs)
    if task == "isolated_word_w95_trajectory_topology_residual_ldam":
        return TrajectoryTopologyResidualTCN(**kwargs)
    if task == "isolated_word_w96_temporal_self_similarity_residual_ldam":
        return TemporalSelfSimilarityResidualTCN(**kwargs)
    if task == "isolated_word_w97_fingertip_turning_curvature_residual_ldam":
        return FingertipTurningCurvatureResidualTCN(**kwargs)
    if task == "isolated_word_w98_finger_extension_permutation_entropy_residual_ldam":
        return FingerExtensionPermutationEntropyResidualTCN(**kwargs)
    if task == "isolated_word_w83_world_hand_geometry_residual_ldam":
        return WorldHandGeometryResidualTCN(**kwargs)
    if task == "isolated_word_quality_gated_body_anchor_ldam":
        return QualityGatedBodyAnchorResidualTCN(**kwargs)
    if task == "isolated_word_palm_axis_residual_ldam":
        return PalmAxisResidualTCN(**kwargs)
    if task == "isolated_word_dense_hand_spectral_signature_ldam":
        return DenseHandSpectralResidualTCN(**kwargs)
    if task == "isolated_word_latent_style_mix_ldam":
        return LatentStyleMixWordsTCN(**kwargs)
    if task == "isolated_word_motion_biased_attentive_pooling_ldam":
        return MotionBiasedAttentiveWordsTCN(**kwargs)
    if task == "isolated_word_channel_recalibration_ldam":
        return ChannelRecalibrationTCN(**kwargs)
    if task == "isolated_word_hand_branch_structural_dropout_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_dominant_hand_canonicalization_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_cross_signer_supervised_contrast_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_global_inplane_rotation_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_decoupled_classifier_retraining_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_velocity_magnitude_canonicalization_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_original_duration_aware_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_relative_time_coordinates_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_train_only_feature_standardization_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_parallel_receptive_field_tcn_ldam":
        return ParallelReceptiveFieldWordsTCN(**kwargs)
    if task == "isolated_word_parameter_free_temporal_shift_ldam":
        return ParameterFreeTemporalShiftWordsTCN(**kwargs)
    if task == "isolated_word_temporal_weight_standardization_ldam":
        return TemporalWeightStandardizedWordsTCN(**kwargs)
    if task == "isolated_word_depthwise_separable_temporal_tcn_ldam":
        return DepthwiseSeparableTemporalWordsTCN(**kwargs)
    if task == "isolated_word_linear_stochastic_depth_ldam":
        return LinearStochasticDepthWordsTCN(**kwargs)
    if task == "isolated_word_rezero_temporal_residual_ldam":
        return ReZeroTemporalWordsTCN(**kwargs)
    if task == "isolated_word_energy_density_reparameterized_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_confusion_spectral_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_adaptive_temporal_prototype_ldam":
        return AdaptiveTemporalPrototypeWordsTCN(**kwargs)
    if task == "isolated_word_compositional_temporal_prototype_ldam":
        return CompositionalTemporalPrototypeWordsTCN(**kwargs)
    if task == "isolated_word_shape_motion_bilinear_ldam":
        return ShapeMotionBilinearWordsTCN(**kwargs)
    if task == "isolated_word_signer_covariance_alignment_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_w80_interior_hand_reconstruction_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_w81_activity_boundary_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_w84_explicit_hand_presence_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_energy_phase_residual_ldam":
        return EnergyPhaseResidualTCN(**kwargs)
    if task == "isolated_word_multiscale_temporal_difference_residual_ldam":
        return MultiscaleTemporalDifferenceResidualTCN(**kwargs)
    if task == "isolated_word_dct_spectral_residual_ldam":
        return DCTSpectralResidualWordsTCN(**kwargs)
    if task == "isolated_word_fixed_hand_graph_residual_ldam":
        return FixedHandGraphResidualTCN(**kwargs)
    if task == "isolated_word_cosine_classifier_ldam":
        return CosineClassifierWordsTCN(**kwargs)
    if task == "isolated_word_train_prior_logit_adjusted_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_classifier_coherence_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_focal_ldam_deferred_reweighting":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_class_balanced_sampling_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_stochastic_dropout_consistency_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_canonical_motion_vat_ldam":
        return IsolatedWordsTCN(**kwargs)
    if task == "isolated_word_path_signature_early_fusion_ldam":
        return PathSignatureEarlyFusionTCN(**kwargs)
    if task == "isolated_word_dual_view_residual":
        return DualViewResidualWordsTCN(**kwargs)
    if task == "isolated_word_gated_dual_view_residual":
        return GatedDualViewResidualWordsTCN(**kwargs)
    raise ValueError(f"Tarea no soportada: {task!r}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
