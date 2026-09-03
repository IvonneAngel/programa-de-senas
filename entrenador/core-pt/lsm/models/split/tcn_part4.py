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
    """W53: resume inicio, núcleo y final con fronteras de energía canónica."""

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
    """W34: factoriza grupos manuales disjuntos sin usar etiquetas fonológicas."""

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
    """W35: ubicación manual respecto a hombros como corrección nula sobre W33."""

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
    """W38: W35 cuya corrección corporal se anula sin evidencia mano–pose."""

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

