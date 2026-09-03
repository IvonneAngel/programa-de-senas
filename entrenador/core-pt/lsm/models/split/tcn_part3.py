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



class ResidualChronologicalPool(_BaseModule):
    """Promedio temporal más una corrección cronológica con compuerta residual."""

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
    """W73: W33 más clasificación directa contra prototipos temporales entrenables."""

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
    """W75: Soft-DTW contra prototipos compuestos por átomos compartidos."""

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

