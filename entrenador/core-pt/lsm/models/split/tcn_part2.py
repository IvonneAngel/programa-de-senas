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
    """TCN sucesora con un codificador temporal compartido entre manos."""

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
    """TCN sucesora con decodificador auxiliar de landmarks manuales."""

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
    """TCN sucesora con corrección residual de traslación y escala intra-clip."""

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
    """Resume una secuencia conservando roles de inicio, núcleo y cierre."""

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

