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
    """W41: mezcla estadística temporal solo en entrenamiento sobre W33."""

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
    """W44: reemplaza el promedio W33 por atención temporal de un solo modelo."""

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
# Reemplaza y descarta los parámetros de la cabeza lineal W33;
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
    """W40: clasificador único con firma manual de trayectoria antes del TCN."""

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

