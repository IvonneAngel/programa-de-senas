"""Entrenamiento reproducible del benchmark sucesor restaurado."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler

from lsm.models.tcn import DeformableTemporalOffsetsTCN, DeterministicEchoReservoirTCN, LowRankBiSSMTemporalTCN, LowRankTemporalRelationTCN, PartialTemporalConvTCN, SignerAdversarialTCN, TemporalTCN, parameter_count, reverse_gradient


WARMUP_EPOCHS = 4
TOTAL_EPOCHS = 40
LR_START = 0.0002
LR_MAX = 0.002
LR_MIN = 0.00002
JACOBIAN_LAMBDA = 80.0
SUPCON_LAMBDA = 0.10
SUPCON_TEMPERATURE = 0.10
DOMAIN_ADVERSARIAL_LAMBDA = 0.10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def learning_rate_for_epoch(epoch: int, mode: str, total_epochs: int = TOTAL_EPOCHS) -> float:
    if epoch < 1 or epoch > total_epochs:
        raise ValueError('La época debe pertenecer al intervalo preregistrado')
    if mode in {'control', 'palm_frame126_control', 'bone_vector126_control', 'bone_angular166_control', 'bone_tetra136_control', 'bone_cov168_control', 'bone_code190_control', 'cross_signer_supcon', 'domain_adversarial_signer', 'jacobian_margin', 'deformable_temporal_offsets', 'partial_temporal_conv', 'same_corpus_auxiliary_lexicon', 'low_rank_bissm', 'deterministic_echo_reservoir', 'low_rank_temporal_relation'}:
        return LR_MAX
    if mode != 'warmup_cosine':
        raise ValueError(f'Modo desconocido: {mode}')
    if epoch <= WARMUP_EPOCHS:
        return LR_START + (LR_MAX - LR_START) * (epoch - 1) / (WARMUP_EPOCHS - 1)
    progress = (epoch - WARMUP_EPOCHS) / (total_epochs - WARMUP_EPOCHS)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + float(np.cos(np.pi * progress)))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group['lr'] = value


def load_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row['task'] == 'successor_positions126' and row['feature_status'] == 'ok']
    if len(accepted) != 1890:
        raise ValueError(f'La recuperación requiere 1,890 filas extraíbles; recibió {len(accepted)}')
    counts = {split: sum(row['split_model'] == split for row in accepted) for split in ('train', 'validation', 'test')}
    if counts != {'train': 1470, 'validation': 210, 'test': 210}:
        raise ValueError(f'Split recuperado inválido: {counts}')
    if len({row['label_lsm'] for row in accepted}) != 210:
        raise ValueError('La recuperación requiere exactamente 210 etiquetas')
    return accepted


def load_auxiliary_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row['task'] == 'successor_same_corpus_auxiliary_lexicon' and row['feature_status'] == 'ok']
    if len(accepted) != 236 or len({row['label_lsm'] for row in accepted}) != 39:
        raise ValueError(f'El auxiliar requiere 236 filas y 39 etiquetas; recibió {len(accepted)} / {len({row["label_lsm"] for row in accepted})}')
    if any(row['signer_id'] in {'S08', 'S09'} or row['split_model'] != 'auxiliary_train' for row in accepted):
        raise ValueError('El auxiliar solo puede contener S01–S07 con split auxiliary_train')
    return accepted


class CachedPositionsDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], cache_root: Path, label_ids: dict[str, int], feature_dim: int = 126):
        self.rows = rows
        self.cache_root = cache_root
        self.label_ids = label_ids
        self.feature_dim = feature_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        values = np.load(self.cache_root / row['feature_path'], allow_pickle=False)
        if values.shape != (30, self.feature_dim) or not np.isfinite(values).all():
            raise ValueError(f"{row['sample_id']}: tensor inválido {values.shape}")
        return torch.from_numpy(values.astype(np.float32, copy=False)), torch.tensor(self.label_ids[row['label_lsm']], dtype=torch.long)


class CachedSignerDataset(CachedPositionsDataset):
    """Dataset de entrenamiento con identidades S01--S07, prohibidas en S08/S09."""

    def __init__(self, rows: list[dict[str, str]], cache_root: Path, label_ids: dict[str, int], signer_ids: dict[str, int]):
        if set(signer_ids) != {f'S{index:02d}' for index in range(1, 8)} or set(signer_ids.values()) != set(range(7)):
            raise ValueError('La candidata adversarial exige exactamente firmantes train S01–S07')
        if any(row['signer_id'] not in signer_ids or row['split_model'] != 'train' for row in rows):
            raise ValueError('La cabeza adversarial solo puede recibir filas train S01–S07')
        super().__init__(rows, cache_root, label_ids, feature_dim=126)
        self.signer_ids = signer_ids

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, lexical_target = super().__getitem__(index)
        return features, lexical_target, torch.tensor(self.signer_ids[self.rows[index]['signer_id']], dtype=torch.long)


class CrossSignerBalancedBatchSampler(Sampler[list[int]]):
    """Lotes de 9 clases × 7 firmantes: positivos intraclase son siempre interfirmante."""

    CLASSES_PER_BATCH = 9
    SAMPLES_PER_CLASS = 7

    def __init__(self, rows: list[dict[str, str]], label_ids: dict[str, int]):
        self.epoch = 0
        by_label: dict[int, list[int]] = {value: [] for value in label_ids.values()}
        for index, row in enumerate(rows):
            by_label[label_ids[row['label_lsm']]].append(index)
        self.groups: list[list[int]] = []
        for label in sorted(by_label):
            indices = sorted(by_label[label], key=lambda index: rows[index]['signer_id'])
            signers = [rows[index]['signer_id'] for index in indices]
            if len(indices) != self.SAMPLES_PER_CLASS or len(set(signers)) != self.SAMPLES_PER_CLASS:
                raise ValueError(f'La clase {label} requiere exactamente siete firmantes train distintos')
            self.groups.append(indices)
        if len(self.groups) != 210:
            raise ValueError(f'Se requieren 210 clases train, no {len(self.groups)}')

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError('La época debe ser positiva')
        self.epoch = epoch

    def __len__(self) -> int:
        return 24

    def __iter__(self):
        offset = ((self.epoch - 1) * 6) % len(self.groups)
        ordered = self.groups[offset:] + self.groups[:offset]
        for start in range(0, len(ordered), self.CLASSES_PER_BATCH):
            selected = ordered[start:start + self.CLASSES_PER_BATCH]
            if len(selected) < self.CLASSES_PER_BATCH:
                selected = selected + ordered[:self.CLASSES_PER_BATCH - len(selected)]
            yield [index for group in selected for index in group]


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    macro_f1: float
    jacobian_penalty: float | None = None
    supervised_contrastive_loss: float | None = None
    domain_loss: float | None = None
    domain_accuracy: float | None = None


def fixed_rademacher_projection(classes: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator(device='cpu').manual_seed(seed)
    values = torch.empty(classes, dtype=torch.float32).bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
    return values.to(device) / np.sqrt(classes)


def jacobian_projection_penalty(features: torch.Tensor, logits: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
    if not features.requires_grad:
        raise ValueError('La penalización Jacobiana requiere gradientes de entrada en entrenamiento')
    if projection.ndim != 1 or projection.numel() != logits.shape[1]:
        raise ValueError('La proyección debe tener una coordenada por clase')
    scalar_projection = (logits * projection.unsqueeze(0)).sum()
    input_gradient = torch.autograd.grad(scalar_projection, features, create_graph=True, retain_graph=True)[0]
    return input_gradient.square().mean()


def supervised_contrastive_loss(embeddings: torch.Tensor, targets: torch.Tensor, temperature: float = SUPCON_TEMPERATURE) -> torch.Tensor:
    if embeddings.ndim != 2 or targets.ndim != 1 or embeddings.shape[0] != targets.numel():
        raise ValueError('Embeddings y objetivos contrastivos incompatibles')
    if temperature <= 0.0:
        raise ValueError('La temperatura contrastiva debe ser positiva')
    normalized = nn.functional.normalize(embeddings, p=2.0, dim=1)
    similarities = normalized @ normalized.transpose(0, 1) / temperature
    diagonal = torch.eye(targets.numel(), dtype=torch.bool, device=targets.device)
    positives = targets[:, None].eq(targets[None, :]) & ~diagonal
    if not torch.all(positives.any(dim=1)):
        raise ValueError('Cada ancla contrastiva requiere al menos un positivo intraclase')
    masked = similarities.masked_fill(diagonal, float('-inf'))
    log_probabilities = masked - torch.logsumexp(masked, dim=1, keepdim=True)
    positive_log_probabilities = torch.where(positives, log_probabilities, torch.zeros_like(log_probabilities))
    return -positive_log_probabilities.sum(dim=1).div(positives.sum(dim=1)).mean()


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None, jacobian_projection: torch.Tensor | None = None, contrastive_weight: float | None = None, domain_adversarial_lambda: float | None = None) -> EpochMetrics:
    training = optimizer is not None
    if jacobian_projection is not None and not training:
        raise ValueError('La penalización Jacobiana está prohibida fuera de entrenamiento')
    if contrastive_weight is not None and not training:
        raise ValueError('La pérdida contrastiva supervisada está prohibida fuera de entrenamiento')
    if domain_adversarial_lambda is not None and not training:
        raise ValueError('La cabeza adversarial está prohibida fuera de entrenamiento')
    if domain_adversarial_lambda is not None and (domain_adversarial_lambda < 0.0 or not isinstance(model, SignerAdversarialTCN)):
        raise ValueError('La candidata adversarial requiere SignerAdversarialTCN y lambda no negativa')
    model.train(training)
    running_loss = 0.0
    running_jacobian = 0.0
    running_contrastive = 0.0
    running_domain = 0.0
    domain_correct = 0
    count = 0
    prediction_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    for batch in loader:
        if len(batch) == 2:
            features, targets = batch
            signer_targets = None
        elif len(batch) == 3:
            features, targets, signer_targets = batch
        else:
            raise ValueError('Lote de recuperación incompatible')
        features = features.to(device)
        targets = targets.to(device)
        if signer_targets is not None:
            signer_targets = signer_targets.to(device)
        if jacobian_projection is not None:
            features = features.detach().requires_grad_(True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        signer_logits = None
        if domain_adversarial_lambda is not None:
            if signer_targets is None:
                raise ValueError('La candidata adversarial requiere objetivos de firmante train')
            logits, signer_logits = model.forward_with_signer(features, domain_adversarial_lambda)
            embeddings = None
        elif contrastive_weight is None:
            logits = model(features)
            embeddings = None
        else:
            if not hasattr(model, 'forward_with_embedding'):
                raise TypeError('El contraste supervisado requiere una TCN con embedding explícito')
            logits, embeddings = model.forward_with_embedding(features)
        supervised_loss = criterion(logits, targets)
        jacobian_penalty = None
        if jacobian_projection is not None:
            jacobian_penalty = jacobian_projection_penalty(features, logits, jacobian_projection)
            loss = supervised_loss + JACOBIAN_LAMBDA * jacobian_penalty
        else:
            loss = supervised_loss
        domain_loss = None
        if signer_logits is not None and signer_targets is not None:
            domain_loss = criterion(signer_logits, signer_targets)
            loss = loss + domain_loss
        contrastive_loss = None
        if contrastive_weight is not None:
            contrastive_loss = supervised_contrastive_loss(embeddings, targets)
            loss = loss + contrastive_weight * contrastive_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('Pérdida no finita')
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        running_loss += float(loss.detach()) * targets.size(0)
        if jacobian_penalty is not None:
            running_jacobian += float(jacobian_penalty.detach()) * targets.size(0)
        if contrastive_loss is not None:
            running_contrastive += float(contrastive_loss.detach()) * targets.size(0)
        if domain_loss is not None and signer_logits is not None and signer_targets is not None:
            running_domain += float(domain_loss.detach()) * targets.size(0)
            domain_correct += int((signer_logits.argmax(dim=1) == signer_targets).sum().item())
        count += targets.size(0)
        prediction_chunks.append(logits.argmax(dim=1).detach().cpu().numpy())
        target_chunks.append(targets.detach().cpu().numpy())
    predictions = np.concatenate(prediction_chunks)
    targets = np.concatenate(target_chunks)
    return EpochMetrics(
        loss=running_loss / count,
        macro_f1=float(f1_score(targets, predictions, average='macro', zero_division=0)),
        jacobian_penalty=None if jacobian_projection is None else running_jacobian / count,
        supervised_contrastive_loss=None if contrastive_weight is None else running_contrastive / count,
        domain_loss=None if domain_adversarial_lambda is None else running_domain / count,
        domain_accuracy=None if domain_adversarial_lambda is None else domain_correct / count,
    )


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transfer_auxiliary_encoder(auxiliary: TemporalTCN, target: TemporalTCN) -> None:
    transferable = {key: value for key, value in auxiliary.state_dict().items() if not key.startswith('head.5.')}
    result = target.load_state_dict(transferable, strict=False)
    if set(result.missing_keys) != {'head.5.weight', 'head.5.bias'} or result.unexpected_keys:
        raise AssertionError(f'Transferencia auxiliar inválida: {result}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--mode', choices=('control', 'palm_frame126_control', 'bone_vector126_control', 'bone_angular166_control', 'bone_tetra136_control', 'bone_cov168_control', 'bone_code190_control', 'cross_signer_supcon', 'domain_adversarial_signer', 'warmup_cosine', 'jacobian_margin', 'deformable_temporal_offsets', 'partial_temporal_conv', 'same_corpus_auxiliary_lexicon', 'low_rank_bissm', 'deterministic_echo_reservoir', 'low_rank_temporal_relation'), required=True)
    parser.add_argument('--auxiliary-manifest', type=Path, default=None)
    parser.add_argument('--auxiliary-cache-root', type=Path, default=None)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=TOTAL_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--skip-test-evaluation', action='store_true')
    args = parser.parse_args()
    if not args.skip_test_evaluation:
        raise ValueError('La recuperación exige --skip-test-evaluation para proteger S09')
    if args.epochs != TOTAL_EPOCHS:
        raise ValueError('Las 40 épocas son parte del protocolo de recuperación preregistrado')
    seed_everything(args.seed)
    rows = load_rows(args.manifest)
    train_rows = [row for row in rows if row['split_model'] == 'train']
    validation_rows = [row for row in rows if row['split_model'] == 'validation']
    labels = {label: index for index, label in enumerate(sorted({row['label_lsm'] for row in train_rows}))}
    if set(labels) != {row['label_lsm'] for row in validation_rows}:
        raise ValueError('S08 debe contener las mismas 210 etiquetas sin usar S09')
    feature_dim = 190 if args.mode == 'bone_code190_control' else (168 if args.mode == 'bone_cov168_control' else (166 if args.mode == 'bone_angular166_control' else (136 if args.mode == 'bone_tetra136_control' else 126)))
    signer_ids = {signer: index for index, signer in enumerate(sorted({row['signer_id'] for row in train_rows}))}
    train_dataset = CachedSignerDataset(train_rows, args.cache_root, labels, signer_ids) if args.mode == 'domain_adversarial_signer' else CachedPositionsDataset(train_rows, args.cache_root, labels, feature_dim)
    validation_dataset = CachedPositionsDataset(validation_rows, args.cache_root, labels, feature_dim)
    train_generator = torch.Generator().manual_seed(args.seed)
    pair_sampler = CrossSignerBalancedBatchSampler(train_rows, labels) if args.mode == 'cross_signer_supcon' else None
    train_loader = DataLoader(train_dataset, batch_sampler=pair_sampler, num_workers=0) if pair_sampler else DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=train_generator, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device(args.device)
    model_type = {
        'deformable_temporal_offsets': DeformableTemporalOffsetsTCN,
        'partial_temporal_conv': PartialTemporalConvTCN,
        'low_rank_bissm': LowRankBiSSMTemporalTCN,
        'deterministic_echo_reservoir': DeterministicEchoReservoirTCN,
        'low_rank_temporal_relation': LowRankTemporalRelationTCN,
        'domain_adversarial_signer': SignerAdversarialTCN,
    }.get(args.mode, TemporalTCN)
    auxiliary_metadata = None
    if args.mode == 'same_corpus_auxiliary_lexicon':
        if args.auxiliary_manifest is None or args.auxiliary_cache_root is None:
            raise ValueError('La candidata auxiliar requiere --auxiliary-manifest y --auxiliary-cache-root')
        auxiliary_rows = load_auxiliary_rows(args.auxiliary_manifest)
        auxiliary_labels = {label: index for index, label in enumerate(sorted({row['label_lsm'] for row in auxiliary_rows}))}
        auxiliary_dataset = CachedPositionsDataset(auxiliary_rows, args.auxiliary_cache_root, auxiliary_labels)
        auxiliary_loader = DataLoader(auxiliary_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(args.seed + 10_000), num_workers=0)
        auxiliary_model = TemporalTCN(feature_dim=126, classes=len(auxiliary_labels), frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device)
        auxiliary_optimizer = torch.optim.AdamW(auxiliary_model.parameters(), lr=LR_MAX, weight_decay=args.weight_decay)
        auxiliary_criterion = nn.CrossEntropyLoss()
        auxiliary_history = []
        for auxiliary_epoch in range(1, args.epochs + 1):
            auxiliary_metrics = run_epoch(auxiliary_model, auxiliary_loader, auxiliary_criterion, device, auxiliary_optimizer)
            record = {'epoch': auxiliary_epoch, 'train_loss': auxiliary_metrics.loss, 'train_macro_f1': auxiliary_metrics.macro_f1}
            auxiliary_history.append(record)
            print(json.dumps({'stage': 'auxiliary_train', **record}), flush=True)
        model = TemporalTCN(feature_dim=126, classes=len(labels), frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device)
        transfer_auxiliary_encoder(auxiliary_model, model)
        auxiliary_metadata = {
            'manifest_sha256': manifest_sha256(args.auxiliary_manifest),
            'cache_root': str(args.auxiliary_cache_root),
            'train_samples': len(auxiliary_rows),
            'classes': len(auxiliary_labels),
            'forbidden_signers': ['S08', 'S09'],
            'transferred_keys_excluding_final_classifier': [key for key in model.state_dict() if not key.startswith('head.5.')],
            'history': auxiliary_history,
        }
    else:
        model = model_type(feature_dim=feature_dim, classes=len(labels), frames=30, channels=64, dilations=(1, 2, 4, 8), dropout=0.20).to(device)
    expected_parameters = {
        'deformable_temporal_offsets': 160_538,
        'low_rank_bissm': 161_059,
        'deterministic_echo_reservoir': 160_531,
        'low_rank_temporal_relation': 165_425,
        'bone_angular166_control': 166_674,
        'bone_tetra136_control': 160_914,
        'bone_cov168_control': 167_058,
        'bone_code190_control': 171_282,
        'domain_adversarial_signer': 159_897,
    }.get(args.mode, 158_994)
    if parameter_count(model) != expected_parameters:
        raise AssertionError(f'Presupuesto inesperado: {parameter_count(model)}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_MAX, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    projection = fixed_rademacher_projection(len(labels), device, args.seed + 1) if args.mode == 'jacobian_margin' else None
    args.out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        learning_rate = learning_rate_for_epoch(epoch, args.mode, args.epochs)
        set_optimizer_lr(optimizer, learning_rate)
        if pair_sampler is not None:
            pair_sampler.set_epoch(epoch)
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, projection, SUPCON_LAMBDA if args.mode == 'cross_signer_supcon' else None, DOMAIN_ADVERSARIAL_LAMBDA if args.mode == 'domain_adversarial_signer' else None)
        with torch.no_grad():
            validation_metrics = run_epoch(model, validation_loader, criterion, device)
        record = {
            'epoch': epoch,
            'learning_rate': learning_rate,
            'train_loss': train_metrics.loss,
            'train_macro_f1': train_metrics.macro_f1,
            'train_jacobian_penalty': train_metrics.jacobian_penalty,
            'train_supervised_contrastive_loss': train_metrics.supervised_contrastive_loss,
            'train_domain_loss': train_metrics.domain_loss,
            'train_domain_accuracy': train_metrics.domain_accuracy,
            'validation_loss': validation_metrics.loss,
            'validation_macro_f1': validation_metrics.macro_f1,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if validation_metrics.macro_f1 > best_f1:
            best_f1 = validation_metrics.macro_f1
            best_epoch = epoch
            stale = 0
            torch.save({'model_state_dict': model.state_dict(), 'labels': labels, 'epoch': epoch}, args.out / 'best.pt')
        else:
            stale += 1
            if stale >= args.patience:
                break
    metrics = {
        'kind': 'successor_recovery_benchmark',
        'mode': args.mode,
        'representation': 'bone_code190' if args.mode == 'bone_code190_control' else ('bone_cov168' if args.mode == 'bone_cov168_control' else ('bone_angular166' if args.mode == 'bone_angular166_control' else ('bone_tetra136' if args.mode == 'bone_tetra136_control' else ('bone_vector126' if args.mode == 'bone_vector126_control' else ('palm_frame126' if args.mode == 'palm_frame126_control' else 'positions126'))))),
        'seed': args.seed,
        'test_evaluated': False,
        'manifest_sha256': manifest_sha256(args.manifest),
        'cache_root': str(args.cache_root),
        'train_samples': len(train_rows),
        'validation_samples': len(validation_rows),
        'test_samples_closed': 210,
        'classes': len(labels),
        'parameter_count': parameter_count(model),
        'best_validation_macro_f1': best_f1,
        'best_epoch': best_epoch,
        'history': history,
        'environment': {'python': sys.version, 'torch': torch.__version__, 'platform': platform.platform()},
        'auxiliary_pretraining': auxiliary_metadata,
        'supervised_contrastive': None if args.mode != 'cross_signer_supcon' else {'lambda': SUPCON_LAMBDA, 'temperature': SUPCON_TEMPERATURE, 'classes_per_batch': CrossSignerBalancedBatchSampler.CLASSES_PER_BATCH, 'samples_per_class': CrossSignerBalancedBatchSampler.SAMPLES_PER_CLASS, 's08_pairs_used': False, 's09_pairs_used': False},
        'domain_adversarial': None if args.mode != 'domain_adversarial_signer' else {'lambda': DOMAIN_ADVERSARIAL_LAMBDA, 'signer_ids': signer_ids, 'head_train_only': True, 's08_signer_targets_used': False, 's09_signer_targets_used': False},
        'elapsed_seconds': time.time() - started,
    }
    (args.out / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'best_validation_macro_f1': best_f1, 'best_epoch': best_epoch, 'test_evaluated': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()