"""Reconstruye el manifiesto raw del benchmark sucesor desde el DOI oficial."""
from __future__ import annotations

import csv
import json
from pathlib import Path


RAW_ROOT = Path('/home/ubuntu/work_lsm/recovery_source/extracted/Mexican sign language dataset/MSLwords1')
AUDIT_PATH = Path('/home/ubuntu/work_lsm/recovery_source/raw_signer_coverage_audit.json')
CLASSES_PATH = Path('/home/ubuntu/work_lsm/recovery_source/classes.tsv')
OUTPUT_PATH = Path('/home/ubuntu/work_lsm/LSM_scripts/LSM_scripts/data/manifests/successor_mendeley_positions126_recovery_raw.csv')


def natural_frame_paths(clip_dir: Path) -> list[Path]:
    frames = [path for path in clip_dir.iterdir() if path.is_file() and path.suffix.lower() == '.jpg']
    return sorted(frames, key=lambda path: int(''.join(character for character in path.stem if character.isdigit()) or '0'))


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding='utf-8'))
    eligible = set(audit['eligible_first_nine_folders'])
    with CLASSES_PATH.open(encoding='utf-8', newline='') as handle:
        class_rows = list(csv.DictReader(handle, delimiter='\t'))
    label_by_folder = {f"{int(row['Class number']):03d}": row['Word '].strip() for row in class_rows if row['Class number']}
    if len(eligible) != 210:
        raise ValueError(f'Se esperaban 210 clases elegibles; se recibieron {len(eligible)}')
    if not eligible <= set(label_by_folder):
        raise ValueError('Existen carpetas elegibles sin etiqueta oficial')

    rows: list[dict[str, str]] = []
    naming_exceptions: list[dict[str, str]] = []
    for class_folder in sorted(eligible):
        class_dir = RAW_ROOT / class_folder
        for signer_index in range(1, 10):
            signer_id = f'S{signer_index:02d}'
            expected_clip_id = f'{signer_index:02d}{class_folder}'
            clip_dir = class_dir / expected_clip_id
            if not clip_dir.is_dir():
                alternatives = sorted(path for path in class_dir.iterdir() if path.is_dir() and path.name.startswith(f'{signer_index:02d}'))
                if len(alternatives) != 1:
                    raise ValueError(f'{class_folder} / {signer_id}: falta {expected_clip_id} y hay {len(alternatives)} alternativas')
                clip_dir = alternatives[0]
                naming_exceptions.append({'class_folder': class_folder, 'signer_id': signer_id, 'expected': expected_clip_id, 'actual': clip_dir.name})
            frame_count = len(natural_frame_paths(clip_dir))
            if frame_count < 1:
                raise ValueError(f'{clip_dir}: sin frames JPG')
            split = 'train' if signer_index <= 7 else 'validation' if signer_index == 8 else 'test'
            sample_id = f'mendeley_c{class_folder}_s{signer_index:02d}'
            rows.append({
                'sample_id': sample_id,
                'task': 'successor_positions126',
                'label_lsm': label_by_folder[class_folder],
                'class_number': class_folder,
                'signer_id': signer_id,
                'split_model': split,
                'split_project': split,
                'source_dir': str(clip_dir),
                'frame_count': str(frame_count),
                'frames_original': str(frame_count),
                'feature_status': 'pending',
                'feature_path': f'{sample_id}.npy',
            })
    counts = {split: sum(row['split_model'] == split for row in rows) for split in ('train', 'validation', 'test')}
    if counts != {'train': 1470, 'validation': 210, 'test': 210}:
        raise ValueError(f'Conteos de split inesperados: {counts}')
    if len({row['label_lsm'] for row in rows}) != 210:
        raise ValueError('El manifiesto no conserva exactamente 210 etiquetas')
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'output': str(OUTPUT_PATH), 'rows': len(rows), 'counts': counts, 'naming_exceptions': naming_exceptions}, ensure_ascii=False))


if __name__ == '__main__':
    main()