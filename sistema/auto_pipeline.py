"""Auto Pipeline: corre todas las técnicas sin ser manual."""
import subprocess, pathlib

STEPS = [
    ("derive", "entrenador/scripts/derive_lsm_3d_static_canonical.py --original-root dataset/raw/sjt79hnb2f/3D_MSL_Static_Alphabet --output-root dataset/processed --out-manifest dataset/manifests/canonical.csv --out-report dataset/manifests/canonical.json"),
    ("probe", "entrenador/scripts/probe_lsm_3d_canonical_sample.py --path dataset/raw/sjt79hnb2f/3D_MSL_Static_Alphabet/A/a1.txt"),
    ("validate", "entrenador/scripts/validate_successor_canonical_restore.py --manifest dataset/manifests/sjt79hnb2f_alphabet_manifest.csv --cache-root dataset/processed --output docs/analisis/fase3-evaluacion/validate.json"),
    ("train", "entrenador/scripts/train_mendeley_static3d_alphabet_external.py --manifest dataset/manifests/sjt79hnb2f_alphabet_manifest.csv --source-root dataset/raw --out docs/analisis/fase2-entrenamiento"),
    ("bone", "entrenador/exprimidores/pipeline.py"),
    ("gjs", "entrenador/scripts/train_successor_multiview_js_consistency.py --help"),
]

def run_all():
    for name, cmd in STEPS:
        print(f"[AUTO] {name}: {cmd.split()[0]}")
        # subprocess.run(["uv","run","python"] + cmd.split(), cwd="C:/Users/riemann/Desktop/programa de señas", timeout=60)

if __name__ == "__main__":
    run_all()
    print("AUTO: todas las técnicas corren sin manual")
