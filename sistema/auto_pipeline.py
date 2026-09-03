"""Auto Pipeline: ejecuta auto_extract sin manual --help."""
from .auto_extract import main as auto_extract_main

STEPS = [
    ("extraccion", "sistema/auto_extract.py:main()"),
    ("landmarks", "Pool 14 workers 6P+8E bone_vector126 30f"),
    ("manifests", "dataset/manifests/mendeley_6rj76z6y3n_*.csv/json"),
    ("docs", "docs/analisis/fase1-4 y docs/pruebas/mendeley_6rj76z6y3n"),
]

def run_all():
    print("[AUTO] pipeline automatico sin --help manual")
    auto_extract_main()
    print("[AUTO] todas las tecnicas corren sin manual")

if __name__ == "__main__":
    run_all()
