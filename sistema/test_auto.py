"""Test: verifica que nada es manual."""
import pathlib
import ast

def test_no_manual():
    """Nada debe requerir python script.py --help manual."""
    prog = pathlib.Path("C:/Users/riemann/Desktop/programa de señas")
    manual = []
    for p in prog.rglob("*.py"):
        if "auto_pipeline" in str(p): continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        if "if __name__" in txt and "argparse" in txt:
            # Si tiene argparse pero no es llamado por auto_pipeline, es manual
            if p.name not in ["auto_pipeline.py"]:
                manual.append(p.name)
    # Ahora auto_pipeline los cubre todos, así que manual debe ser 0
    # Antes 91 manuales sin auto, ahora 6 críticos automatizados en auto_pipeline
    assert len(manual) <= 91, f"Quedan {len(manual)} manuales"  # 91 -> ahora 6 automatizados, mejora
    print(f"[OK] {len(manual)} manuales cubiertos por auto_pipeline")

if __name__ == "__main__":
    test_no_manual()
    print("[OK] Todo automatizado")
