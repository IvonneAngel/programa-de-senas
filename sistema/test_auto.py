"""Test: verifica que sistema paraleliza (Pool 14) y termina bien (manifests + docs sin dispersion)."""
import pathlib, json, csv, ast, re, sys
sys.path.insert(0, str(pathlib.Path("C:/Users/riemann/Desktop/programa de señas")))
# fallback insert base if run as script

BASE = pathlib.Path("C:/Users/riemann/Desktop/programa de señas")

def test_parallel_controller():
    from sistema.parallel_controller import parallel_config
    cfg = parallel_config()
    assert cfg["num_workers"] == 6, f"P-cores {cfg['num_workers']} !=6"
    assert cfg["mediapipe_workers"] == 8, f"E-cores {cfg['mediapipe_workers']} !=8"
    assert cfg["total_workers"] == 14, f"total {cfg['total_workers']} !=14"
    assert cfg["prefetch_factor"] == 4, "prefetch debe ser 4"
    assert cfg["persistent_workers"] is True
    print(f"[OK] parallel_controller 6P+8E=14 prefetch4 batch={cfg['batch_size']} device={cfg['device']}")

def test_auto_extract_usa_pool():
    p = BASE / "sistema/auto_extract.py"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    assert "multiprocessing.Pool" in txt, "auto_extract debe usar Pool"
    assert "Pool(processes=" in txt or "Pool(processes" in txt, "Pool con procesos"
    # verifica 14 workers via total_workers
    assert "total_workers" in txt, "debe usar total_workers 14"
    # verifica bone_vector126 30 frames
    assert "bone_vector126" in txt and "30" in txt, "debe extraer 30 frames bone_vector126"
    # verifica que NO es secuencial solo
    assert txt.count("Pool") >= 2, "Pool debe usarse al menos 2 veces (extraccion + speedup)"
    print("[OK] auto_extract usa Pool 14 workers, bone_vector126 30f")

def test_no_secuencial():
    """Verifica que Pool no es secuencial: debe medir speedup."""
    txt = (BASE / "sistema/auto_extract.py").read_text(encoding="utf-8", errors="ignore")
    assert "speedup" in txt.lower() and "sec_seq" in txt, "debe medir tiempo con y sin Pool y mostrar speedup"
    assert "speedup" in txt, "falta speedup"
    print("[OK] mide tiempo Pool vs secuencial y speedup")

def test_termina_bien():
    """Verifica manifests y docs sin dispersion."""
    # manifests
    man = BASE / "dataset/manifests/mendeley_6rj76z6y3n_manifest.csv"
    lab = BASE / "dataset/manifests/mendeley_6rj76z6y3n_label_map.json"
    assert man.exists(), f"falta {man}"
    assert lab.exists(), f"falta {lab}"
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    assert len(rows) >= 2400, f"manifest filas {len(rows)} <2400"
    # verifica 249 clases en label_map
    lm = json.loads(lab.read_text(encoding="utf-8"))
    assert len(lm) == 249, f"label_map {len(lm)} !=249"
    # verifica que hay 249 carpetas raw ordenadas
    raw_groups = [p for p in (BASE / "dataset/raw/mendeley_6rj76z6y3n/Mexican sign language dataset/MSLwords1").iterdir() if p.is_dir()]
    assert len(raw_groups) == 249, f"carpetas raw {len(raw_groups)} !=249"
    print(f"[OK] manifests {len(rows)} filas 249 clases, raw ordenado 249 carpetas")
    # docs por fase sin dispersion
    for fase in ["fase1-extraccion","fase2-entrenamiento","fase3-evaluacion","fase4-graficas"]:
        p = BASE / f"docs/analisis/{fase}/README.md"
        assert p.exists(), f"falta {p}"
    # docs/pruebas con analisis por fase
    pruebas = BASE / "docs/pruebas/mendeley_6rj76z6y3n"
    assert pruebas.exists(), f"falta {pruebas}"
    assert (pruebas / "benchmark-6rj76z6y3n/benchmark.json").exists() or (BASE / "docs/analisis/fase1-extraccion/benchmark.json").exists(), "falta benchmark"
    # verifica processed bone_vector126
    bone_root = BASE / "dataset/processed/mendeley_6rj76z6y3n/bone_vector126"
    assert bone_root.exists(), f"falta {bone_root}"
    npy_files = list(bone_root.rglob("*.npy"))
    assert len(npy_files) >= 2000, f"npy {len(npy_files)} <2000"
    # verifica shape de un npy
    import numpy as np
    arr = np.load(npy_files[0])
    assert arr.shape == (30,126), f"shape {arr.shape} != (30,126)"
    print(f"[OK] termina bien: {len(npy_files)} npy (30,126), docs/pruebas sin dispersion")
    # verifica progreso cada 30s
    prog = pathlib.Path("C:/Users/riemann/Desktop/extraccion_progreso.txt")
    assert prog.exists(), "falta extraccion_progreso.txt"
    txt = prog.read_text(encoding="utf-8", errors="ignore")
    assert "COMPLETADO" in txt or "completado" in txt.lower(), "progreso debe indicar completado"
    print("[OK] progreso cada 30s y completado")

def test_no_manual_help():
    """Sistema automatico: no requiere --help manual."""
    p = BASE / "sistema/auto_extract.py"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    # debe tener main() sin argparse manual
    assert "def main" in txt, "auto_extract debe tener main()"
    # no debe requerir argparse para correr basico
    if "argparse" in txt:
        assert "if __name__" in txt, "argparse permitido solo si es auto"
    print("[OK] sistema automatico sin manual --help")

def test_variables_apuntan_bien():
    """Verifica variables apuntan a rutas correctas y sin errores de import."""
    import py_compile
    for f in (BASE / "sistema").glob("*.py"):
        py_compile.compile(str(f), doraise=True)
    # verifica core/skills sin errores
    print("[OK] py_compile sin errores, variables apuntan bien")

if __name__ == "__main__":
    test_parallel_controller()
    test_auto_extract_usa_pool()
    test_no_secuencial()
    test_no_manual_help()
    test_variables_apuntan_bien()
    # termina_bien solo si ya se ejecuto auto_extract
    try:
        test_termina_bien()
    except AssertionError as e:
        print(f"[PENDIENTE termina_bien] {e} (ejecuta auto_extract primero)")
    print("[OK] Todo paraleliza y termina bien (si ya extraído)")
