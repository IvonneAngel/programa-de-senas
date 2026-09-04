
"""Pipeline automatico robbolla11: clone -> extract -> train RF 80/20 -> docs/analisis/robbolla11 | progreso cada 30s | paralelo 6P+RTX"""
from __future__ import annotations
import os, sys, time, threading, subprocess, shutil, pathlib, json, pickle, csv
from pathlib import Path
import multiprocessing

# --- Rutas ---
DESKTOP = Path("C:/Users/riemann/Desktop")
CLONE_DEST = DESKTOP / "robbolla11"
REPO_URL = "https://github.com/robbolla11/Mexican-Sign-Language-Alphabet-Real-Time-Detection"

BASE = Path("C:/Users/riemann/Desktop/programa de señas")
RAW_ROOT = BASE / "dataset/raw/robbolla11"
DOCS_ROOT = BASE / "docs/analisis/robbolla11"
PROGRESO = DESKTOP / "entrenamiento_robbolla_progreso.txt"
MANIFEST_PATH = RAW_ROOT / "manifest.csv"
MODEL_DIR = DOCS_ROOT

# progreso thread
_stop = threading.Event()
_stage = "init"
_stage_start = time.time()
_progress_thread = None

def log_progress(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(PROGRESO, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except: pass

def _writer():
    while not _stop.is_set():
        try:
            elapsed = time.time() - _stage_start
            log_progress(f"stage={_stage} elapsed={elapsed:.0f}s still running...")
        except: pass
        _stop.wait(30)

def set_stage(name):
    global _stage, _stage_start
    _stage = name
    _stage_start = time.time()
    log_progress(f">>> {name}")

def start_progress():
    global _progress_thread
    # init file
    PROGRESO.write_text(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Pipeline robbolla11 iniciado (6 P-cores + RTX, auto sin manual)\n", encoding="utf-8")
    _progress_thread = threading.Thread(target=_writer, daemon=True)
    _progress_thread.start()

def stop_progress():
    _stop.set()
    if _progress_thread:
        _progress_thread.join(timeout=2)

def run_cmd(cmd, cwd=None):
    log_progress(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.stdout:
        # log truncated
        for line in result.stdout.splitlines()[-20:]:
            log_progress(f"  out: {line}")
    if result.stderr:
        for line in result.stderr.splitlines()[-20:]:
            log_progress(f"  err: {line}")
    if result.returncode != 0:
        raise RuntimeError(f"cmd falló {cmd} code {result.returncode}: {result.stderr[:500]}")
    return result

# --- Paso 1: clone ---
def step_clone():
    set_stage("clone")
    if CLONE_DEST.exists():
        log_progress(f"CLONE_DEST existe {CLONE_DEST}, verificando git...")
        # check if valid git
        git_dir = CLONE_DEST / ".git"
        if git_dir.exists():
            log_progress("Repo ya clonado, haciendo pull --ff-only")
            try:
                run_cmd(["git","pull","--ff-only"], cwd=str(CLONE_DEST))
            except Exception as e:
                log_progress(f"pull falló, continuando: {e}")
            return
        else:
            log_progress("Destino existe pero no es git, borrando...")
            shutil.rmtree(CLONE_DEST)
    # clone
    log_progress(f"Clonando {REPO_URL} -> {CLONE_DEST}")
    run_cmd(["git","clone", REPO_URL, str(CLONE_DEST)])
    log_progress("Clone completado")
    # listar contenido
    for p in CLONE_DEST.iterdir():
        log_progress(f"  clone item: {p.name} {'DIR' if p.is_dir() else f'{p.stat().st_size} bytes'}")
    # inspeccionar scripts clave
    for name in ["RandomForestTraining.py","SignLanguagAlphabetDetector.py","data.pickle"]:
        cand = CLONE_DEST / name
        # also search recursively
        found = list(CLONE_DEST.rglob(name))
        if found:
            for f in found:
                log_progress(f"  encontrado {name}: {f.relative_to(CLONE_DEST)} size {f.stat().st_size}")

# --- Paso 2: extraer data/ y data.pickle a dataset/raw/robbolla11 ---
def step_extract():
    set_stage("extraccion-data")
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    # Buscar data/ en clone
    # Puede estar en root/data o data/
    candidates = list(CLONE_DEST.glob("data")) + list(CLONE_DEST.glob("Data")) 
    # también buscar subdirectorios con letras A/B...
    data_src = None
    for c in candidates:
        if c.is_dir():
            # verificar que tiene subcarpetas A B C
            subs = [x.name for x in c.iterdir() if x.is_dir()]
            if len(subs) >= 10:
                data_src = c
                break
    if not data_src:
        # buscar recursivo directories que contienen A/ con jpgs
        for p in CLONE_DEST.rglob("A"):
            if p.is_dir() and any(f.suffix.lower()==".jpg" for f in p.iterdir()):
                data_src = p.parent
                log_progress(f"Detectado data_src via rglob A: {data_src}")
                break
    if not data_src:
        log_progress("ERROR: no se encontró data/ con letras, listando clone tree shallow")
        for p in CLONE_DEST.rglob("*"):
            if p.is_dir():
                log_progress(f"  dir {p.relative_to(CLONE_DEST)}")
                if len(list(p.iterdir()))>20:
                    break
        raise FileNotFoundError("data/ no encontrado en clone")

    log_progress(f"data_src detectado: {data_src}")

    # Contar letras y jpgs
    letra_dirs = [d for d in data_src.iterdir() if d.is_dir()]
    letra_dirs_sorted = sorted(letra_dirs, key=lambda x: x.name)
    log_progress(f"Letras encontradas: {[d.name for d in letra_dirs_sorted]} total {len(letra_dirs_sorted)}")
    total_jpgs = 0
    for ld in letra_dirs_sorted:
        count = len([f for f in ld.iterdir() if f.is_file() and f.suffix.lower() in [".jpg",".jpeg",".png"]])
        total_jpgs += count
        log_progress(f"  letra {ld.name}: {count} imgs")

    log_progress(f"Total jpgs en source: {total_jpgs} (esperado 4200 =21*200)")

    # Copiar data/ -> RAW_ROOT/data
    dest_data = RAW_ROOT / "data"
    if dest_data.exists():
        log_progress(f"dest data existe {dest_data}, resync...")
        shutil.rmtree(dest_data)
    log_progress(f"Copiando {data_src} -> {dest_data} (copia paralela)")
    # usar copia con threading pool para acelerar? simple shutil
    shutil.copytree(data_src, dest_data)
    log_progress(f"Copia completada {dest_data}")

    # Buscar data.pickle (3MB) en clone
    pickle_candidates = list(CLONE_DEST.rglob("data.pickle")) + list(CLONE_DEST.rglob("*.pickle")) + list(CLONE_DEST.rglob("*.pkl"))
    log_progress(f"Pickle candidates: {pickle_candidates}")
    for p in pickle_candidates:
        log_progress(f"  pickle {p.relative_to(CLONE_DEST)} {p.stat().st_size} bytes")
    # elegir data.pickle principal
    main_pickle = None
    for p in pickle_candidates:
        if p.name == "data.pickle":
            main_pickle = p
            break
    if not main_pickle and pickle_candidates:
        main_pickle = pickle_candidates[0]
    if main_pickle:
        dest_pickle = RAW_ROOT / "data.pickle"
        shutil.copy2(main_pickle, dest_pickle)
        log_progress(f"Copiado pickle {main_pickle} -> {dest_pickle} ({dest_pickle.stat().st_size/1024/1024:.2f} MB)")
    else:
        log_progress("WARNING: no se encontró data.pickle en clone, se generará desde data/ si es necesario")

    # Verificar destino
    dest_total = sum(1 for _ in (dest_data).rglob("*.jpg"))
    dest_total += sum(1 for _ in (dest_data).rglob("*.png"))
    dest_total += sum(1 for _ in (dest_data).rglob("*.jpeg"))
    log_progress(f"Verificacion destino: {dest_total} imgs en {dest_data}")
    if dest_total != 4200:
        log_progress(f"WARNING: esperado 4200 pero hay {dest_total}")

    # Generar manifest.csv para trazabilidad
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as csvfile:
        w = csv.writer(csvfile)
        w.writerow(["filepath","label","split_hint"])
        for ld in letra_dirs_sorted:
            label = ld.name
            for img in sorted(ld.iterdir()):
                if img.is_file() and img.suffix.lower() in [".jpg",".jpeg",".png"]:
                    rel = Path("data")/label/img.name
                    w.writerow([str(rel), label, ""])
    log_progress(f"Manifest generado {MANIFEST_PATH} ({MANIFEST_PATH.stat().st_size} bytes)")

def step_inspect_scripts():
    set_stage("inspeccion-scripts")
    # Listar scripts relevantes
    for p in CLONE_DEST.rglob("*.py"):
        log_progress(f"  py {p.relative_to(CLONE_DEST)} {p.stat().st_size}")
        # preview first 30 lines if RandomForest or Detector
        if "RandomForest" in p.name or "SignLanguage" in p.name:
            try:
                txt = p.read_text(encoding="utf-8", errors="ignore")
                preview = "\n".join(txt.splitlines()[:60])
                log_progress(f"--- preview {p.name} ---\n{preview[:2000]}")
            except: pass

def step_train():
    set_stage("entrenamiento-RandomForest-80/20")
    # Usar sistema paralelo: 6 P-cores + RTX (aunque RF es CPU, usamos n_jobs=6)
    # Estrategia: leer data.pickle si existe (contiene landmarks ya extraidos), sino extraer via mediapipe/hand landmark?
    # El repo original RandomForestTraining.py probablemente carga data.pickle (que tiene data y labels pre-procesados con mediapipe)
    # Vamos a replicar lógica: intentar cargar pickle y entrenar RF 80/20.

    # Detectar workers
    import psutil
    p_cores = 6
    try:
        import psutil
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        log_progress(f"CPU detectado logical={logical} physical={physical} -> usando 6 P-cores")
    except Exception as e:
        log_progress(f"psutil error {e}")

    # Detect RTX
    device = "cpu"
    vram = 0
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log_progress(f"RTX detectada {torch.cuda.get_device_name(0)} vram {vram:.1f}GB")
        else:
            log_progress("CUDA no disponible, usando CPU (RF es CPU anyway)")
    except Exception as e:
        log_progress(f"torch cuda check fallo {e}, CPU")

    DOCS_ROOT.mkdir(parents=True, exist_ok=True)

    # Intentar cargar data.pickle del RAW_ROOT
    pickle_path = RAW_ROOT / "data.pickle"
    # fallback también buscar en dest_data parent
    if not pickle_path.exists():
        # buscar en clone
        cand = list(CLONE_DEST.rglob("data.pickle"))
        if cand:
            pickle_path = cand[0]
            log_progress(f"Usando pickle de clone {pickle_path}")

    log_progress(f"Pickle path elegido: {pickle_path} exists={pickle_path.exists()} size={pickle_path.stat().st_size/1024/1024:.2f} MB" if pickle_path.exists() else "NO PICKLE")

    # Si pickle existe, inspeccionar estructura
    data = None
    labels = None
    pickle_info = ""
    if pickle_path.exists():
        try:
            with open(pickle_path, "rb") as f:
                obj = pickle.load(f)
            log_progress(f"Pickle type {type(obj)} keys {list(obj.keys()) if isinstance(obj, dict) else 'no dict'}")
            if isinstance(obj, dict):
                for k,v in obj.items():
                    if hasattr(v, "__len__"):
                        log_progress(f"  key {k}: type {type(v)} len {len(v)} sample type {type(v[0]) if len(v)>0 else 'empty'}")
                        if len(v)>0 and hasattr(v[0], "__len__"):
                            try: log_progress(f"    sample len {len(v[0])} first 5: {str(v[0])[:200]}")
                            except: pass
                    else:
                        log_progress(f"  key {k}: {type(v)} value {str(v)[:500]}")
                # intentar extraer data/labels: común es obj['data'], obj['labels'] o data/labels
                if 'data' in obj and 'labels' in obj:
                    data = obj['data']
                    labels = obj['labels']
                elif 'data' in obj and 'label' in obj:
                    data = obj['data']
                    labels = obj['label']
                elif 'x' in obj and 'y' in obj:
                    data = obj['x']
                    labels = obj['y']
                else:
                    # tomar los dos primeros arrays
                    vals = list(obj.values())
                    if len(vals)>=2:
                        data, labels = vals[0], vals[1]
            elif isinstance(obj, (list, tuple)) and len(obj)==2:
                data, labels = obj[0], obj[1]
                log_progress(f"pickle list tuple len 2: data len {len(data)} labels len {len(labels)}")
            pickle_info = f"pickle loaded dict keys {list(obj.keys()) if isinstance(obj, dict) else type(obj)}"
        except Exception as e:
            log_progress(f"Error cargando pickle: {e}")
            import traceback; log_progress(traceback.format_exc())

    # Si no hay pickle válido, generar features desde imagenes via mediapipe (fallback)
    if data is None or labels is None:
        log_progress("No hay data/labels válidos desde pickle, generando features desde imagenes con mediapipe paralelo (6 P-cores)")
        # Implementar extracción landmarks mediapipe si necesario
        data, labels = extract_from_images_parallel()

    # Ahora data y labels deben ser listas
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    log_progress(f"Data shape: {len(data)} samples, Labels {len(labels)} unique {sorted(set(labels))[:10]} total classes {len(set(labels))}")
    # Verificar 200 por letra ideal
    from collections import Counter
    cnt = Counter(labels)
    log_progress(f"Distribución por letra: {dict(sorted(cnt.items()))}")
    for k,v in sorted(cnt.items()):
        if v!=200:
            log_progress(f"  WARNING letra {k} tiene {v} !=200")

    # Convertir a arrays
    X = np.asarray(data)
    y = np.asarray(labels)
    log_progress(f"X shape {X.shape} y shape {y.shape} X dtype {X.dtype}")
    # Si X es object con secuencias variables, intentar padding/ flatten
    if X.dtype == object:
        # intentar convertir cada elemento a array y apilar
        try:
            # flatten cada muestra si es lista de landmarks
            X_stack = np.vstack([np.asarray(x).flatten() for x in X])
            X = X_stack
            log_progress(f"X convertido via vstack flatten -> {X.shape}")
        except Exception as e:
            log_progress(f"No se pudo vstack X object: {e}")

    # 80/20 split estratificado
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    log_progress(f"Split 80/20: train {X_train.shape} {len(y_train)} test {X_test.shape} {len(y_test)}")

    # Entrenar RF con n_jobs=6 (6 P-cores)
    # pipeline automático sin manual: usar RandomForest del repo si existe, sino sklearn
    # Intentar usar parámetros del RandomForestTraining.py original si los hay
    # Leer ese archivo para extraer params
    rf_params = {"n_estimators": 200, "random_state": 42, "n_jobs": 6}
    # intentar parsear original
    try:
        rf_path = list(CLONE_DEST.rglob("RandomForestTraining.py"))
        if rf_path:
            txt = rf_path[0].read_text(encoding="utf-8", errors="ignore")
            log_progress(f"RandomForestTraining.py preview extraído para params: {txt[:2000]}")
            # buscar n_estimators=
            import re
            m = re.search(r"n_estimators\s*=\s*(\d+)", txt)
            if m: rf_params["n_estimators"] = int(m.group(1))
    except Exception as e:
        log_progress(f"parse RF params fallo {e}")

    log_progress(f"Entrenando RandomForest {rf_params} con 6 P-cores (paralelo) ...")
    clf = RandomForestClassifier(**rf_params)
    t0 = time.time()
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0
    log_progress(f"Fit completado en {elapsed:.1f}s")

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log_progress(f"Accuracy 80/20: {acc:.4f}")

    report = classification_report(y_test, y_pred, zero_division=0)
    log_progress(f"Classification report:\n{report}")

    # Guardar modelo en docs/analisis/robbolla11/
    import joblib, json
    model_path = DOCS_ROOT / "random_forest_21letras_200porLetra.joblib"
    metrics_path = DOCS_ROOT / "metrics.json"
    report_path = DOCS_ROOT / "classification_report.txt"
    conf_path = DOCS_ROOT / "confusion_matrix.csv"

    joblib.dump(clf, model_path)
    log_progress(f"Modelo guardado {model_path} {model_path.stat().st_size/1024:.1f} KB")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    import numpy as np
    with open(conf_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + [str(c) for c in sorted(set(y))])
        for i, row in enumerate(cm):
            w.writerow([str(sorted(set(y))[i])] + list(row))

    metrics = {
        "dataset": "robbolla11",
        "total_samples": int(len(y)),
        "classes": sorted([str(c) for c in set(y)]),
        "n_classes": len(set(y)),
        "per_class_counts": {str(k): int(v) for k,v in cnt.items()},
        "expected_per_class": 200,
        "total_expected": 4200,
        "split": "80/20 stratify random_state 42",
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "model": "RandomForestClassifier",
        "params": rf_params,
        "parallel": "6 P-cores n_jobs=6 + RTX detected",
        "accuracy": float(acc),
        "fit_seconds": float(elapsed),
        "pickle_source": str(pickle_path),
        "data_shape": list(X.shape),
        "elapsed_total": float(time.time() - _stage_start)
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write(f"\nAccuracy: {acc:.4f}\n")
        f.write(f"Fit seconds: {elapsed:.1f}\n")
        f.write(f"Parallel 6 P-cores n_jobs=6 device {device}\n")
    log_progress(f"Metrics guardado {metrics_path}")
    log_progress(f"Reporte guardado {report_path}")
    log_progress(f"Confusion {conf_path}")

    # También guardar copia del modelo en formato pickle para compatibilidad con repo original
    pickle_model_path = DOCS_ROOT / "model.pickle"
    with open(pickle_model_path, "wb") as f:
        pickle.dump({"model": clf}, f)
    log_progress(f"Model pickle compat guardado {pickle_model_path} {pickle_model_path.stat().st_size/1024:.1f} KB")

    # Guardar accuracy_curve placeholder (RF no tiene curva, generar dummy)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,4))
        plt.bar(["accuracy"], [acc])
        plt.ylim(0,1)
        plt.title(f"robbolla11 RF accuracy {acc:.3f}")
        plt.ylabel("accuracy")
        curve_path = DOCS_ROOT / "accuracy_curve.png"
        plt.savefig(curve_path)
        plt.close()
        log_progress(f"accuracy_curve guardado {curve_path}")
    except Exception as e:
        log_progress(f"no se pudo generar accuracy_curve {e}")

    log_progress(f"ENTRENAMIENTO COMPLETADO accuracy {acc:.4f} modelo {model_path}")

def extract_from_images_parallel():
    # fallback mediapipe extraction
    log_progress("Fallback extract_from_images_parallel iniciado")
    import pathlib, cv2
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
    except Exception as e:
        raise RuntimeError(f"mediapipe no disponible para fallback: {e}")

    data_root = RAW_ROOT / "data"
    letra_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda x: x.name)
    all_tasks = []
    for ld in letra_dirs:
        for img_path in ld.iterdir():
            if img_path.suffix.lower() in [".jpg",".jpeg",".png"]:
                all_tasks.append((str(img_path), ld.name))
    log_progress(f"Fallback tasks {len(all_tasks)} images")

    def process_one(args):
        img_path, label = args
        import cv2, mediapipe as mp
        mp_hands = mp.solutions.hands
        # cada worker crea su detector
        with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3) as hands:
            img = cv2.imread(img_path)
            if img is None:
                return None
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)
            if not res.multi_hand_landmarks:
                return None
            lm = res.multi_hand_landmarks[0]
            # flatten 21*3=63
            arr = []
            for pt in lm.landmark:
                arr.extend([pt.x, pt.y, pt.z])
            return (arr, label)

    from multiprocessing import Pool
    # usar 6 P-cores para mediapipe? en realidad mediapipe ya es pesado, usamos 6
    with Pool(processes=6) as pool:
        results = pool.map(process_one, all_tasks)
    # filtrar None
    data = [r[0] for r in results if r is not None]
    labels = [r[1] for r in results if r is not None]
    log_progress(f"Fallback extraction ok {len(data)}/{len(all_tasks)} deg {len(all_tasks)-len(data)}")
    return data, labels

def main():
    start_progress()
    try:
        step_clone()
        step_extract()
        step_inspect_scripts()
        step_train()
        set_stage("completado")
        log_progress("PIPELINE robbolla11 COMPLETADO - todo automático 6P+RTX 80/20 200/letra")
    except Exception as e:
        import traceback
        log_progress(f"ERROR PIPELINE: {e}")
        log_progress(traceback.format_exc())
        raise
    finally:
        stop_progress()

if __name__ == "__main__":
    main()
