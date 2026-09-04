
import subprocess, sys, os, json, time, pathlib, shutil

manifest = "dataset/manifests/mendeley_6rj76z6y3n_manifest.csv"
cache_root = "dataset/processed"
task = "successor_positions126"
out = "docs/analisis/fase2-entrenamiento"
device = "cuda"
batch_size = 8
epochs = 40
num_workers = 6

env = os.environ.copy()
env["PYTHONPATH"] = "entrenador/core-pt" + os.pathsep + env.get("PYTHONPATH","")

cmd = ["C:/Users/riemann/.venv/Scripts/python.exe", "-u", "entrenador/core-pt/lsm/training/train_classifier.py",
       "--manifest", manifest, "--cache-root", cache_root, "--task", task,
       "--out", out, "--device", device, "--batch-size", str(batch_size),
       "--epochs", str(epochs), "--num-workers", str(num_workers)]

progress_path = pathlib.Path("C:/Users/riemann/Desktop/entrenamiento_progreso.txt")
log_path = pathlib.Path("training_output.log")
train_log = pathlib.Path("docs/analisis/fase2-entrenamiento/train.log")

print(f"[launcher] CMD {' '.join(cmd)}")
# Open logs
log_file = open(log_path, "w", encoding="utf-8")
train_log.parent.mkdir(parents=True, exist_ok=True)
train_log_file = open(train_log, "w", encoding="utf-8")

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)

for line in proc.stdout:
    print(line, end="")
    log_file.write(line)
    train_log_file.write(line)
    log_file.flush()
    train_log_file.flush()
    try:
        obj=json.loads(line.strip())
        if "epoch" in obj and "validation" in obj:
            ep=obj["epoch"]
            if ep %5==0 or ep==1:
                msg = f"Epoca {ep:02d}/{epochs} - train_loss {obj['train']['loss']:.4f} - val_acc {obj['validation']['accuracy']:.4f} - val_macro_f1 {obj['validation']['macro_f1']:.4f} - elapsed {obj.get('elapsed_seconds',0):.1f}s"
                print(f"[PROGRESO] {msg}")
                with open(progress_path,"a",encoding="utf-8") as pf:
                    pf.write(msg+"\n")
    except: pass

proc.wait()
log_file.close()
train_log_file.close()
print(f"[launcher] training finished code {proc.returncode}")

# Post-process
fase2 = pathlib.Path("docs/analisis/fase2-entrenamiento")
fase3 = pathlib.Path("docs/analisis/fase3-evaluacion")
fase4 = pathlib.Path("docs/analisis/fase4-graficas")
if (fase2/"best.pt").exists():
    fase3.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fase2/"best.pt", fase3/"best.pt")
    print(f"copiado best.pt")
if (fase2/"metrics.json").exists():
    fase3.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fase2/"metrics.json", fase3/"metrics.json")
    print(f"copiado metrics.json")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        data=json.loads((fase2/"metrics.json").read_text(encoding="utf-8"))
        hist=data.get("history",[])
        epochs_vals=[h["epoch"] for h in hist]
        train_loss=[h["train"].get("loss",0) for h in hist]
        val_acc=[h["validation"].get("accuracy",0) for h in hist]
        val_f1=[h["validation"].get("macro_f1",0) for h in hist]
        fig, ax1=plt.subplots(figsize=(10,6))
        ax1.set_xlabel("Epoca")
        ax1.set_ylabel("Loss / Accuracy")
        ax1.plot(epochs_vals, train_loss, label="train_loss", color="red", linestyle="--")
        ax1.plot(epochs_vals, val_acc, label="val_accuracy", color="blue")
        ax1.plot(epochs_vals, val_f1, label="val_macro_f1", color="green")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        best_f1=data.get("best_validation_macro_f1",0)
        best_epoch=max(hist, key=lambda x: x["validation"].get("macro_f1",0))["epoch"] if hist else 1
        ax1.set_title(f"mendeley_6rj76z6y3n successor_positions126 - {len(epochs_vals)} epocas - best val F1 {best_f1:.4f}")
        ax1.axvline(best_epoch, color="gray", linestyle=":", label=f"best {best_epoch}")
        plt.tight_layout()
        fase4.mkdir(parents=True, exist_ok=True)
        out_png=fase4/"accuracy_curve.png"
        plt.savefig(out_png, dpi=150)
        print(f"generada curva en {out_png}")
        with open(progress_path,"a",encoding="utf-8") as pf:
            pf.write(f"\nEntrenamiento completado. Best val macro F1: {best_f1:.4f} en epoca {best_epoch}\n")
            pf.write(f"Curva guardada en {out_png}\n")
            pf.write(f"best.pt guardado en {fase2}/best.pt y {fase3}/best.pt\n")
            pf.write(f"metrics guardado en {fase2}/metrics.json y {fase3}/metrics.json\n")
    except Exception as e:
        print(f"error grafica {e}")
        import traceback; traceback.print_exc()
