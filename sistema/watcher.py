
import time, json, pathlib, sys, shutil

log_path = pathlib.Path("training_output.log")
progreso = pathlib.Path("C:/Users/riemann/Desktop/entrenamiento_progreso.txt")
fase2 = pathlib.Path("docs/analisis/fase2-entrenamiento")
fase3 = pathlib.Path("docs/analisis/fase3-evaluacion")
fase4 = pathlib.Path("docs/analisis/fase4-graficas")

# Monitor log file for new epochs and update progreso every 5
seen_epochs = set()
# Load already seen from progreso
if progreso.exists():
    txt = progreso.read_text(encoding="utf-8")
    for line in txt.splitlines():
        if line.startswith("Epoca"):
            try:
                ep = int(line.split()[1].split("/")[0])
                seen_epochs.add(ep)
            except: pass

print(f"[watcher] starting, seen {seen_epochs}")

# Also track train_proc pid if provided via env
import os, psutil

while True:
    time.sleep(5)
    if not log_path.exists():
        continue
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except: continue
    new_epochs = []
    for line in lines:
        line=line.strip()
        if not line.startswith("{"): continue
        try:
            obj=json.loads(line)
        except: continue
        if "epoch" in obj and "validation" in obj:
            ep=obj["epoch"]
            if ep not in seen_epochs:
                seen_epochs.add(ep)
                # write progreso every 5 or 1
                if ep %5==0 or ep==1:
                    msg = f"Epoca {ep:02d}/40 - train_loss {obj['train']['loss']:.4f} - val_acc {obj['validation']['accuracy']:.4f} - val_macro_f1 {obj['validation']['macro_f1']:.4f} - elapsed {obj.get('elapsed_seconds',0):.1f}s"
                    print(f"[watcher progreso] {msg}")
                    with open(progreso,"a",encoding="utf-8") as pf:
                        pf.write(msg+"\n")
                new_epochs.append(ep)
        if "done" in obj:
            print(f"[watcher] training done {obj}")
            # Do post-processing
            time.sleep(2)
            try:
                if (fase2/"best.pt").exists():
                    fase3.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fase2/"best.pt", fase3/"best.pt")
                    print(f"copiado best.pt a {fase3}")
                if (fase2/"metrics.json").exists():
                    fase3.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fase2/"metrics.json", fase3/"metrics.json")
                    print(f"copiado metrics.json a {fase3}")
                    # Generate accuracy_curve.png
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
                    with open(progreso,"a",encoding="utf-8") as pf:
                        pf.write(f"\nEntrenamiento completado. Best val macro F1: {best_f1:.4f} en epoca {best_epoch}\n")
                        pf.write(f"Curva guardada en {out_png}\n")
                        pf.write(f"best.pt guardado en {fase2}/best.pt y {fase3}/best.pt\n")
                        pf.write(f"metrics guardado en {fase2}/metrics.json y {fase3}/metrics.json\n")
            except Exception as e:
                print(f"post-process error {e}")
                import traceback; traceback.print_exc()
            sys.exit(0)
    # Check if train process still alive
    alive=False
    for proc in psutil.process_iter(['cmdline']):
        try:
            cmd=" ".join(proc.info['cmdline'] or [])
            if "train_classifier" in cmd and "mendeley" in cmd:
                alive=True
                break
        except: pass
    if not alive:
        # No training process, but maybe done file not yet written? Wait a bit
        # Check if metrics.json has 40 epochs
        if (fase2/"metrics.json").exists():
            try:
                data=json.loads((fase2/"metrics.json").read_text(encoding="utf-8"))
                if len(data.get("history",[])) >= 40 or "best_validation_macro_f1" in data:
                    # Assume finished
                    print("[watcher] no training proc but metrics exists, doing post")
                    # trigger post
                    pass
            except: pass
        # Continue looping a bit more
        pass
