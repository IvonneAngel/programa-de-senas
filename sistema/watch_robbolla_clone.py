
import pathlib, time, psutil, subprocess, sys
from pathlib import Path

clone = Path("C:/Users/riemann/Desktop/robbolla11")
progress = Path("C:/Users/riemann/Desktop/entrenamiento_robbolla_progreso.txt")
log = Path("C:/Users/riemann/Desktop/robbolla_pipeline_launch.log")

while True:
    # check if clone now has more than .git
    items = list(clone.iterdir()) if clone.exists() else []
    has_data = any((clone/"data").exists() or (clone/"Data").exists() for _ in [1])
    # check git alive
    git_alive = any('git' in (p.info['name'] or '').lower() for p in psutil.process_iter(['name']) if p.info['name'])
    # size
    try:
        size_mb = sum(f.stat().st_size for f in clone.rglob("*") if f.is_file())/1024/1024
    except: size_mb=0
    print(f"[watcher {time.strftime('%H:%M:%S')}] git_alive={git_alive} size={size_mb:.1f}MB items={[x.name for x in items][:5]} has_data={has_data}")
    if not git_alive and len(items) > 1:
        print("[watcher] clone parece terminado, verificando")
        # list root
        for p in clone.iterdir():
            print(f"  {p.name} {'DIR' if p.is_dir() else p.stat().st_size}")
        break
    # also detect clone finished by checking git status returns commits
    try:
        r = subprocess.run(["git","log","--oneline","-1"], cwd=str(clone), capture_output=True, text=True, timeout=5)
        if r.returncode==0 and r.stdout.strip():
            print(f"[watcher] git log found: {r.stdout[:200]}")
            # check that we have data dir
            if len(items)>1:
                break
    except: pass
    time.sleep(10)

print("[watcher] done monitoring clone")
