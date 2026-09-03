from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_LOGGING_MIN_LOG_LEVEL", "3")
warnings.filterwarnings(
    "ignore",
    message=r".*SymbolDatabase\.GetPrototype\(\) is deprecated.*",
    category=UserWarning,
)
APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from aplicacion.argumentos import parse_args
from aplicacion.camara import open_camera
from aplicacion.demo import run_demo_loop
from aplicacion.ejecutar_camara import run_camera_loop
from aplicacion.gestos import resolve_runtime_mode
from core.base.rutas import ruta_sesiones
from core.modelo.usar_modelo import load_runtime_model, start_runtime_model_load
from core.sistema.eventos import RuntimeEventLogger


def main() -> None:
    args = parse_args()
    event_logger = RuntimeEventLogger(
        ruta_sesiones(),
        enabled=not args.no_log_events and not args.smoke_test,
    )

    cap = None if args.demo_mode else open_camera(args.camera)
    runtime_mode = resolve_runtime_mode(
        bool(cap is not None and cap.isOpened()),
        demo_mode=args.demo_mode,
        camera_optional=args.camera_optional,
    )

    sync_model = args.smoke_test or runtime_mode == "demo"
    load_model = load_runtime_model if sync_model else start_runtime_model_load
    runtime_model = load_model(
        APP_DIR,
        args.recognition_mode,
        frames=args.frames,
        keypoints=args.keypoint_size,
        threshold=args.threshold,
        **({} if sync_model else {"start_immediately": False}),
    )

    if runtime_mode == "demo":
        if cap is not None:
            cap.release()
        run_demo_loop(args, runtime_model.ready, runtime_model.labels, event_logger)
        return

    run_camera_loop(args, cap, runtime_model, event_logger)


if __name__ == "__main__":
    main()
