from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import TCPServer
from types import SimpleNamespace
from time import strftime


CODIGO = Path(__file__).resolve().parents[1]
PROYECTO = CODIGO.parent
DIST = CODIGO / "interfaz-ui" / "dist"
TRADUCTOR = CODIGO / "aplicacion" / "traductor_de_senas.py"

if str(CODIGO) not in sys.path:
    sys.path.insert(0, str(CODIGO))

MENSAJES_INTERNOS = {
    "cargando modelo",
    "lenguaje de señas",
    "lenguaje de senas",
    "leyendo seña",
    "leyendo sena",
    "modelo lsm no entrenado",
    "modelo no seguro",
    "muestra tu mano",
    "palabra aun no validada",
    "palabra aún no validada",
}

MODELO_LOCK = threading.RLock()
DETECTOR_LOCK = threading.RLock()
MODELO = None
ESTADO_PREDICCION = None
DETECTOR = None
ARGS_MODELO = SimpleNamespace(
    frames=30,
    keypoint_size=226,
    threshold=0.88,
    min_confidence_margin=0.20,
    recognition_mode="alphabet",
    stability_window=7,
    stability_votes=4,
    allow_rule_demo=False,
)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIST), **kwargs)

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path in {"/api/estado", "/api/health"}:
            self._json({"ok": True, "status": "listo"})
            return
        if self.path == "/" or self.path.startswith("/assets/") or self.path.startswith("/mediapipe/"):
            super().do_GET()
            return
        index = DIST / "index.html"
        if index.exists():
            self.path = "/index.html"
            super().do_GET()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/prediccion":
            try:
                self._json(predecir_desde_ui(self._read_json()))
            except Exception as error:
                self._json({"ok": False, "error": str(error)}, 500)
            return
        if self.path == "/api/prediccion-frame":
            try:
                self._json(predecir_frame_desde_ui(self._read_json()))
            except Exception as error:
                self._json({"ok": False, "error": str(error)}, 500)
            return
        if self.path != "/api/abrir-traductor":
            self.send_error(404)
            return
        if not TRADUCTOR.exists():
            self._json({"ok": False, "error": "traductor no encontrado"}, 404)
            return
        command = [
            sys.executable,
            str(TRADUCTOR),
            "--camera",
            "0",
            "--recognition-mode",
            "auto",
            "--stability-window",
            "7",
            "--stability-votes",
            "4",
        ]
        kwargs = {"cwd": str(CODIGO)}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
        subprocess.Popen(command, **kwargs)
        self._json({"ok": True})

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            return


class ServidorRapido(ThreadingHTTPServer):
    def server_bind(self) -> None:
        TCPServer.server_bind(self)
        self.server_name = str(self.server_address[0])
        self.server_port = int(self.server_address[1])


def predecir_desde_ui(data: dict) -> dict:
    width = int(data.get("width") or 1)
    height = int(data.get("height") or 1)
    results = crear_resultados(data.get("hands"))
    return predecir_con_resultados(results, width, height)


def predecir_con_resultados(results, width: int, height: int) -> dict:
    primary_hand = elegir_mano_principal(results)

    if primary_hand is None:
        return {
            "ok": True,
            "translation": "",
            "source": "sin mano",
            "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        }

    states = calcular_dedos(primary_hand)
    with MODELO_LOCK:
        runtime_model, prediction_state = obtener_runtime()
        prediction = prediction_state.update(
            ARGS_MODELO,
            runtime_model,
            results,
            primary_hand,
            width,
            height,
            states,
        )
    return {
        "ok": True,
        "translation": limpiar_traduccion(prediction.translation),
        "raw_translation": prediction.translation,
        "source": prediction.source,
        "confidence": prediction.confidence,
        "fingers": prediction.fingers,
        "model_ready": runtime_model.ready,
        "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
    }


def predecir_frame_desde_ui(data: dict) -> dict:
    frame = decodificar_frame(data.get("image"))
    if frame is None:
        return {
            "ok": True,
            "translation": "",
            "source": "frame vacio",
            "hands": [],
            "timestamp": strftime("%Y-%m-%d %H:%M:%S"),
        }

    detector = obtener_detector()
    import cv2

    height, width = frame.shape[:2]
    with DETECTOR_LOCK:
        detection_results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    results = crear_resultados_desde_detector(detection_results)
    hands = manos_para_ui(results)
    base = predecir_con_resultados(results, width, height)
    base["hands"] = hands
    base["frame_width"] = width
    base["frame_height"] = height
    return base


def decodificar_frame(value: object):
    if not isinstance(value, str) or not value:
        return None
    if "," in value:
        value = value.split(",", 1)[1]
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return None

    import cv2
    import numpy as np

    data = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return frame


def obtener_detector():
    global DETECTOR
    with DETECTOR_LOCK:
        if DETECTOR is None:
            from core.vision.recursos_mediapipe import configure_mediapipe_resources, prepare_mediapipe_import_path

            prepare_mediapipe_import_path()
            import mediapipe as mp

            configure_mediapipe_resources(mp)
            DETECTOR = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.35,
                min_tracking_confidence=0.35,
            )
        return DETECTOR


def calentar_servidor() -> None:
    try:
        obtener_detector()
        obtener_runtime()
    except Exception as error:
        print(f"calentamiento_servidor_error={error}", flush=True)


def manos_para_ui(results) -> list[dict[str, object]]:
    hands: list[dict[str, object]] = []
    for name, landmarks in (
        ("left", getattr(results, "left_hand_landmarks", None)),
        ("right", getattr(results, "right_hand_landmarks", None)),
    ):
        if landmarks is None:
            continue
        hands.append(
            {
                "handedness": name,
                "landmarks": [
                    {"x": float(point.x), "y": float(point.y), "z": float(point.z)}
                    for point in landmarks.landmark
                ],
            }
        )
    return hands


def crear_resultados_desde_detector(results) -> SimpleNamespace:
    landmarks = list(getattr(results, "multi_hand_landmarks", None) or [])
    right = landmarks[0] if landmarks else None
    left = landmarks[1] if len(landmarks) > 1 else None
    return SimpleNamespace(
        left_hand_landmarks=left,
        right_hand_landmarks=right,
        pose_landmarks=None,
        face_landmarks=None,
    )


def obtener_runtime():
    global MODELO, ESTADO_PREDICCION
    with MODELO_LOCK:
        if MODELO is None:
            from aplicacion.prediccion import PredictionState
            from core.modelo.usar_modelo import load_runtime_model

            MODELO = load_runtime_model(
                CODIGO,
                ARGS_MODELO.recognition_mode,
                frames=ARGS_MODELO.frames,
                keypoints=ARGS_MODELO.keypoint_size,
                threshold=ARGS_MODELO.threshold,
            )
            ESTADO_PREDICCION = PredictionState(ARGS_MODELO)
        return MODELO, ESTADO_PREDICCION


def elegir_mano_principal(results):
    return results.right_hand_landmarks or results.left_hand_landmarks


def calcular_dedos(primary_hand) -> dict[str, bool]:
    from aplicacion.gestos import finger_states

    return finger_states(primary_hand)


def crear_resultados(hands) -> SimpleNamespace:
    left = None
    right = None
    if not isinstance(hands, list):
        hands = []

    for index, hand in enumerate(hands):
        if not isinstance(hand, dict):
            continue
        landmarks = crear_landmarks(hand.get("landmarks"))
        if landmarks is None:
            continue
        handedness = str(hand.get("handedness") or "").lower()
        if handedness == "left":
            left = landmarks
        elif handedness == "right":
            right = landmarks
        elif index == 0:
            right = landmarks
        else:
            left = landmarks

    return SimpleNamespace(
        left_hand_landmarks=left,
        right_hand_landmarks=right,
        pose_landmarks=None,
        face_landmarks=None,
    )


def crear_landmarks(points) -> SimpleNamespace | None:
    if not isinstance(points, list) or len(points) < 21:
        return None

    landmarks = []
    for point in points[:21]:
        if not isinstance(point, dict):
            return None
        landmarks.append(
            SimpleNamespace(
                x=float(point.get("x") or 0.0),
                y=float(point.get("y") or 0.0),
                z=float(point.get("z") or 0.0),
            )
        )
    return SimpleNamespace(landmark=landmarks)


def limpiar_traduccion(value: object) -> str:
    text = str(value or "").replace("_", " ").strip()
    if not text:
        return ""
    return "" if text.lower() in MENSAJES_INTERNOS else text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not DIST.exists():
        raise SystemExit("falta construir la interfaz ui")

    threading.Thread(target=calentar_servidor, daemon=True).start()

    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        webbrowser.open(url)
    server = ServidorRapido(("127.0.0.1", args.port), Handler)
    print(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
