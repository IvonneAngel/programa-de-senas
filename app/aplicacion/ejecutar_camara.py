from __future__ import annotations

import argparse
import time

import cv2

from aplicacion.camara import read_frame
from aplicacion.constantes import WINDOW_HEIGHT, WINDOW_NAME, WINDOW_WIDTH
from aplicacion.gestos import choose_primary_hand, finger_states
from aplicacion.grabacion import ProcessRecorder
from aplicacion.prediccion import PredictionState
from core.interfaz.pantalla import build_practice_cards, compose_practice_view, draw_interface
from core.modelo.usar_modelo import RuntimeModel
from core.sistema.eventos import RuntimeEventLogger
from core.vision.recursos_mediapipe import configure_mediapipe_resources, prepare_mediapipe_import_path


def run_camera_loop(args: argparse.Namespace, cap, runtime_model: RuntimeModel, event_logger: RuntimeEventLogger) -> None:
    prepare_mediapipe_import_path()
    import mediapipe as mp

    configure_mediapipe_resources(mp)
    # Tasks Hand+Pose Landmarker (migración de Holistic)
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    hand_model = str((Path(__file__).resolve().parents[1] / "assets" / "models" / "hand_landmarker.task"))
    pose_model = str((Path(__file__).resolve().parents[1] / "assets" / "models" / "pose_landmarker_lite.task"))
    hand_landmarker = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=hand_model),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=0.30, min_tracking_confidence=0.30))
    pose_landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=pose_model),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1, min_pose_detection_confidence=0.30, min_tracking_confidence=0.30))
    prediction_state = PredictionState(args)
    practice_cards = build_practice_cards(args.practice_word) if args.practice_guide else []
    recorder = ProcessRecorder(args, event_logger)

    try:
        if args.smoke_test:
            frame = read_frame(cap)
            if frame is None:
                raise RuntimeError("La camara abrio pero no entrego imagen.")
            print("smoke-test-ok", frame.shape, "model_ready", runtime_model.ready)
            return

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        event_logger.start(_event_payload(args, runtime_model))
        started_at = time.monotonic()
        model_load_started = False

        while True:
            if args.max_runtime_seconds > 0 and time.monotonic() - started_at >= args.max_runtime_seconds:
                break

            frame = read_frame(cap, attempts=5)
            if frame is None:
                continue
            if args.mirror:
                frame = cv2.flip(frame, 1)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            hand_result = hand_landmarker.detect_for_video(mp_image, int(time.monotonic()*1000))
            pose_result = pose_landmarker.detect_for_video(mp_image, int(time.monotonic()*1000))
            results = type('obj', (), {'hand_landmarks': hand_result.hand_landmarks, 'pose_landmarks': pose_result.pose_landmarks})()
            primary_hand = choose_primary_hand(results)
            states = finger_states(primary_hand) if primary_hand is not None else None
            height, width = frame.shape[:2]
            prediction = prediction_state.update(args, runtime_model, results, primary_hand, width, height, states)

            event_logger.log_translation(
                prediction.translation,
                fingers=prediction.fingers,
                source=prediction.source,
                confidence=prediction.confidence,
            )
            draw_interface(frame, results, prediction.translation, prediction.fingers, runtime_model.ready, allow_rule_demo=args.allow_rule_demo)
            display_frame = _practice_frame(args, frame, practice_cards, started_at)
            recorder.update(display_frame)
            cv2.imshow(WINDOW_NAME, display_frame)
            if not model_load_started:
                start_loader = getattr(runtime_model, "start", None)
                if callable(start_loader):
                    start_loader()
                model_load_started = True

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        recorder.close()
        event_logger.finish()
        hand_landmarker.close()
        pose_landmarker.close()
        cap.release()
        cv2.destroyAllWindows()


def _practice_frame(args: argparse.Namespace, frame, practice_cards: list, started_at: float):
    if not args.practice_guide:
        return frame
    return compose_practice_view(
        frame,
        practice_cards,
        elapsed_seconds=time.monotonic() - started_at,
        rotate_seconds=args.practice_rotate_seconds,
    )


def _event_payload(args: argparse.Namespace, runtime_model: RuntimeModel) -> dict:
    return {
        "camera": args.camera,
        "model_ready": runtime_model.ready,
        "labels": runtime_model.labels,
        "recognition_mode": args.recognition_mode,
        "runtime_model": runtime_model.as_event_payload(),
        "record_process": args.record_process,
        "record_seconds": args.record_seconds,
        "record_fps": args.record_fps,
        "record_scale": args.record_scale,
        "stability_window": args.stability_window,
        "stability_votes": args.stability_votes,
        "max_runtime_seconds": args.max_runtime_seconds,
        "practice_guide": args.practice_guide,
        "practice_word": args.practice_word,
        "allow_rule_demo": args.allow_rule_demo,
    }
