from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass

import numpy as np

from aplicacion.gestos import basic_translation
from core.base.mensajes import (
    LOADING_MODEL_TEXT,
    LOW_CONFIDENCE_TEXT,
    LOW_QUALITY_TEXT,
    NO_REAL_MODEL_TEXT,
    SHOW_HAND_TEXT,
    WARMING_SEQUENCE_TEXT,
)
from core.modelo.estabilizacion import TranslationStabilizer
from core.modelo.usar_modelo import RuntimeModel
from core.vision.puntos import extract_keypoints_for_size
from core.vision.reglas_gestos import count_fingers


@dataclass
class PredictionResult:
    translation: str
    source: str
    fingers: int | None
    confidence: float | None


class PredictionState:
    def __init__(self, args: argparse.Namespace):
        self.sequence: deque[np.ndarray] = deque(maxlen=args.frames)
        self.center_history: deque[tuple[float, float]] = deque(maxlen=18)
        self.confidence = 0.0
        self.stabilizer = TranslationStabilizer(
            window_size=args.stability_window,
            min_votes=args.stability_votes,
        )

    def clear(self) -> None:
        self.sequence.clear()
        self.center_history.clear()
        self.confidence = 0.0

    def update(
        self,
        args: argparse.Namespace,
        runtime_model: RuntimeModel,
        results,
        primary_hand,
        width: int,
        height: int,
        states: dict[str, bool] | None,
    ) -> PredictionResult:
        if primary_hand is None or states is None:
            self.clear()
            return PredictionResult(self.stabilizer.update(SHOW_HAND_TEXT), "none", None, None)

        self.center_history.append(
            (
                float(np.mean([point.x for point in primary_hand.landmark])),
                float(np.mean([point.y for point in primary_hand.landmark])),
            )
        )
        fingers = count_fingers(states)

        if runtime_model.ready:
            result = self._from_model(args, runtime_model, results)
        elif getattr(runtime_model, "status", "") in {"idle", "loading"}:
            result = PredictionResult(LOADING_MODEL_TEXT, "model_loading", fingers, None)
        elif args.allow_rule_demo:
            result = PredictionResult(
                basic_translation(states, primary_hand, results, width, height, self.center_history),
                "rule_demo",
                fingers,
                None,
            )
        else:
            result = PredictionResult(NO_REAL_MODEL_TEXT, "not_trained", fingers, None)

        result.translation = self.stabilizer.update(result.translation)
        result.fingers = fingers
        return result

    def _from_model(self, args: argparse.Namespace, runtime_model: RuntimeModel, results) -> PredictionResult:
        self.sequence.append(extract_keypoints_for_size(results, args.keypoint_size))
        source = f"tensorflow_{args.recognition_mode}_warming"
        translation = WARMING_SEQUENCE_TEXT
        confidence = None

        if len(self.sequence) == args.frames:
            needs_motion = args.recognition_mode in {"word", "sentence", "auto"}
            if needs_motion and not has_live_motion(self.sequence):
                return PredictionResult(LOW_CONFIDENCE_TEXT, f"tensorflow_{args.recognition_mode}_sin_movimiento", None, None)

            input_data = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
            prediction = runtime_model.predict(input_data)
            self.confidence = prediction.confidence
            confidence = self.confidence
            source = f"tensorflow_{args.recognition_mode}"

            if self.confidence < args.threshold or prediction.confidence_margin < args.min_confidence_margin:
                translation = LOW_CONFIDENCE_TEXT
                source = f"{source}_low_confidence"
                confidence = None
            elif not prediction.quality_usable:
                translation = LOW_QUALITY_TEXT
                source = f"{source}_low_quality"
                confidence = None
            else:
                translation = prediction.display_text

        return PredictionResult(translation, source, None, confidence)


def has_live_motion(sequence: deque[np.ndarray]) -> bool:
    frames = np.array(list(sequence)[-10:], dtype=np.float32)
    movement = float(np.mean(np.abs(np.diff(frames, axis=0)))) if len(frames) > 1 else 0.0
    return movement > 0.00001
