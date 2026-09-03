from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Literal

import numpy as np

from core.base.configuracion import (
    CONFIDENCE_THRESHOLD,
DEFAULT_LABELS_PATH,
DEFAULT_MODEL_PATH,
FRAME_COUNT,
KEYPOINT_SIZE,
)
from core.base.mensajes import display_phrase
from core.modelo.calidad_modelo import ModelQualityProfile, load_model_quality_profile
from core.modelo.validar_modelo import ModelValidationError, load_labels
from core.datos.manifiesto_objetivos import iter_alphabet_specs, iter_sentence_specs


RecognitionMode = Literal["auto", "word", "alphabet", "sentence"]


IGNORED_TRANSLATION_LABELS = {
    "lenguaje_de_senas",
    "lenguaje_de_señas",
    "sign_language",
    "senas",
    "señas",
}


@dataclass(frozen=True, slots=True)
class RuntimeModelSpec:
    mode: RecognitionMode
    title: str
    model_path: Path
    labels_path: Path
    min_labels: int
    expected_labels: tuple[str, ...] | None
    display_labels: dict[str, str]
    threshold: float = CONFIDENCE_THRESHOLD
    quality_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RuntimePrediction:
    label: str
    display_text: str
    confidence: float
    mode: RecognitionMode
    second_label: str = ""
    second_confidence: float = 0.0
    confidence_margin: float = 0.0
    quality_status: str = "unknown"
    quality_f1: float | None = None
    quality_usable: bool = True


@dataclass(slots=True)
class RuntimeModel:
    spec: RuntimeModelSpec
    status: str
    labels: list[str]
    model: object | None = None
    quality_profile: ModelQualityProfile | None = None
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.status == "ready" and self.model is not None

    def display_for_label(self, label: str) -> str:
        return self.spec.display_labels.get(label, display_phrase(label))

    def predict(self, input_data: np.ndarray) -> RuntimePrediction:
        if not self.ready:
            raise RuntimeError(self.error or f"Modelo {self.spec.mode} no listo.")

        probabilities = self.model.predict(input_data, verbose=0)[0]
        ranked_indices = np.argsort(probabilities)[::-1]
        index = _first_useful_index(ranked_indices, self.labels)
        confidence = float(probabilities[index])
        label = self.labels[index]
        second_label = ""
        second_confidence = 0.0
        second_index = _next_useful_index(ranked_indices, self.labels, skip_index=index)
        if second_index is not None:
            second_label = self.labels[second_index]
            second_confidence = float(probabilities[second_index])
        confidence_margin = confidence - second_confidence
        quality = (
            self.quality_profile.for_label(label)
            if self.quality_profile is not None
            else load_model_quality_profile(None).for_label(label)
        )
        return RuntimePrediction(
            label=label,
            display_text=self.display_for_label(label),
            confidence=confidence,
            mode=self.spec.mode,
            second_label=second_label,
            second_confidence=second_confidence,
            confidence_margin=confidence_margin,
            quality_status=quality.status,
            quality_f1=quality.f1_score,
            quality_usable=quality.usable,
        )

    def as_event_payload(self) -> dict[str, object]:
        return {
            "mode": self.spec.mode,
            "title": self.spec.title,
            "status": self.status,
            "model_path": str(self.spec.model_path),
            "labels_path": str(self.spec.labels_path),
            "label_count": len(self.labels),
            "labels": self.labels,
            "quality_profile_path": str(self.quality_profile.path) if self.quality_profile and self.quality_profile.path else "",
            "error": self.error,
        }


@dataclass(slots=True)
class AutoRuntimeModel:
    alphabet: RuntimeModel
    word: RuntimeModel
    status: str = "ready"
    error: str = ""

    @property
    def ready(self) -> bool:
        return self.alphabet.ready or self.word.ready

    @property
    def labels(self) -> list[str]:
        labels: list[str] = []
        if self.alphabet.labels:
            labels.extend([f"letra:{label}" for label in self.alphabet.labels])
        if self.word.labels:
            labels.extend([f"palabra:{label}" for label in self.word.labels])
        return labels

    def predict(self, input_data: np.ndarray) -> RuntimePrediction:
        candidates: list[RuntimePrediction] = []
        if self.alphabet.ready:
            candidates.append(self.alphabet.predict(input_data))
        if self.word.ready:
            candidates.append(self.word.predict(input_data))
        if not candidates:
            raise RuntimeError(self.error or "No hay modelos listos para modo automatico.")

        scored = sorted(candidates, key=_auto_score, reverse=True)
        best = scored[0]
        if best.mode == "word" and best.label in IGNORED_TRANSLATION_LABELS and len(scored) > 1:
            best = scored[1]
        return RuntimePrediction(
            label=best.label,
            display_text=best.display_text,
            confidence=best.confidence,
            mode="auto",
            second_label=best.second_label,
            second_confidence=best.second_confidence,
            confidence_margin=best.confidence_margin,
            quality_status=best.quality_status,
            quality_f1=best.quality_f1,
            quality_usable=best.quality_usable,
        )

    def as_event_payload(self) -> dict[str, object]:
        return {
            "mode": "auto",
            "title": "Modelo automatico letra/palabra",
            "status": self.status,
            "label_count": len(self.labels),
            "alphabet": self.alphabet.as_event_payload(),
            "word": self.word.as_event_payload(),
            "error": self.error,
        }


def _auto_score(prediction: RuntimePrediction) -> float:
    score = prediction.confidence + max(0.0, prediction.confidence_margin) * 0.35
    if prediction.mode == "alphabet":
        score += 0.08
    if prediction.mode == "word" and prediction.quality_status == "weak":
        score -= 0.12
    if prediction.label in IGNORED_TRANSLATION_LABELS:
        score -= 1.0
    return score


def _shape_to_list(shape: object) -> list[object]:
    if shape is None:
        return []
    if isinstance(shape, list):
        if shape and isinstance(shape[0], (list, tuple)):
            return list(shape[0])
        return list(shape)
    return list(shape)


def _first_useful_index(ranked_indices: np.ndarray, labels: list[str]) -> int:
    for raw_index in ranked_indices:
        index = int(raw_index)
        if labels[index] not in IGNORED_TRANSLATION_LABELS:
            return index
    return int(ranked_indices[0])


def _next_useful_index(ranked_indices: np.ndarray, labels: list[str], *, skip_index: int) -> int | None:
    for raw_index in ranked_indices:
        index = int(raw_index)
        if index == skip_index:
            continue
        if labels[index] not in IGNORED_TRANSLATION_LABELS:
            return index
    return None


class TfliteModel:
    def __init__(self, model_path: Path):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="senas_modelo_")
        safe_model_path = Path(self._temp_dir.name) / model_path.name
        shutil.copy2(model_path, safe_model_path)
        self.interpreter = _create_tflite_interpreter(safe_model_path)
        self.interpreter.allocate_tensors()
        self.input_info = self.interpreter.get_input_details()[0]
        self.output_info = self.interpreter.get_output_details()[0]
        self.input_shape = tuple(self.input_info["shape"])
        self.output_shape = tuple(self.output_info["shape"])

    def predict(self, input_data: np.ndarray, verbose: int = 0) -> np.ndarray:
        expected_dtype = self.input_info["dtype"]
        self.interpreter.set_tensor(self.input_info["index"], input_data.astype(expected_dtype, copy=False))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_info["index"])


def _create_tflite_interpreter(model_path: Path):
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=str(model_path), num_threads=2)
    except ModuleNotFoundError:
        import tensorflow as tf

        return tf.lite.Interpreter(model_path=str(model_path), num_threads=2)


def _alphabet_display_labels() -> dict[str, str]:
    return {spec.data_label: spec.display_label for spec in iter_alphabet_specs()}


def _sentence_display_labels() -> dict[str, str]:
    return {spec.data_label: spec.display_label for spec in iter_sentence_specs()}


def _resolve_runtime_path(root: Path, value: str | Path | None, default_relative: str | Path) -> Path:
    path = Path(value) if value is not None else Path(default_relative)
    return path if path.is_absolute() else root / path


def _preferred_model_path(root: Path, model_path: str | Path | None, default_relative: str | Path) -> Path:
    if model_path is not None:
        return _resolve_runtime_path(root, model_path, default_relative)
    default_path = _resolve_runtime_path(root, None, default_relative)
    tflite_path = default_path.with_suffix(".tflite")
    if tflite_path.exists():
        return tflite_path
    return default_path


def runtime_model_spec(
    app_dir: str | Path,
    mode: RecognitionMode,
    *,
    model_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> RuntimeModelSpec:
    root = Path(app_dir)

    if mode == "auto":
        return RuntimeModelSpec(
            mode="auto",
            title="Modelo automatico letra/palabra",
            model_path=root / "modelos" / "auto_runtime",
            labels_path=root / "modelos" / "auto_labels.json",
            min_labels=2,
            expected_labels=None,
            display_labels={},
            threshold=threshold,
        )

    if mode == "word":
        return RuntimeModelSpec(
            mode="word",
            title="Modelo de palabras",
            model_path=_preferred_model_path(root, model_path, DEFAULT_MODEL_PATH),
            labels_path=_resolve_runtime_path(root, labels_path, DEFAULT_LABELS_PATH),
            min_labels=2,
            expected_labels=None,
            display_labels={},
            threshold=threshold,
            quality_path=root / "modelos" / "model_quality.json",
        )

    if mode == "alphabet":
        expected = tuple(spec.data_label for spec in iter_alphabet_specs())
        return RuntimeModelSpec(
            mode="alphabet",
            title="Modelo de alfabeto LSM",
            model_path=_preferred_model_path(root, model_path, Path("modelos") / "lsm_alphabet_model.keras"),
            labels_path=_resolve_runtime_path(root, labels_path, Path("modelos") / "lsm_alphabet_labels.json"),
            min_labels=2,
            expected_labels=None,
            display_labels=_alphabet_display_labels(),
            threshold=threshold,
        )

    if mode == "sentence":
        expected = tuple(spec.data_label for spec in iter_sentence_specs())
        return RuntimeModelSpec(
            mode="sentence",
            title="Modelo de oraciones LSM",
            model_path=_preferred_model_path(root, model_path, Path("modelos") / "lsm_sentence_model.keras"),
            labels_path=_resolve_runtime_path(root, labels_path, Path("modelos") / "lsm_sentence_labels.json"),
            min_labels=len(expected),
            expected_labels=expected,
            display_labels=_sentence_display_labels(),
            threshold=threshold,
        )

    raise ValueError(f"Modo de reconocimiento no soportado: {mode!r}")


def validate_runtime_labels(spec: RuntimeModelSpec) -> list[str]:
    labels = load_labels(spec.labels_path, min_labels=spec.min_labels)
    if spec.expected_labels is not None and labels != list(spec.expected_labels):
        missing = [label for label in spec.expected_labels if label not in labels]
        unexpected = [label for label in labels if label not in spec.expected_labels]
        raise ModelValidationError(
            "Etiquetas incompatibles para "
            f"{spec.mode}: faltantes={missing[:10]}, inesperadas={unexpected[:10]}"
        )
    return labels


def validate_runtime_model_shape(
    model: object,
    labels: list[str],
    *,
    frames: int = FRAME_COUNT,
    keypoints: int = KEYPOINT_SIZE,
) -> None:
    input_shape = _shape_to_list(model.input_shape)
    output_shape = _shape_to_list(model.output_shape)
    if len(input_shape) != 3 or input_shape[1] != frames or input_shape[2] != keypoints:
        raise ModelValidationError(
            f"Entrada incompatible: esperaba (None, {frames}, {keypoints}), encontre {input_shape}"
        )
    if not output_shape or output_shape[-1] != len(labels):
        raise ModelValidationError(
            f"Salida incompatible: output_shape={output_shape}, labels={len(labels)}"
        )


def load_runtime_model(
    app_dir: str | Path,
    mode: RecognitionMode = "word",
    *,
    model_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    frames: int = FRAME_COUNT,
    keypoints: int = KEYPOINT_SIZE,
    threshold: float = CONFIDENCE_THRESHOLD,
) -> RuntimeModel | AutoRuntimeModel:
    if mode == "auto":
        alphabet = load_runtime_model(
            app_dir,
            "alphabet",
            frames=frames,
            keypoints=keypoints,
            threshold=threshold,
        )
        word = load_runtime_model(
            app_dir,
            "word",
            frames=frames,
            keypoints=keypoints,
            threshold=threshold,
        )
        status = "ready" if alphabet.ready or word.ready else "invalid"
        errors = "; ".join(error for error in [alphabet.error, word.error] if error)
        return AutoRuntimeModel(alphabet=alphabet, word=word, status=status, error=errors)

    spec = runtime_model_spec(
        app_dir,
        mode,
        model_path=model_path,
        labels_path=labels_path,
        threshold=threshold,
    )

    if not spec.model_path.exists() or not spec.labels_path.exists():
        return RuntimeModel(
            spec=spec,
            status="missing",
            labels=[],
            error=f"No existe modelo o etiquetas para {spec.mode}.",
        )

    try:
        labels = validate_runtime_labels(spec)

        if spec.model_path.suffix.lower() == ".tflite":
            model = TfliteModel(spec.model_path)
        else:
            import tensorflow as tf

            model = tf.keras.models.load_model(spec.model_path, compile=False)
        validate_runtime_model_shape(model, labels, frames=frames, keypoints=keypoints)
        quality_profile = load_model_quality_profile(spec.quality_path)
        return RuntimeModel(
            spec=spec,
            status="ready",
            labels=labels,
            model=model,
            quality_profile=quality_profile,
        )
    except Exception as exc:
        return RuntimeModel(spec=spec, status="invalid", labels=[], error=str(exc))


class AsyncRuntimeModel:
    def __init__(
        self,
        app_dir: str | Path,
        mode: RecognitionMode = "word",
        *,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        frames: int = FRAME_COUNT,
        keypoints: int = KEYPOINT_SIZE,
        threshold: float = CONFIDENCE_THRESHOLD,
        start_immediately: bool = True,
    ):
        self._lock = threading.RLock()
        self._model: RuntimeModel | AutoRuntimeModel | None = None
        self._error = ""
        self._started = False
        self._args = (app_dir, mode)
        self._kwargs = {
            "model_path": model_path,
            "labels_path": labels_path,
            "frames": frames,
            "keypoints": keypoints,
            "threshold": threshold,
        }
        self._thread = threading.Thread(target=self._load, name="senas-model-loader", daemon=True)
        if start_immediately:
            self.start()

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread.start()

    def _load(self) -> None:
        try:
            model = load_runtime_model(*self._args, **self._kwargs)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
            return
        with self._lock:
            self._model = model
            self._error = getattr(model, "error", "")

    @property
    def status(self) -> str:
        with self._lock:
            if self._model is None:
                if not self._started:
                    return "idle"
                return "invalid" if self._error else "loading"
            return self._model.status

    @property
    def ready(self) -> bool:
        with self._lock:
            return bool(self._model is not None and self._model.ready)

    @property
    def labels(self) -> list[str]:
        with self._lock:
            return list(self._model.labels) if self._model is not None else []

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def wait(self, timeout: float | None = None) -> RuntimeModel | AutoRuntimeModel | None:
        self.start()
        self._thread.join(timeout)
        with self._lock:
            return self._model

    def predict(self, input_data: np.ndarray) -> RuntimePrediction:
        with self._lock:
            model = self._model
            error = self._error
        if model is None:
            raise RuntimeError(error or "Modelo cargando.")
        return model.predict(input_data)

    def as_event_payload(self) -> dict[str, object]:
        with self._lock:
            model = self._model
            error = self._error
            started = self._started
        if model is not None:
            return model.as_event_payload()
        return {
            "mode": self._args[1],
            "title": "Modelo cargando",
            "status": "loading" if started and not error else "idle" if not error else "invalid",
            "label_count": 0,
            "labels": [],
            "error": error,
        }


def start_runtime_model_load(
    app_dir: str | Path,
    mode: RecognitionMode = "word",
    *,
    model_path: str | Path | None = None,
    labels_path: str | Path | None = None,
    frames: int = FRAME_COUNT,
    keypoints: int = KEYPOINT_SIZE,
    threshold: float = CONFIDENCE_THRESHOLD,
    start_immediately: bool = True,
) -> AsyncRuntimeModel:
    return AsyncRuntimeModel(
        app_dir,
        mode,
        model_path=model_path,
        labels_path=labels_path,
        frames=frames,
        keypoints=keypoints,
        threshold=threshold,
        start_immediately=start_immediately,
    )