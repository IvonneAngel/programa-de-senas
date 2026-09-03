from __future__ import annotations

from typing import Any


def build_tiny_conv_classifier(frame_count: int, keypoint_size: int, class_count: int) -> Any:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(frame_count, keypoint_size)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def build_small_conv_classifier(frame_count: int, keypoint_size: int, class_count: int) -> Any:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(frame_count, keypoint_size)),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(96, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(96, kernel_size=3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def build_lstm_classifier(frame_count: int, keypoint_size: int, class_count: int) -> Any:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(frame_count, keypoint_size)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(class_count, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


def build_classifier(
    frame_count: int,
    keypoint_size: int,
    class_count: int,
    architecture: str = "tiny-conv",
) -> Any:
    if architecture == "tiny-conv":
        return build_tiny_conv_classifier(frame_count, keypoint_size, class_count)
    if architecture == "small-conv":
        return build_small_conv_classifier(frame_count, keypoint_size, class_count)
    if architecture == "lstm":
        return build_lstm_classifier(frame_count, keypoint_size, class_count)
    raise ValueError(f"Unknown architecture: {architecture}")