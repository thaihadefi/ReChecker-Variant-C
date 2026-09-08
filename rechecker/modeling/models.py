"""Shared BiLSTM-attention backbone for B0-B4."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from rechecker.representation.sequences import PAD_SEGMENT
from rechecker.representation.variants import VariantSpec

from .attention import MaskedAttention


def _classifier(inputs, valid_mask, config):
    encoded = layers.Bidirectional(
        layers.LSTM(config.hidden_units, return_sequences=True), name="bilstm"
    )(inputs, mask=valid_mask)
    context = MaskedAttention(name="attention")(encoded, mask=valid_mask)
    context = layers.ReLU(name="context_relu")(context)
    context = layers.Dropout(config.dropout, name="context_dropout")(context)
    context = layers.Dense(
        config.dense_units, activation="relu", name="classifier"
    )(context)
    context = layers.Dropout(config.dropout, name="classifier_dropout")(context)
    return layers.Dense(2, activation="softmax", name="prediction")(context)


def _compile(model: Model, learning_rate: float) -> Model:
    model.compile(
        optimizer=tf.keras.optimizers.Adamax(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_model(config, spec: VariantSpec) -> Model:
    vectors = layers.Input(
        shape=(config.max_len, config.vector_dim), name="token_vectors"
    )
    if spec.use_segment_embedding:
        segments = layers.Input(
            shape=(config.max_len,), dtype="int32", name="segment_ids"
        )
        valid = layers.Lambda(
            lambda value: tf.not_equal(value, PAD_SEGMENT), name="padding_mask"
        )(segments)
        segment_vectors = layers.Embedding(
            input_dim=4,
            output_dim=config.vector_dim,
            name="segment_embedding",
        )(segments)
        masked_segments = layers.Multiply(name="masked_segment_embedding")([
            segment_vectors,
            layers.Lambda(
                lambda value: tf.cast(value[..., tf.newaxis], tf.float32)
            )(valid),
        ])
        encoded_input = layers.Add(name="input_representation")([
            vectors, masked_segments
        ])
        inputs = [vectors, segments]
    else:
        valid = layers.Input(
            shape=(config.max_len,), dtype="bool", name="token_mask"
        )
        encoded_input = vectors
        inputs = [vectors, valid]

    output = _classifier(encoded_input, valid, config)
    return _compile(Model(inputs, output, name=f"rechecker_{spec.name}"), config.learning_rate)
