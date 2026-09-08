"""Attention layers used by every ReChecker ablation."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import initializers, layers


class MaskedAttention(layers.Layer):
    """Single-context additive attention with masked softmax."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.initializer = initializers.get("glorot_uniform")

    def build(self, input_shape):
        width = input_shape[-1]
        self.projection = self.add_weight(
            shape=(width, width), initializer=self.initializer, name="projection"
        )
        self.bias = self.add_weight(shape=(width,), initializer="zeros", name="bias")
        self.context = self.add_weight(
            shape=(width,), initializer=self.initializer, name="context"
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        hidden = tf.tanh(tf.tensordot(inputs, self.projection, axes=1) + self.bias)
        logits = tf.tensordot(hidden, self.context, axes=1)
        if mask is not None:
            logits = tf.where(
                tf.cast(mask, tf.bool), logits, tf.cast(-1e9, logits.dtype)
            )
        weights = tf.nn.softmax(logits, axis=1)
        return tf.reduce_sum(inputs * weights[..., tf.newaxis], axis=1)

    def compute_mask(self, inputs, mask=None):
        return None
