import tensorflow as tf
from tensorflow.keras import layers, initializers
from tensorflow.keras.models import Model


class AttentionWithContext(layers.Layer):
    """TF2/Keras3 rewrite of ReChecker's AttentionWithContext, same equations:
    u_t = tanh(W h_t + b); alpha_t = softmax(u_t^T u); v_t = alpha_t * h_t.
    Functionally identical to the original (additive/Bahdanau, 1 context vector,
    1 head), just using tf ops instead of the legacy keras.backend API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init = initializers.get('glorot_uniform')

    def build(self, input_shape):
        dim = input_shape[-1]
        self.W = self.add_weight(shape=(dim, dim), initializer=self.init, name="att_W")
        self.b = self.add_weight(shape=(dim,), initializer="zeros", name="att_b")
        self.u = self.add_weight(shape=(dim,), initializer=self.init, name="att_u")
        super().build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.exp(ait)
        a = a / (tf.reduce_sum(a, axis=1, keepdims=True) + tf.keras.backend.epsilon())
        a = tf.expand_dims(a, axis=-1)
        return x * a


class Addition(layers.Layer):
    def call(self, x):
        return tf.reduce_sum(x, axis=1)


def build_baseline_model(max_len=100, vec_dim=300, dropout=0.2, lr=0.002):
    """Exact architecture of ReChecker goc's BLSTM_Attention."""
    inp = layers.Input(shape=(max_len, vec_dim), name="token_vectors")
    x = layers.Bidirectional(layers.LSTM(300, return_sequences=True))(inp)
    x = AttentionWithContext()(x)
    x = Addition()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(300)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = Model(inp, out, name="ReChecker_goc")
    model.compile(tf.keras.optimizers.Adamax(learning_rate=lr), "categorical_crossentropy", metrics=["accuracy"])
    return model


def build_phuong_an_c_model(max_len=100, vec_dim=300, dropout=0.2, lr=0.002):
    """Phuong an C: same BiLSTM+Attention backbone, but input token vectors are
    FastText (handled outside this function, at data-prep stage) and a learned
    segment embedding (W=0 / C=1 / [SEP]=2 / PAD=3) is added element-wise to
    the token vector before the BiLSTM, masked to 0 at PAD positions so padded
    steps stay exactly the zero vector (same property as the baseline)."""
    inp_vec = layers.Input(shape=(max_len, vec_dim), name="token_vectors")
    inp_seg = layers.Input(shape=(max_len,), dtype="int32", name="segment_ids")

    seg_emb = layers.Embedding(input_dim=4, output_dim=vec_dim, name="segment_embedding")(inp_seg)
    pad_mask = layers.Lambda(
        lambda t: tf.expand_dims(tf.cast(tf.not_equal(t, 3), tf.float32), axis=-1),
        output_shape=(max_len, 1),
    )(inp_seg)
    seg_emb = layers.Multiply()([seg_emb, pad_mask])

    x = layers.Add()([inp_vec, seg_emb])
    x = layers.Bidirectional(layers.LSTM(300, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Addition()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(300)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(2, activation="softmax")(x)
    model = Model([inp_vec, inp_seg], out, name="Phuong_an_C")
    model.compile(tf.keras.optimizers.Adamax(learning_rate=lr), "categorical_crossentropy", metrics=["accuracy"])
    return model
