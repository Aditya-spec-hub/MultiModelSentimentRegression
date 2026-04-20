from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model


def _rnn_block(x, rnn_type: str, units: int = 64):
    rnn_type = rnn_type.lower()
    if rnn_type == "gru":
        return layers.GRU(units, return_sequences=True)(x)
    if rnn_type == "bigru":
        return layers.Bidirectional(layers.GRU(units, return_sequences=True))(x)
    return layers.LSTM(units, return_sequences=True)(x)  # default lstm


def _attn_pool(x, name: str):
    score = layers.Dense(1, activation="tanh", name=f"{name}_score")(x)      # (B,T,1)
    weights = layers.Softmax(axis=1, name=f"{name}_softmax")(score)           # (B,T,1)
    weighted = layers.Multiply(name=f"{name}_weighted")([x, weights])         # (B,T,C)
    pooled = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1), name=f"{name}_sum")(weighted)  # (B,C)
    return pooled


def build_model(
    rnn_type: str = "lstm",
    use_attention: bool = True,
    max_len: int = 100,
    visual_dim: int = 35,
    audio_dim: int = 74,
    text_dim: int = 300,
    hidden_units: int = 64,
):
    """
    Dynamic-shape multimodal model.
    IMPORTANT: pass dims from your prepared tensors in main.py.
    """
    v_in = layers.Input(shape=(max_len, visual_dim), name="visual")
    a_in = layers.Input(shape=(max_len, audio_dim), name="audio")
    t_in = layers.Input(shape=(max_len, text_dim), name="text")

    hv = _rnn_block(v_in, rnn_type=rnn_type, units=hidden_units)
    ha = _rnn_block(a_in, rnn_type=rnn_type, units=hidden_units)
    ht = _rnn_block(t_in, rnn_type=rnn_type, units=hidden_units)

    if use_attention:
        fv = _attn_pool(hv, "visual")
        fa = _attn_pool(ha, "audio")
        ft = _attn_pool(ht, "text")
    else:
        fv = layers.GlobalAveragePooling1D(name="visual_gap")(hv)
        fa = layers.GlobalAveragePooling1D(name="audio_gap")(ha)
        ft = layers.GlobalAveragePooling1D(name="text_gap")(ht)

    x = layers.Concatenate(name="fusion")([fv, fa, ft])
    x = layers.Dense(128, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.30, name="drop1")(x)
    x = layers.Dense(64, activation="relu", name="fc2")(x)
    out = layers.Dense(1, name="sentiment")(x)

    return Model(inputs=[v_in, a_in, t_in], outputs=out, name=f"{rnn_type}_multimodal")