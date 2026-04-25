
import tensorflow as tf
from tensorflow.keras.layers import Dense, Softmax, Concatenate, Multiply


def attention_fusion(v, a, t):

    # Stack modalities → (batch, 3, hidden_dim)
    stacked = tf.stack([v, a, t], axis=1)

    # Compute attention scores
    score = Dense(1)(stacked)  # (batch, 3, 1)

    # Softmax over modalities
    weights = Softmax(axis=1)(score)  # (batch, 3, 1)

    # Weighted sum
    weighted = Multiply()([stacked, weights])
    fused = tf.reduce_sum(weighted, axis=1)  # (batch, hidden_dim)

    return fused
