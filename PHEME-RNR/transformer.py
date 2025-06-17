
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MultiHeadSelfAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0, "embedding dimension not divisible by num heads"
        self.projection_dim = embed_dim // num_heads
        self.wq = keras.layers.Dense(embed_dim)
        self.wk = keras.layers.Dense(embed_dim)
        self.wv = keras.layers.Dense(embed_dim)
        self.combine_heads = keras.layers.Dense(embed_dim)

    def attention(self, q, k, v):
        score = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dk)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, v)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        q = self.separate_heads(q, batch_size)
        k = self.separate_heads(k, batch_size)
        v = self.separate_heads(v, batch_size)
        attention, _ = self.attention(q, k, v)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        return self.combine_heads(concat_attention)

class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([
            keras.layers.Dense(ff_dim, activation="relu"),
            keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=False):
        attn_output = self.att(x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Prevent training code from executing during import
if __name__ == "__main__":
    embed_dim = 768
    ff_dim = 32
    num_heads = 1

    text_input = keras.Input(shape=(None, 768,), dtype='float32', name='text')
    l_mask = keras.layers.Masking(mask_value=-99.)(text_input)
    encoded_text = TransformerLayer(embed_dim, num_heads, ff_dim)(l_mask)
    out_dense1 = keras.layers.LSTM(100)(encoded_text)
    out_dense = keras.layers.Dense(30, activation='relu')(out_dense1)
    out = keras.layers.Dense(2, activation='softmax')(out_dense)

    model = keras.Model(text_input, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()
