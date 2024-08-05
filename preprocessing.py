import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Step 1: Load and Prepare Data
df = combined_df
df.set_index("Timestamp", inplace=True)
df.sort_index(inplace=True)

print(df)
df.dropna(inplace=True)
X = df
y = df[['Level1', 'Level2', 'PCPV', 'DT']]

print(f"x shape: {X.shape}")
print(f"y shape: {y.shape}")

timestamps = df.index
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test)

sequence_length = 10  # Use a sequence length of 10 for time series
num_features = X_train_scaled.shape[1]

# Create sequences function
def create_sequences(data, target, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = target[i:i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

X_train_input, y_train_input = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_test_input, y_test_input = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

print(f"Training data shape: {X_train_input.shape}")
print(f"Testing data shape: {X_test_input.shape}")

# Step 2: Define Transformer Components

# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def get_angles(self, pos, i):
        angle_rates = 1 / tf.pow(tf.cast(10000, tf.float32), (2 * (i // 2)) / tf.cast(self.d_model, tf.float32))
        return pos * angle_rates

    def call(self, inputs):
        seq_length = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_length, delta=1, dtype=tf.float32)[:, tf.newaxis]
        angles = self.get_angles(positions, tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :])
        
        angles = tf.concat([tf.sin(angles[:, 0::2]), tf.cos(angles[:, 1::2])], axis=-1)
        
        pos_encoding = angles[tf.newaxis, ...]

        return inputs + pos_encoding

# Encoder Layer
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None):
        attn_output = self.mha(x, x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Dense(d_model)
        self.pos_encoding = PositionalEncoding(d_model, maximum_position_encoding)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training)

        return x

# Transformer Model
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)  # Change target_vocab_size to 4

    def call(self, inp, training=None):
        enc_output = self.encoder(inp, training=training)
        final_output = self.final_layer(enc_output)
        return final_output
