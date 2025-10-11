# app/ml/model_zoo.py
from tensorflow import keras
from tensorflow.keras import layers

def build_model(model_id: int, input_shape):
    """
    Cinco arquiteturas progressivamente mais complexas.
    Todas compiladas com Adam/MSE + MAE para comparação.
    """
    m = None

    if model_id == 1:
        # Base (a que você mandou)
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])

    elif model_id == 2:
        # LSTM + residual densa
        x_in = layers.Input(shape=input_shape)
        x = layers.LSTM(64, return_sequences=True)(x_in)
        x = layers.LSTM(64)(x)
        h = layers.Dense(32, activation='relu')(x)
        out = layers.Dense(1)(layers.Concatenate()([x, h]))
        m = keras.Model(x_in, out)

    elif model_id == 3:
        # Bidirecional + Dropout maior
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

    elif model_id == 4:
        # CNN + LSTM (captura motivos locais + dependência longa)
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv1D(32, 3, padding='causal', activation='relu'),
            layers.Conv1D(32, 3, padding='causal', activation='relu'),
            layers.MaxPooling1D(2),
            layers.LSTM(64),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

    elif model_id == 5:
        # Stacked LSTM + LayerNorm + Dropout
        m = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.LSTM(96, return_sequences=True),
            layers.LayerNormalization(),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])

    else:
        raise ValueError("model_id deve ser 1..5")

    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

MODEL_NAMES = {
    1: "LSTM→LSTM (64/32) + Dense",
    2: "LSTM (64/64) + skip Dense",
    3: "BiLSTM (64/32) + Dropout",
    4: "CNN(Conv1D) + LSTM",
    5: "Stacked LSTM (96/64/32) + LN/Dropout",
}
