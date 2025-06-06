from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
import tensorflow as tf
import os
from tensorflow.keras import layers, Model
from tensorflow.keras.saving import register_keras_serializable


@register_keras_serializable()
class GainGenerator(Model):
    def __init__(self, input_dim=5, hidden_dim=64, **kwargs):
        super(GainGenerator, self).__init__(**kwargs)
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(hidden_dim, activation="relu")
        self.output_layer = layers.Dense(input_dim, activation="linear")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

MODEL_DIR = 'model_assets'  # 必須先定義


gain_model = load_model(
    os.path.join(MODEL_DIR, "gain_generator_model_with_toolwear.keras"),
    custom_objects={"GainGenerator": GainGenerator}
)

app = Flask(__name__)

INPUT_COLS = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
FULL_COLS = ['Air temperature', 'Rotational speed', 'Tool wear', 'Torque',
             'Process temperature', 'Power', 'PowerWear', 'TempPerPower']

model_bin = xgb.XGBClassifier()
model_bin.load_model(os.path.join(MODEL_DIR, "best_predictive_bin_model"))

model_multi = xgb.XGBClassifier()
model_multi.load_model(os.path.join(MODEL_DIR, "best_predictive_multi_model"))

scaler = joblib.load(os.path.join(MODEL_DIR, "gain_scaler_with_toolwear.pkl"))

def gain_impute(input_df):
    arr = scaler.transform(input_df.values.reshape(1, -1))
    nan_idx = np.isnan(arr[0])
    mask = (~nan_idx).astype(np.float32).reshape(1, -1)
    arr_filled = arr.copy()
    arr_filled[0, nan_idx] = 0.0
    generator_input = np.concatenate([arr_filled, mask], axis=1)
    x_tilde = gain_model(tf.convert_to_tensor(generator_input)).numpy()
    arr[0, nan_idx] = x_tilde[0, nan_idx]
    imputed = scaler.inverse_transform(arr)
    return pd.DataFrame(imputed, columns=INPUT_COLS)

def add_derived_features(df):
    df['Power'] = df['Torque'] * df['Rotational speed']
    df['PowerWear'] = df['Power'] * df['Tool wear']
    df['TempPerPower'] = df['Process temperature'] / (df['Power'] + 1e-6)
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_bin = None
    prediction_multi = None
    imputed_values = None
    input_values = {col: "" for col in INPUT_COLS}

    if request.method == "POST":
        values = []
        for col in INPUT_COLS:
            val = request.form.get(col)
            input_values[col] = val
            try:
                values.append(float(val))
            except:
                values.append(np.nan)

        input_df = pd.DataFrame([values], columns=INPUT_COLS)

        if input_df.isnull().values.any():
            imputed_df = gain_impute(input_df.copy())
            imputed_values = imputed_df.iloc[0].to_dict()
            input_df = imputed_df

        input_df = add_derived_features(input_df)
        input_df = input_df[FULL_COLS]

        pred_bin = model_bin.predict(input_df)[0]
        pred_multi = model_multi.predict(input_df)[0]

        prediction_bin = f"Binary Failure Prediction: {'Failure' if pred_bin == 1 else 'No Failure'}"
        multi_class_labels = {
                0: 'No Failure',
                1: 'Heat Dissipation Failure (HDF)',
                2: 'Power Failure (PWF)',
                3: 'Overstrain Failure (OSF)',
                4: 'Tool Wear Failure (TWF)',
                5: 'Random Failure (RNF)'
            }
        prediction_multi = f"Multi-class Prediction: {multi_class_labels.get(pred_multi, 'Unknown')}"


    return render_template("index.html",
                           prediction_bin=prediction_bin,
                           prediction_multi=prediction_multi,
                           input_values=input_values,
                           imputed_values=imputed_values)

if __name__ == "__main__":
    print("載入 XGBoost binary model 成功")
    print("載入 XGBoost multi model 成功")
    print("載入 GAIN 補值模型成功")
    print("載入 GAIN scaler 成功")
    app.run(debug=True, port=5000)
