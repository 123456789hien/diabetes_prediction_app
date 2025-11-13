from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)
model = tf.keras.models.load_model("diabetes_model.h5")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        input_data = np.array(data).reshape(1, -1)

        prediction = model.predict(input_data)[0][0]
        risk = round(float(prediction) * 100, 2)

        if prediction > 0.7:
            result = f"🔴 High Risk ({risk}%) — Immediate medical check-up recommended."
        elif prediction > 0.4:
            result = f"🟠 Moderate Risk ({risk}%) — Maintain balanced lifestyle & recheck soon."
        else:
            result = f"🟢 Low Risk ({risk}%) — Keep up your healthy habits!"

        return render_template('index.html', prediction_text=result)
    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
