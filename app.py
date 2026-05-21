from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('career_model.pkl', 'rb'))
label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def predict():

    prediction_text = ""

    if request.method == 'POST':

        # Get values from form
        programming = float(request.form['programming'])
        math = float(request.form['math'])
        study_hours = float(request.form['study_hours'])
        technical = float(request.form['technical'])
        communication = float(request.form['communication'])
        analytical = float(request.form['analytical'])

        # Arrange input data
        features = np.array([[programming, math, study_hours,
                              technical, communication, analytical]])

        # Make prediction
        prediction = model.predict(features)

        # Decode prediction
        predicted_career = label_encoder.inverse_transform(prediction)

        prediction_text = f"Recommended Career: {predicted_career[0]}"

    return render_template('index.html',
                           prediction_text=prediction_text)
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)