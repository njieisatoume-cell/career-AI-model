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

@app.route('/predict', methods=['POST'])
def predict():

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
    predicted_career = label_encoder.inverse_transform(prediction)

    return render_template('index.html',
                    prediction_text=f"Recommended Career: {predicted_career[0]}"
    )
if __name__ == "__main__":
    app.run(debug=True)