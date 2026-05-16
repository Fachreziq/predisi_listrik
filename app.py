from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model paling ringan
model = joblib.load('models/linear_regression.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():

    voltage = float(request.form['voltage'])
    intensity = float(request.form['intensity'])
    sub1 = float(request.form['sub1'])
    sub2 = float(request.form['sub2'])
    sub3 = float(request.form['sub3'])

    features = np.array([
        [voltage, intensity, sub1, sub2, sub3]
    ])

    prediction = model.predict(features)[0]

    return render_template(
        'result.html',
        prediction=round(prediction, 3),
        voltage=voltage,
        intensity=intensity,
        sub1=sub1,
        sub2=sub2,
        sub3=sub3
    )


if __name__ == '__main__':
    app.run(debug=True)