from flask import Flask, render_template, request
import pickle
import numpy as np
from twilio.rest import Client
from dotenv import load_dotenv
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()

TWILIO_SID = os.getenv("TWILIO_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH")
TWILIO_PHONE = os.getenv("TWILIO_PHONE")
MY_PHONE = os.getenv("MY_PHONE_NUMBER")

client = Client(TWILIO_SID, TWILIO_AUTH)


# -----------------------------
# Flask app setup
# -----------------------------
app = Flask(__name__)

# Load Ridge model and scaler
with open('models/ridge.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURE_ORDER = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']

@app.route('/')
def home():
    return render_template('sliders.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input values
        input_data = {
            'Temperature': float(request.form['temperature']),
            'RH': float(request.form['rh']),
            'Ws': float(request.form['ws']),
            'Rain': float(request.form['rain']),
            'FFMC': float(request.form['ffmc']),
            'DMC': float(request.form['dmc']),
            'ISI': float(request.form['isi']),
            'Classes': int(request.form['classes']),
            'Region': int(request.form['region'])
        }

        # Arrange features in correct order
        features = np.array([[input_data[f] for f in FEATURE_ORDER]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict fire risk
        prediction = model.predict(features_scaled)[0]

        # Normalize between 0 and 1
        prediction_clamped = max(0, min(1, prediction))

        # Determine risk level
        if prediction_clamped >= 0.5:
            result = "High Fire Risk"
            color = "red"

            # -----------------------------
            # Send SMS alert (only to your number)
            # -----------------------------
            alert_message = (
                f"🔥 ALERT: High Fire Risk detected in Region {input_data['Region']}! "
                f"Temperature: {input_data['Temperature']}°C, "
                f"Humidity: {input_data['RH']}%. "
                f"Take precautions immediately!"
            )
            try:
                message = client.messages.create(
                    body=alert_message,
                    from_=TWILIO_NUM,
                    to=MY_PHONE
                )
                print("✅ SMS alert sent successfully!")
            except Exception as sms_error:
                print(f"⚠️ SMS sending failed: {sms_error}")

        else:
            result = "Low Fire Risk"
            color = "green"

        # Show result on web page
        return render_template('result.html', result=result, score=prediction_clamped, color=color)

    except Exception as e:
        return render_template('result.html', result=f"Error: {str(e)}", score=0, color="black")


if __name__ == '__main__':
    app.run(debug=True)
