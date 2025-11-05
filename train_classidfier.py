import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle
import os

# Load dataset (replace with your actual dataset path)
df = pd.read_csv('forestfires.csv')  

# Features and target
FEATURES = ['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region']
X = df[FEATURES]
y = df['FireRisk']  # 0 = Low, 1 = High

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_scaled, y)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save classifier and scaler
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Classifier and scaler saved in 'models/' folder.")
