# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('asl_hand_sign_data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Model trained
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print(f"Model trained with accuracy: {model.score(X_test, y_test) * 100:.2f}%")
joblib.dump(model, 'asl_model.pkl') # model saved as pkl file for prediction.
