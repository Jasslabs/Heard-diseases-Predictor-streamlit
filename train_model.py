import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
try:
    df = pd.read_csv('Heart_Disease_Prediction.csv')
except FileNotFoundError:
    print("Error: Heart_Disease_Prediction.csv not found. Please ensure the file is in the same directory.")
    exit()

# Preprocessing
# Encoding the target variable (Presence = 1, Absence = 0)
le = LabelEncoder()
df['Heart Disease'] = le.fit_transform(df['Heart Disease'])

# Splitting features and target
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model and the label encoder
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("Model trained and saved successfully as 'heart_model.pkl'!")