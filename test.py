import joblib
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("savedmodel.pth")

# Load test data
X_test, y_test = joblib.load("test_data.pkl")

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {accuracy:.4f}")
