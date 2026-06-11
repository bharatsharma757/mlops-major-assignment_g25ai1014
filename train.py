from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load dataset
data = fetch_olivetti_faces()

X = data.data
y = data.target

# 70-30 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "savedmodel.pth")

# Save test data
joblib.dump((X_test, y_test), "test_data.pkl")

print("Model saved as savedmodel.pth")
