from flask import Flask, request, render_template_string
from sklearn.datasets import fetch_olivetti_faces
import joblib

app = Flask(__name__)

model = joblib.load("savedmodel.pth")
data = fetch_olivetti_faces()

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Face Predictor</title>
</head>
<body>
    <h2>Olivetti Face Predictor</h2>

    <form method="POST">
        Enter Index (0-399):
        <input type="number" name="index" min="0" max="399" required>
        <input type="submit" value="Predict">
    </form>

    {% if prediction is not none %}
    <h3>Predicted Class: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        idx = int(request.form["index"])
        sample = data.data[idx].reshape(1, -1)
        prediction = model.predict(sample)[0]

    return render_template_string(
        HTML,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
