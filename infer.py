from utils.model import NationalityPredictor
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from typing import List

app = FastAPI()

predictor = NationalityPredictor.from_pretrained('model.pth', device="cpu")

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Nationality Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 500px; margin: auto; }
            input[type=text] { width: 100%; padding: 10px; font-size: 1.1em; }
            button { padding: 10px 20px; font-size: 1.1em; margin-top: 10px; }
            .results { margin-top: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Nationality Predictor</h2>
            <input type="text" id="nameInput" placeholder="Enter a name..." />
            <button onclick="predict()">Predict</button>
            <div class="results" id="results"></div>
        </div>
        <script>
            async function predict() {
                const name = document.getElementById('nameInput').value;
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = "Loading...";
                try {
                    const response = await fetch(`/predict?name=${encodeURIComponent(name)}`);
                    if (!response.ok) {
                        resultsDiv.innerHTML = "Error: " + response.statusText;
                        return;
                    }
                    const data = await response.json();
                    if (data.length === 0) {
                        resultsDiv.innerHTML = "No predictions available.";
                        return;
                    }
                    let html = "<h3>Top 5 Predictions</h3><table><tr><th>Nationality</th><th>Probability</th></tr>";
                    for (const row of data) {
                        html += `<tr><td>${row.nationality}</td><td>${(row.probability*100).toFixed(2)}%</td></tr>`;
                    }
                    html += "</table>";
                    resultsDiv.innerHTML = html;
                } catch (e) {
                    resultsDiv.innerHTML = "Error: " + e;
                }
            }
        </script>
    </body>
    </html>
    """

@app.get("/predict")
def predict(name: str = Query(..., description="Name to predict nationality for")) -> List[dict]:
    """
    Predict the top 5 nationalities for a given name.
    Returns a list of dicts: [{"nationality": ..., "probability": ...}, ...]
    """
    probs = predictor.predict_proba([name])[0].cpu()
    prob_list = list(probs.numpy())
    # Pair nationalities and probabilities, sort, and take top 5
    top5 = sorted(zip(list(predictor.encoder.index_to_class.values()), prob_list[1:]), key=lambda x: x[1], reverse=True)[:5]
    return [{"nationality": nat, "probability": float(f"{prob:.5f}")} for nat, prob in top5]
