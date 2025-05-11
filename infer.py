from utils.model import NationalityPredictor
from fastapi import FastAPI, Query
from typing import List

app = FastAPI()

predictor = NationalityPredictor.from_pretrained('model.pth', device="cpu")

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
