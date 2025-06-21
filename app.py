from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

# Define the model architecture
class WineQualityModel(nn.Module):
    def __init__(self):
        super(WineQualityModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = WineQualityModel()
model.load_state_dict(torch.load("wine_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define input data structure
class WineFeatures(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

# FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(features: WineFeatures):
    input_tensor = torch.tensor([[features.fixed_acidity,
                                  features.volatile_acidity,
                                  features.citric_acid,
                                  features.residual_sugar,
                                  features.chlorides,
                                  features.free_sulfur_dioxide,
                                  features.total_sulfur_dioxide,
                                  features.density,
                                  features.pH,
                                  features.sulphates,
                                  features.alcohol]], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()
    return {"predicted_quality": round(prediction)}
