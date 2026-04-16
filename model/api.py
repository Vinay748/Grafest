import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import warnings
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

warnings.filterwarnings('ignore')

app = FastAPI(title="Neural Nexus Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Global exception handler — returns full traceback in JSON so errors are visible
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import traceback
    tb = traceback.format_exc()
    print("[GLOBAL ERROR HANDLER]", tb)
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "details": str(exc), "traceback": tb}
    )

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}

CATEGORY_MAP = {
    '1': 'Category 1 - Highly Polluted',
    '2': 'Category 2 - Polluted',
    '3': 'Category 3 - Moderately Clean',
    '4': 'Category 4 - Clean',
    '5': 'Category 5 - Very Clean'
}

class WasteIntelligenceEngine:
    WASTE_TYPES = ['Plastic', 'Organic', 'Metallic', 'E-waste', 'Hazardous', 'Anomalies']
    
    WASTE_PROFILES = {
        0: {'Plastic': 0.35, 'Organic': 0.25, 'Metallic': 0.10, 'E-waste': 0.08, 'Hazardous': 0.12, 'Anomalies': 0.10},
        1: {'Plastic': 0.30, 'Organic': 0.30, 'Metallic': 0.12, 'E-waste': 0.06, 'Hazardous': 0.07, 'Anomalies': 0.15},
        2: {'Plastic': 0.25, 'Organic': 0.35, 'Metallic': 0.08, 'E-waste': 0.04, 'Hazardous': 0.03, 'Anomalies': 0.25},
        3: {'Plastic': 0.15, 'Organic': 0.40, 'Metallic': 0.05, 'E-waste': 0.02, 'Hazardous': 0.01, 'Anomalies': 0.37},
        4: {'Plastic': 0.05, 'Organic': 0.10, 'Metallic': 0.02, 'E-waste': 0.01, 'Hazardous': 0.00, 'Anomalies': 0.82}
    }
    
    POLLUTION_SCORES = {
        0: (9.0, 10.0, 'Critical', 'High'),
        1: (7.0, 8.5, 'Severe', 'High'),
        2: (4.5, 6.5, 'Moderate', 'Medium'),
        3: (2.0, 4.0, 'Low', 'Low'),
        4: (0.0, 1.5, 'Minimal', 'Low')
    }
    
    ACTION_RECOMMENDATIONS = {
        0: [
            'EMERGENCY: Deploy hazardous waste cleanup team IMMEDIATELY',
            'Isolate area — potential toxic/chemical contamination',
            'Report to environmental regulatory authorities',
        ],
        1: [
            'HIGH PRIORITY: Schedule professional cleanup within 48 hours',
            'Deploy waste segregation teams (plastic, metal, organic)',
            'Alert local community via notification system',
        ],
        2: [
            'MODERATE: Schedule cleanup within 1 week',
            'Increase waste bin capacity in the area',
            'Regular sweeping and maintenance schedule',
        ],
        3: [
            'LOW PRIORITY: Routine maintenance adequate',
            'Ensure adequate waste bin availability',
            'Monitor for any degradation trends',
        ],
        4: [
            'EXCELLENT: Area is well-maintained!',
            'Continue current maintenance practices',
            'Use as benchmark/reference area',
        ]
    }
    
    @staticmethod
    def analyze(predicted_class, confidence, class_probs):
        waste_composition = {wtype: 0.0 for wtype in WasteIntelligenceEngine.WASTE_TYPES}
        for cls_idx in range(5):
            cls_prob = float(class_probs[cls_idx])  # cast numpy.float32 → Python float
            profile = WasteIntelligenceEngine.WASTE_PROFILES[cls_idx]
            for wtype, pct in profile.items():
                waste_composition[wtype] += pct * cls_prob
                
        total = sum(waste_composition.values())
        waste_composition = {k: v/total for k, v in waste_composition.items()}
        
        pollution_score = 0.0
        for cls_idx in range(5):
            low, high, _, _ = WasteIntelligenceEngine.POLLUTION_SCORES[cls_idx]
            mid = (low + high) / 2
            pollution_score += mid * float(class_probs[cls_idx])  # cast: numpy.float32 → Python float
            
        _, _, severity, level = WasteIntelligenceEngine.POLLUTION_SCORES[predicted_class]
        
        color_map = {
            'Anomalies': 'var(--color-tertiary)',
            'Organic': 'var(--color-secondary)',
            'Plastic': 'var(--color-primary)',
            'Metallic': 'var(--color-outline-variant)',
            'E-waste': 'var(--color-error)',
            'Hazardous': 'var(--color-error-dim)'
        }
        
        frontend_comp = []
        for k, v in waste_composition.items():
            frontend_comp.append({
                "label": k,
                "pct": round(float(v) * 100, 1),  # cast: numpy.float32 → Python float
                "color": color_map.get(k, "var(--color-primary)")
            })
            
        frontend_comp.sort(key=lambda x: x["pct"], reverse=True)
        
        return {
            'classification': CATEGORY_MAP[str(predicted_class + 1)],
            'confidence': round(float(confidence) * 100, 1),
            'pollutionScore': round(pollution_score, 1),
            'severity': severity,
            'composition': frontend_comp,
            'topActions': WasteIntelligenceEngine.ACTION_RECOMMENDATIONS[predicted_class][:3]
        }

class WasteIntelligenceModel(nn.Module):
    def __init__(self, num_classes=5, dropout_rate=0.4):
        super(WasteIntelligenceModel, self).__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

print("====================================")
print(" Neural Nexus PyTorch Engine Booting")
print(f" Target Device: {device}")

model = WasteIntelligenceModel(num_classes=5).to(device)

# Load the weights provided by the user
weights_path = 'neural_nexus_model_final.pth'
try:
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    print(f" Weights successfully loaded from {weights_path}!")
except Exception as e:
    print(f" Error loading {weights_path}: {e}")
    # Try alternative name just in case
    try:
        model.load_state_dict(torch.load('best_model.pth', map_location=device, weights_only=False))
        print(" Weights successfully loaded from best_model.pth!")
    except Exception as e2:
         print(f" Error loading best_model.pth: {e2}")

model.eval()
print("====================================")

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from pydantic import BaseModel
import base64

class ImagePayload(BaseModel):
    image_base64: str

@app.post("/predict_base64")
async def predict_base64(payload: ImagePayload):
    try:
        image_data = base64.b64decode(payload.image_base64)
        try:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as img_err:
            return {"error": "Image Decode Error", "details": str(img_err)}
            
        img_tensor = val_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        result = WasteIntelligenceEngine.analyze(predicted_class, confidence, probs)
        return result
    except Exception as e:
        import traceback
        return {"error": "Internal Processing Error", "details": str(e), "traceback": traceback.format_exc()}

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        try:
            img = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as img_err:
            return {"error": "Image Decode Error", "details": str(img_err)}
            
        img_tensor = val_transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            
        predicted_class = int(np.argmax(probs))
        confidence = float(np.max(probs))
        
        result = WasteIntelligenceEngine.analyze(predicted_class, confidence, probs)
        return result
    except Exception as e:
        import traceback
        return {"error": "Internal Processing Error", "details": str(e), "traceback": traceback.format_exc()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,           # Disabled: reload causes subprocess isolation that hides tracebacks
        limit_max_requests=None,
        timeout_keep_alive=120,
    )
