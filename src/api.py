"""
FastAPI demo endpoint (optional deliverable).

POST an image to /predict and get back: per-label probabilities, the predicted
findings above threshold, a base64 Grad-CAM overlay for the top finding, and the
RAG-grounded LLM interpretation (with disclaimer).

Run:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000/docs
"""
import base64
import io
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File
from PIL import Image

import config as C
from src import models as M
from src.gradcam import GradCAM, overlay_and_save
from src.rag.interpret import interpret

app = FastAPI(title="ChestX-ray14 Assistive Interpreter")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "resnet18"
_model = None


def _get_model():
    global _model
    if _model is None:
        _model = M.get_model(MODEL_NAME, pretrained=False).to(DEVICE)
        ckpt = C.CHECKPOINT_DIR / f"{MODEL_NAME}_best.pt"
        if ckpt.exists():
            _model.load_state_dict(torch.load(ckpt, map_location=DEVICE)["state_dict"])
        _model.eval()
    return _model


def _transform(img: Image.Image):
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((C.IMG_SIZE, C.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])
    return t(img).unsqueeze(0)


@app.get("/")
def root():
    return {"status": "ok", "disclaimer": C.DISCLAIMER}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    x = _transform(img).to(DEVICE)
    model = _get_model()

    with torch.no_grad():
        probs = torch.sigmoid(model(x)).cpu().numpy()[0]
    predictions = {lab: float(p) for lab, p in zip(C.LABELS, probs)}

    # Grad-CAM for the top finding
    top_label = max(predictions, key=predictions.get)
    x.requires_grad_(True)
    cam_engine = GradCAM(model, M.gradcam_target_layer(model, MODEL_NAME))
    cam, _ = cam_engine(x, C.LABELS.index(top_label))
    buf_path = C.FIGURE_DIR / "api_gradcam.png"
    C.ensure_dirs()
    img.save(buf_path.with_suffix(".input.png"))
    overlay_and_save(buf_path.with_suffix(".input.png"), cam, buf_path)
    cam_b64 = base64.b64encode(Path(buf_path).read_bytes()).decode()

    rag = interpret(predictions)
    return {
        "predictions": predictions,
        "top_finding": top_label,
        "gradcam_png_base64": cam_b64,
        "interpretation": rag["summary"],
        "retrieved_ids": rag["retrieved_ids"],
        "disclaimer": C.DISCLAIMER,
    }
