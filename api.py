from pathlib import Path
import base64
import io

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


app = FastAPI(
    title="RoadDamageAI API",
    description="API untuk mendeteksi pothole menggunakan YOLO",
    version="1.0.0",
)

MODEL_PATH = (
    Path(__file__).parent
    / "train-2"
    / "weights"
    / "best.pt"
)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model tidak ditemukan: {MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))


@app.get("/")
def home():
    return {
        "message": "RoadDamageAI API berjalan",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/jpg",
    ):
        raise HTTPException(
            status_code=400,
            detail="Unggah gambar JPG atau PNG.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="File gambar kosong.",
        )

    try:
        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="File yang diunggah bukan gambar valid.",
        )

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False,
    )

    result = results[0]
    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            coordinates = (
                box.xyxy[0].tolist()
            )

            x_min, y_min, x_max, y_max = [
                int(value) for value in coordinates
            ]

            detections.append({
                "bounding_box": [
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ],
                "class_label": result.names[class_id],
                "confidence_score": round(
                    confidence, 4
                ),
            })

    # YOLO menghasilkan gambar anotasi dalam format BGR.
    annotated_image = result.plot()

    # Ubah BGR menjadi RGB agar warna gambar benar.
    annotated_pil = Image.fromarray(
        annotated_image[:, :, ::-1]
    )

    buffer = io.BytesIO()
    annotated_pil.save(buffer, format="JPEG")

    annotated_base64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return {
        "filename": file.filename,
        "total_detections": len(detections),
        "detections": detections,
        "annotated_image": annotated_base64,
    }