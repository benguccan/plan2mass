from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/demo/polygon")
def demo_polygon(height: float = 5):
    return {
        "polygon": [
            [0, 0],
            [12, 0],
            [8, 7],
            [0, 6],
        ],
        "height": height,
    }


@app.post("/upload-plan")
async def upload_plan(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".pdf", ".png", ".jpg", ".jpeg"]:
        return {"ok": False, "error": "Only PDF/PNG/JPG allowed"}

    file_id = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, file_id)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"ok": True, "file_id": file_id, "path": save_path}