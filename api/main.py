import os
import time
import zipfile
import tempfile
import asyncio
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
# Add parent dir to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import prediction, preprocessing, model

app = FastAPI(title="Garbage Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "garbage_classifier_v1.h5"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

# Global state for monitoring
STATUS = {
    "is_retraining": False,
    "last_retrain_metrics": None,
    "was_deployed": False,
    "merged_images_count": 0
}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Garbage Classification API"}


@app.get("/uptime")
def get_uptime():
    uptime_seconds = time.time() - START_TIME
    return {"uptime_seconds": uptime_seconds, "status": "running"}


@app.get("/metrics")
def get_metrics():
    return {
        "is_retraining": STATUS["is_retraining"],
        "last_retrain_metrics": STATUS["last_retrain_metrics"],
        "was_deployed": STATUS["was_deployed"],
        "merged_images_count": STATUS["merged_images_count"]
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model file not found. Please train the model first.")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = prediction.predict_image(MODEL_PATH, tmp_path)
        os.remove(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/upload-bulk")
async def upload_bulk(file: UploadFile = File(...)):
    """Accepts a ZIP file containing class folders with images."""
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Must be a ZIP file")
        
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "upload.zip"
            with open(zip_path, "wb") as f:
                f.write(await file.read())
            
            extract_dir = Path(tmpdir) / "extracted"
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # Find class folders
            class_dirs = [d for d in extract_dir.iterdir() if d.is_dir() and d.name in preprocessing.CLASS_NAMES]
            if not class_dirs:
                # check one level deeper, ignoring __MACOSX
                subdirs = [d for d in extract_dir.iterdir() if d.is_dir() and d.name != "__MACOSX"]
                for sd in subdirs:
                    potential_class_dirs = [d for d in sd.iterdir() if d.is_dir() and d.name in preprocessing.CLASS_NAMES]
                    if potential_class_dirs:
                        class_dirs = potential_class_dirs
                        extract_dir = sd
                        break
            
            if not class_dirs:
                raise HTTPException(status_code=400, detail="ZIP file does not contain valid class folders")
            
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            merged_count = preprocessing.merge_uploads_into_raw(extract_dir, RAW_DIR)
            STATUS["merged_images_count"] += merged_count
            
        return {"message": f"Successfully merged {merged_count} images", "merged_count": merged_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")


def _run_retrain_task():
    try:
        STATUS["is_retraining"] = True
        
        # 1. Rebuild train/test dirs from RAW_DIR
        preprocessing.build_train_test_dirs(RAW_DIR, TRAIN_DIR, TEST_DIR)
        
        # 2. Trigger retraining
        metrics, was_deployed = model.retrain_model(
            existing_model_path=MODEL_PATH,
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR
        )
        
        STATUS["last_retrain_metrics"] = metrics
        STATUS["was_deployed"] = was_deployed
        
        if was_deployed:
            prediction.invalidate_model_cache()
            
    except Exception as e:
        STATUS["last_retrain_metrics"] = {"error": str(e)}
    finally:
        STATUS["is_retraining"] = False


@app.post("/retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    if STATUS["is_retraining"]:
        return JSONResponse(status_code=409, content={"message": "Retraining is already in progress"})
    
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Base model not found. Cannot retrain without an initial model.")
        
    background_tasks.add_task(_run_retrain_task)
    return {"message": "Retraining started in the background"}
