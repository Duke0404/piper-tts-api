from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from piper.voice import PiperVoice
import tempfile
import os
from pathlib import Path
import requests

app = FastAPI()

MODELS_DIR = Path("/app/models")
MODEL_URL = os.getenv("VOICE_MODEL_URL", "http://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/high/en_US-lessac-high.onnx")
MODEL_PATH = MODELS_DIR / "model.onnx"

# Download model from Hugging Face Hub if not exists
def download_model():
    MODELS_DIR.mkdir(exist_ok=True)
    
    if not MODEL_PATH.exists():
        try:
            print(f"Downloading model from {MODEL_URL}")
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model downloaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    try:
        download_model()
        app.state.voice = PiperVoice.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize TTS: {str(e)}")

@app.post("/speak")
async def speak(text: str):
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            app.state.voice.synthesize(text, temp_file)
            temp_file.flush()
            return FileResponse(
                temp_file.name,
                media_type="audio/wav",
                filename="speech.wav"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file.name):
            os.remove(temp_file.name)

@app.get("/health")
def health():
    return {"status": "ok"}