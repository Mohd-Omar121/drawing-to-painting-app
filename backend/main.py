from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import io
import os
import requests
from PIL import Image
from backend.huggingface_api import generate_image_local
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Colab backend URL (update this with your current ngrok URL)
COLAB_URL = "https://2f4c007c4900.ngrok-free.app/generate"

@app.post("/generate")
async def generate(
    image: UploadFile = File(...),
    model: str = Form(...),
    theme: str = Form(...),
    prompt: str = Form(...)
):
    try:
        image_bytes = await image.read()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        final_prompt = f"{theme}. {prompt}" if prompt else theme
        
        if model == "Pix2Pix":
            # Run locally
            result_bytes = generate_image_local(input_image, final_prompt)
        else:
            # Send to Colab for ControlNet/FLUX
            files = {"image": ("input.png", image_bytes, "image/png")}
            data = {"model": model, "theme": theme, "prompt": prompt}
            response = requests.post(COLAB_URL, files=files, data=data, timeout=600)
            if response.status_code != 200:
                return JSONResponse(status_code=500, content={"error": f"Colab error: {response.text}"})
            result_bytes = response.content
        
        return StreamingResponse(io.BytesIO(result_bytes), media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
