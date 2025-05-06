from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("images", exist_ok=True)
app.mount("/images", StaticFiles(directory="images"), name="images")


def compare_with_ssim(ref_bytes, img_bytes):
    ref_img_color = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
    img_color = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Resize to match
    if ref_img_color.shape != img_color.shape:
        img_color = cv2.resize(img_color, (ref_img_color.shape[1], ref_img_color.shape[0]))

    ref_gray = cv2.cvtColor(ref_img_color, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # SSIM score and difference map
    score, diff = ssim(ref_gray, img_gray, full=True)
    diff = (diff * 255).astype("uint8")

    # Threshold to find different regions
    thresh = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY_INV)[1]

    # Find contours of differences
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw red rectangles where differences found
    for c in contours:
        if cv2.contourArea(c) > 40:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 0, 255), 2)

    is_match = bool(score == 1.0)  # Adjust threshold if needed
    is_unmatched = bool(score * 100 < 100) 


    filename = None
    if is_unmatched:
        filename = f"{uuid.uuid4().hex}.jpg"
        path = os.path.join("images", filename)
        cv2.imwrite(path, img_color)


    return is_match, score * 100, filename


@app.post("/compare")
async def compare(reference: UploadFile = File(...), image: UploadFile = File(...)):
    reference_bytes = await reference.read()
    image_bytes = await image.read()

    is_match, match_percent, filename = compare_with_ssim(reference_bytes, image_bytes)

    response = {
        "is_match": is_match,
        "match_percent": round(match_percent, 2),
        "image_url": f"/images/{filename}" if filename else None
    }

    return JSONResponse(content=response)
