import numpy as np
import cv2
import joblib
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
from wavelet import w2d

# FastAPI app initialization
app = FastAPI()


# Load model and class mappings at startup
@app.on_event("startup")
def load_model():
    global model, class_number_to_name
    with open("saved_model.pkl", "rb") as model_file:
        model = joblib.load(model_file)
    with open("class_dictionary.json", "r") as f:
        class_number_to_name = {v: k for k, v in json.load(f).items()}


# Preprocess image: resize, apply wavelet transform, and flatten
def preprocess_image(image_bytes: bytes):
    img = np.array(Image.open(BytesIO(image_bytes)))
    img_resized = cv2.resize(img, (32, 32))
    img_har_resized = cv2.resize(w2d(img, "db1", 5), (32, 32))
    return np.hstack((img_resized.flatten(), img_har_resized.flatten())).reshape(1, -1)


# FastAPI endpoint for image classification
@app.post("/classify_image")
async def classify_image(file: UploadFile = File(...)):
    try:
        # Preprocess image and make prediction
        image_data = await file.read()
        processed_image = preprocess_image(image_data)
        predicted_class = model.predict(processed_image)[0]
        predicted_name = class_number_to_name.get(predicted_class, "Unknown")
        return {"message": f"The person in the photo is: {predicted_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")


# Run the app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app)
