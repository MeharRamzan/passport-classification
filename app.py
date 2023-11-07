from fastapi import FastAPI, UploadFile
import uvicorn
from model.classify import PassportClassifier
from PIL import Image


MODEL = None
APP = FastAPI(title="Passport Classification")

@APP.on_event('startup')
def init_model():
    global MODEL
    # Initialize the model when the app starts
    MODEL = PassportClassifier(model_path='passport_classifier.pth')

# Endpoint for a GET request at the root ("/")
@APP.get("/")
def read_root():
    return {"message": "Server is up"}

# Endpoint for a POST request at "/classify"
@APP.post("/classify")
def classify_image(file: UploadFile):
    # Open the uploaded image
    image = Image.open(file.file)
    
    # Classify the image using the initialized model
    prediction = MODEL.classify(image)
    
    return {"results": prediction}

if __name__ == "__main__":
    # Run the FastAPI app using Uvicorn
    uvicorn.run(APP, host="0.0.0.0", port=8000)
