from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import cv2
import numpy as np

app = FastAPI()

origins = [
    "*"
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/image")
async def get_images() -> dict:
    return {}


@app.post("/predict")
async def breed_predict(file: UploadFile = File(...)):
    print('Breed Predict...')
    contents = await file.read()

    # read image as an numpy array
    image = np.asarray(bytearray(contents))
        
    # use imdecode function
    image = cv2.resize(cv2.imdecode(image, cv2.IMREAD_COLOR), (299, 299))
    image = image.reshape(1, 299, 299, 3)
    results = breed_prediction(image)
    print(results)
    return {"result": f'{results}'}



async def load_image(file):
    print('Inside Load Image')
    contents = await file.read()

    # read image as an numpy array
    image = np.asarray(bytearray(contents))
        
    # use imdecode function
    image = cv2.resize(cv2.imdecode(image, cv2.IMREAD_COLOR), (299, 299))
    image = image.reshape(1, 299, 299, 3)
    print('Image Loaded')
    print(type(image), image)
    return image

def load_finetuned_model():
    print('Load Fine Tuned Model...')
    finetuned_model = load_model('models/InceptionV3.h5')
    print('Loaded Fine Tuned Model...')
    return finetuned_model

def breed_prediction(image):
    # image = load_image(imageBuffer)
    model = load_finetuned_model()
    print('Image to Predict...')
    print(model)
    print(type(image))
    print(image.shape)
    results = model.predict(image)
    results = np.argmax(results)
    return results
