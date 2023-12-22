from fastapi import FastAPI, UploadFile,File
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import uvicorn
#starting the server just like expresss.js
app=FastAPI()

#loading the models
MODEL=tf.keras.models.load_model("../models/1")
CLASS_NAMES=["Early Blight","Late Blight","Healthy"]




@app.get("/ping")
async def ping():
    return "hello,main jinda hun"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(file: UploadFile= File(...)):
    image=read_file_as_image(await file.read())
    img_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(img_batch)
    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    print(predicted_class,confidence)
    return {
        "class":predicted_class,
        "confidence":float(confidence)
    }
    pass

#for clean debugging we will be using the python entry points
if __name__=="__main__" :
    uvicorn.run(app,host='localhost',port=8000)



 