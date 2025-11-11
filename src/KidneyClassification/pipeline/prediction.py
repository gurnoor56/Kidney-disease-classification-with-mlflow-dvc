import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # ✅ Correct model path
        model = load_model("artifacts/training/model.h5")

        # ✅ Preprocess image
        img = image.load_img(self.filename, target_size=(224,224))
        img = image.img_to_array(img)
        img = img / 255.0  # normalize
        img = np.expand_dims(img, axis=0)

        # ✅ Prediction
        result = np.argmax(model.predict(img), axis=1)

        # ✅ Correct class mapping
        if result[0] == 0:
            prediction = "Normal"
        else:
            prediction = "Tumor"

        return [{"image": prediction}]
