import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# ✅ Path to your trained model
model_path = "artifacts/training/model.h5"
print("Model path:", model_path, "Exists:", os.path.exists(model_path))

# ✅ Load model
model = load_model(model_path)
print("✅ Model loaded successfully!")

# ✅ Load any kidney test image you want
img_path = "inputImage.jpg"   # <-- put a test image with this name in root folder
print("Image path:", img_path, "Exists:", os.path.exists(img_path))

img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0) / 255.0   # ✅ Important: normalize!

# ✅ Predict
pred = model.predict(img)
print("\nRaw prediction output:", pred)

result = np.argmax(pred, axis=1)
print("Predicted class index:", result)

if result[0] == 1:
    print("✅ Prediction: TUMOR")
else:
    print("✅ Prediction: NORMAL")
