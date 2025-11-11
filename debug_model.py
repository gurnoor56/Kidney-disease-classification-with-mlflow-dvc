from tensorflow.keras.models import load_model
import os

p = os.path.join("artifacts","training","model.h5")
print("Model path exists:", os.path.exists(p), "->", p)

m = load_model(p)
print("Model loaded:", type(m))
m.summary()
