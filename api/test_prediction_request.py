import json
import numpy as np
import requests
from tensorflow.keras.preprocessing import image

# Pathing to the test image
img_path = r"C:\Code\ML Projects\Tomato Disease Classification with Leaf Images\test_images\leaf2.JPG"

# Preprocessing image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize

# Preparing request data
data = json.dumps({"instances": img_array.tolist()})
headers = {"content-type": "application/json"}

# Sending request to TensorFlow Serving (Docker)
url = "http://localhost:8501/v1/models/tomato_model:predict"
response = requests.post(url, data=data, headers=headers)

# Parsing prediction response
print(f"Status Code: {response.status_code}")
response_json = response.json()
print("Response JSON:")
print(json.dumps(response_json, indent=2))

# Decoding predicted class
class_names = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

# Extracting prediction
preds = np.array(response_json["predictions"][0])
predicted_index = np.argmax(preds)
confidence = preds[predicted_index] * 100

# Printing final readable result
print("\n🧾 Prediction Summary:")
print(f"Predicted Disease: {class_names[predicted_index]}")
print(f"Confidence: {confidence:.2f}%")