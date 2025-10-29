import json
import requests
import numpy as np
from tensorflow.keras.preprocessing import image

# 1️⃣ Pathing to the test image
img_path = "C:/Code/ML Projects/Tomato Disease Classification with Leaf Images/test_image.jpg"

# 2️⃣ Preprocessing the image like the model was trained
img = image.load_img(img_path, target_size=(224, 224))  # use your model’s input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # normalize if your training used normalization

# 3️⃣ Preparing the data for TensorFlow Serving format
data = json.dumps({"instances": img_array.tolist()})

# 4️⃣ Defining the model’s REST API endpoint
url = "http://localhost:8501/v1/models/tomato_model:predict"

# 5️⃣ Sending the POST request
response = requests.post(url, data=data, headers={"content-type": "application/json"})

# 6️⃣ Printing the result
print("Status Code:", response.status_code)
print("Response JSON:")
print(response.json())