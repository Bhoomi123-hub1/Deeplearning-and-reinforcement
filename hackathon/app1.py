#!/usr/bin/env python
# coding: utf-8

# In[7]:


import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# === Settings ===
img_size = 224

# === Get class names from training folder ===
dataset_dir = r"C:\Users\Ujwal M L\OneDrive\Documents\Internship\Hackathon\dataset"
class_names = sorted(os.listdir(train_dir))
NUM_CLASSES = len(class_names)

# === Rebuild model architecture EXACTLY as used during training ===
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False  # Feature extraction mode

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# === Load saved weights ===
model.load_weights("plant_disease_weights.h5")
print(" Model weights loaded successfully!")

# === Streamlit UI ===
st.set_page_config(page_title="Plant Disease Classifier")
st.title(" Plant Disease Classifier")
st.caption("Upload a leaf image to identify its disease class.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    img = image.load_img(uploaded_file, target_size=(img_size, img_size))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    pred_idx = np.argmax(prediction[0])
    confidence = prediction[0][pred_idx] * 100

    # Display result
    st.success(f" Predicted Class: **{class_names[pred_idx]}**")
    st.write(f" Confidence: `{confidence:.2f}%`")


# In[ ]:


import streamlit as st
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# === Settings ===
img_size = 224

#  Point to the correct training directory
train_dir = r"C:\Users\Ujwal M L\OneDrive\Documents\Internship\Hackathon\dataset\train"
class_names = sorted(os.listdir(train_dir))
NUM_CLASSES = len(class_names)  # Make sure this equals 3

#  Rebuild the same model architecture
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

#  Load weights that match 3-class output
model.load_weights("plant_disease_weights.h5")
print(" Model weights loaded successfully!")

# === Streamlit UI ===
st.set_page_config(page_title="Plant Disease Classifier")
st.title(" Plant Disease Classifier")
st.caption("Upload a leaf image to identify its disease class.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(img_size, img_size))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_idx = np.argmax(prediction[0])
    confidence = prediction[0][pred_idx] * 100

    st.success(f" Predicted Class: **{class_names[pred_idx]}**")
    st.write(f" Confidence: `{confidence:.2f}%`")


# In[9]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess test image
img_path = "test_leaf.jpg"  # update with your image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Set input and run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# Get prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data[0])
confidence = output_data[0][predicted_class]

print(f"Predicted class index: {predicted_class}")
print(f"Confidence: {confidence:.2f}")


# In[17]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Full path to test image (update filename if needed)
img_path = r"C:\Users\Ujwal M L\OneDrive\Documents\Internship\Hackathon\sample_leaf.jpg"


# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Set model input
interpreter.set_tensor(input_details[0]['index'], img_array)

# Run inference
interpreter.invoke()

# Get prediction result
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output_data[0])
confidence = output_data[0][predicted_index]

# Class names (match your training classes exactly)
class_names = ["train", "val"]  # 👈 Replace with actual class names used in training

# Show result
print(f" Predicted Class: {class_names[predicted_index]}")
print(f" Confidence: {confidence * 100:.2f}%")


# In[20]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Path to your test image — replace the filename with your actual image
img_path = r"C:\Users\Ujwal M L\OneDrive\Documents\Internship\Hackathon\sample_leaf.jpg.JPG"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

# Set input tensor and run inference
interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()

# Get prediction
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_index = np.argmax(output_data[0])
confidence = output_data[0][predicted_index]

# ✅ Replace with your actual class names from training
class_names = ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot"]  # update if needed

# Print result
print(f" Predicted Class: {class_names[predicted_index]}")
print(f" Confidence: {confidence * 100:.2f}%")


# In[21]:


import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names from training
class_names = ["Pepper__bell___healthy", "Pepper__bell___Bacterial_spot"]  # update as needed

st.title(" Plant Disease Classifier (TFLite)")
st.caption("Upload a leaf image to classify the disease.")

uploaded_file = st.file_uploader(" Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred_idx = np.argmax(output[0])
    confidence = output[0][pred_idx]

    st.success(f" Predicted: **{class_names[pred_idx]}**")
    st.write(f" Confidence: `{confidence * 100:.2f}%`")


# In[ ]:




