import streamlit as st
import os
import tensorflow
from tensorflow import keras
import re


# Define the custom CSS to inject into the Streamlit app

# Set up the title of the 
title = '<p style="font-family:serif; color:#CD6155; font-size: 42px; text-align: center; font-weight: bold;">SKINDER</p>'
st.markdown(title, unsafe_allow_html=True)
subheader = '<p style="font-family:serif; color:#CD6155; font-size: 30px; text-align: center;">Find Your Match Today!</p>'
st.markdown(subheader, unsafe_allow_html=True)


# File uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image to a folder in the specified path with a specific name
    save_path = 'photo'
    file_name = 'uploadedphoto.jpg'
    
    # Ensure the folder exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Write the uploaded image to the specified file
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"Image saved as {file_name} in the folder {save_path}")

from PIL import Image
import numpy as np

# Replace 'path_to_image.jpg' with your image file path
image_path = 'uploadedphoto.jpg'
image = Image.open(image_path)

# Resize the image to 28x28 pixels
image = image.resize((28, 28))

# Convert the image to an array
image_array = np.array(image)

# Ensure the image has 3 channels (RGB)
if len(image_array.shape) == 2:  # It's a grayscale image
    image_array = np.stack((image_array,) * 3, axis=-1)
elif image_array.shape[2] == 4:  # It's an image with an alpha channel
    image_array = image_array[:, :, :3]

# Flatten the array and scale pixel values to [0,1] if required
flat_image_array = image_array.flatten() / 255.0  # Only do this if your model was trained on data scaled in this way

# Now, flat_image_array is ready to be input into your model.
# model.predict(np.array([flat_image_array])) would be your next step,
# where 'model' is your trained machine learning model.

sample = image_array.reshape(-1, 28, 28, 3)

# model loaded
from tensorflow.keras.models import load_model
model_path = 'Skin_Cancer.h5'
model = load_model(model_path)

sample_pred = model.predict(sample)

predicted_label = np.argmax(sample_pred , axis=1)

number_to_string = {
    '4':'melanocytic nevi',
    '6':'melanoma',
    '2':'benign keratosis-like lesions', 
    '1':' basal cell carcinoma',
    '5':' pyogenic granulomas and hemorrhage',
    '0':'Actinic keratoses and intraepithelial carcinomae',
    '3':'dermatofibroma'
}

def replace_numbers(text, mapping):
    # Use regex to find all numbers
    return re.sub(r'\d+', lambda x: mapping.get(x.group(), x.group()), text)

new_text = replace_numbers(predicted_label, number_to_string)

st.write(new_text)

