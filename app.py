import streamlit as st
import os
import tensorflow
from tensorflow import keras
import re
import time
import requests
import json
import pandas as pd
import pydeck as pdk

if 'display_columns' not in st.session_state:
    st.session_state.display_columns = False
# Define the custom CSS to inject into the Streamlit app

# Set up the title of the 
title = '<p style="font-family:serif; color:#CD6155; font-size: 60px; text-align: center; font-weight: bold;">SKINDER</p>'
st.markdown(title, unsafe_allow_html=True)
subheader = '<p style="font-family:serif; color:#CD6155; font-size: 30px; text-align: center;">Find Your Match Today!</p>'
st.markdown(subheader, unsafe_allow_html=True)


# File uploader widget
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])


if uploaded_file is not None:
    progress_bar = st.progress(0)
    # Save the uploaded image to a folder in the specified path with a specific name
    save_path = 'photo'
    file_name = 'uploadedphoto.jpg'
    
    # Ensure the folder exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Write the uploaded image to the specified file
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "wb") as f:
        progress_bar.progress(25)
        f.write(uploaded_file.getbuffer())
        progress_bar.progress(50)
        f.write(uploaded_file.getbuffer())
        progress_bar.progress(75)
        f.write(uploaded_file.getbuffer())
        progress_bar.progress(100)
    
    time.sleep(3)
    st.success("Image Successfully Uploaded")

    st.session_state.display_columns = True

from PIL import Image
import numpy as np

# Replace 'path_to_image.jpg' with your image file path
image_path = 'photo/uploadedphoto.jpg'
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

new_text = replace_numbers(str(predicted_label), number_to_string)
match = new_text.replace("[", "").replace("]", "")

displaymatch = '<p style="font-family:serif; color:#CD6155; font-size: 30px; text-align: center;">Your Match is...</p>'
st.markdown(displaymatch, unsafe_allow_html=True)

if st.session_state.display_columns:
    col1, col2 = st.columns(2)

    # First column for words
    with col1:

        # Custom CSS to move text to the right
        custom_css = """
        <style>
        .shifted-text {
            margin-left: 150px;  /* Adjust this value to move the text more or less */
            font-size: 30px;
            font-color: #CC0066;
        }
        </style>
        """

        # Inject custom CSS with st.markdown
        st.markdown(custom_css, unsafe_allow_html=True)

        # Use the custom CSS class in another st.markdown call
        st.markdown(f'<div class="shifted-text">{match}</div>', unsafe_allow_html=True)

    # Second column for image and URL
    with col2:
        if uploaded_file is not None:
            user_uploaded_image = Image.open(uploaded_file)
            st.image(user_uploaded_image, width=200)
        

        if predicted_label == 0:
            url = "https://www.ncbi.nlm.nih.gov/books/NBK557401/"
            st.markdown(f'[More Information on this disease]({url})')
        elif predicted_label == 1:
            url = "https://www.skincancer.org/skin-cancer-information/basal-cell-carcinoma/"
            st.markdown(f'[More Information on this disease]({url})')
        elif predicted_label == 2:
            url = "https://www.mayoclinic.org/diseases-conditions/seborrheic-keratosis/symptoms-causes/syc-20353878"
            st.markdown(f'[More Information on this disease]({url})')
        elif predicted_label == 3:
            url = "https://www.ncbi.nlm.nih.gov/books/NBK470538/#:~:text=Dermatofibroma%20is%20a%20commonly%20occurring,histiocytomas%2C%20or%20common%20fibrous%20histiocytoma."
            st.markdown(f'[More Information on this disease]({url})')
        elif predicted_label == 4:
            url = "https://emedicine.medscape.com/article/1058445-overview"
            st.markdown(f'[More Information on this disease]({url})')
        elif predicted_label == 5:
            url = "https://www.mountsinai.org/health-library/diseases-conditions/pyogenic-granuloma#:~:text=Pyogenic%20granulomas%20are%20skin%20lesions,around%20them%20may%20be%20inflamed."
            st.markdown(f'[More Information on this disease]({url})')
        else:
            url = "https://www.cancer.gov/types/skin/patient/melanoma-treatment-pdq"
            st.markdown(f'[More Information on this disease]({url})')

st.divider()

# You'll need to get an API key from Google Cloud Console and enable the Places API.
GOOGLE_API_KEY = 'AIzaSyAI5_Lklb8yb-mogJLVbrcMCYN2lNZln9U'

def find_skin_doctors(zip_code):
    # Convert the zip code into latitude and longitude
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={zip_code}&key={GOOGLE_API_KEY}"
    response = requests.get(geocode_url)
    location_data = response.json()
    
    if location_data['status'] == 'OK':
        latitude = location_data['results'][0]['geometry']['location']['lat']
        longitude = location_data['results'][0]['geometry']['location']['lng']
        
        # Use the Places API to find dermatologists nearby
        places_url = f"https://maps.googleapis.com/maps/api/place/nearbysearch/json?location={latitude},{longitude}&radius=5000&type=doctor&keyword=dermatologist&key={GOOGLE_API_KEY}"
        places_response = requests.get(places_url)
        places_data = places_response.json()
        
        if places_data['status'] == 'OK':
            places_details = []
            for place in places_data['results']:
                place_info = {
                    'Name': place['name'],
                    'Address': place['vicinity'],
                    'Rating': place.get('rating', 'Not Available'),
                    'Place ID': place['place_id'],
                    'Business Status': place.get('business_status', 'Not Available'),
                    'Open Now': place['opening_hours']['open_now'] if 'opening_hours' in place else 'Not Available',
                    # Additional fields can be conditionally added if they exist.
                    'Phone Number': 'Retrievable via details request',
                    'Website': 'Retrievable via details request',
                    'geometry': place['geometry']
                }
                
                # Add optional fields if available, note that this requires additional API calls
                details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place['place_id']}&fields=formatted_phone_number,website&key={GOOGLE_API_KEY}"
                details_response = requests.get(details_url)
                details_data = details_response.json()
                
                if details_data['status'] == 'OK':
                    result = details_data['result']
                    place_info['Phone Number'] = result.get('formatted_phone_number', 'Not Available')
                    place_info['Website'] = result.get('website', 'Not Available')

                places_details.append(place_info)

            # Create a DataFrame from the details
    df = pd.DataFrame(places_details)

    # Convert latitude and longitude to float and prepare the DataFrame for st.map
    df['lat'] = df.apply(lambda row: row['geometry']['location']['lat'], axis=1)
    df['lon'] = df.apply(lambda row: row['geometry']['location']['lng'], axis=1)
    return df

diseasetitle = '<p style="font-family:serif; color:#CD6155; font-size: 35px; text-align: center;">Find Skin Disease Doctors in Your Area</p>'
st.markdown(diseasetitle, unsafe_allow_html=True)

# User input for the zip code
zip_code = st.text_input('Enter your zip code (example: 60606 for Chicago, IL):', '')

# Button to search for doctors
if st.button('Find Doctors'):
    if zip_code:
        # Call the function to find doctors
        doctors_df = find_skin_doctors(zip_code)

        if not doctors_df.empty:
            # Display the DataFrame in the Streamlit app
            st.dataframe(doctors_df[['Name', 'Address', 'Rating', 'Phone Number', 'Website']])  # Show selected columns in the table

            st.divider()

            # Define a layer to use in pydeck
            layer = pdk.Layer(
                "ScatterplotLayer",
                doctors_df,
                pickable=True,
                opacity=0.8,
                stroked=True,
                filled=True,
                radius_scale=10,
                radius_min_pixels=10,
                radius_max_pixels=100,
                line_width_min_pixels=1,
                get_position='[lon, lat]',
                get_fill_color=[255, 0, 0, 160],
                get_line_color=[0, 0, 0],
            )

            # Set the viewport location
            view_state = pdk.ViewState(
                latitude=doctors_df['lat'].mean(),
                longitude=doctors_df['lon'].mean(),
                zoom=11,
                pitch=50,
            )

            # Render the map with pydeck
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style='mapbox://styles/mapbox/light-v9',
                tooltip={
                    "html": "<b>Name:</b> {Name}<br><b>Address:</b> {Address}<br><b>Rating:</b> {Rating}",
                    "style": {
                        "backgroundColor": "steelblue",
                        "color": "white"
                    }
                }
            )
            st.pydeck_chart(r)           
            
        else:
            st.error('No doctors found or an error occurred. Please try again.')
    else:
        st.error('Please enter a valid zip code.')




