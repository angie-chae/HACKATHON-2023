# HACKATHON-2023

## Team Info:
We are a group of three female junior students at Manhattan High School, Kansas. Our names are Angie Chae, Sama Nepal, Kiku Nagai-Velasquez.

## Project Motivation
In rural Kansas, healthcare challenges are pronounced and must be addressed. Technology can help overcome issues related to healthcare service availability. Residents often struggle with limited access to medical services and face the burden of traveling long distances. To mitigate these issues, we recognized the necessity of developing an AI-powered mobile application to assist those in need. This app is designed to provide medical assistance for skin diseases by allowing users to upload pictures. It identifies potential diseases from these images and includes a feature to locate dermatologists.

## Project Outline
![image](https://github.com/AngieChae/HACKATHON-2023/assets/149910893/aa5d11a1-ed74-425c-b37f-819e4650f8a7)

App Implementation: Used Python package [Streamlit](https://streamlit.io/). 

App Features inlcude (1) skin disease prediction/detection and (2) locate dermatologists around a zip code.

API Use: [Google Place API](https://developers.google.com/maps/documentation/places/web-service/overview)

Map Implementation: [Pydeck](https://pydeck.gl/)

Data used to train AI/Deep Learning Models: [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) consists of "10015 dermatoscopic images which can serve as a training set for academic machine learning purposes. Cases include a representative collection of all important diagnostic categories in the realm of pigmented lesions: Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec), basal cell carcinoma (bcc), benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl), dermatofibroma (df), melanoma (mel), melanocytic nevi (nv) and vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc)."

AI/Machine Learning: Deep learning models were developed for multiclass skin disease detection. The best performing deep learning model is saved in Skin_Cancer.h5, which is largely benefited from [this project](https://www.kaggle.com/code/hadeerismail/skin-cancer-prediction-cnn-acc-98/notebook) 

## App: Skinder
Our innovative app blends the functionality of skin disease detection with the matchmaking ease of Tinder. Users simply upload a photo of their skin ailment, and the app intelligently predicts the possible skin condition. Simultaneously, it provides the convenience of locating the nearest dermatologist for an expert consultation.

## Our App is located here: https://hackathon-2023-szagipajwqdemwnau33rpv.streamlit.app/
This app is fully functional, and we demonstrate its complete capabilities in our video recording. It utilizes a Google Location API key from one of our team members to find skin disease specialists near a user's location. For security purposes, this API key was removed from the app after recording the video. Consequently, the feature 'locate dermatologists around a zip code' has been disabled for security reasons.

## Description of Each File
Skin_Cancer.h5: This is the trained machine learning model that powers our application's core feature – predicting the type of skin disease from uploaded images.

app.py: This script is the backbone of our application. It includes the functionality for image upload, result display, access to additional information on skin conditions, and a feature to locate the closest dermatologists based on the user's area code.

carcinoma.jpg & melanoma.jpg: These images serve as samples within our app, allowing users to see example outputs and test the app's functionality.

requirements.txt: This document lists all the dependencies and packages required to run our app, ensuring that anyone setting up the app has the necessary environment for it to function correctly
