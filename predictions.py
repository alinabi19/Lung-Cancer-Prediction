import pickle
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def custom_image_tf(image_path, model):
    class_names = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']
    img = image.load_img(image_path, target_size=(224, 224))

    # Convert the image to a numpy array
    img = image.img_to_array(img)

    # Add an extra dimension to the array to match the shape expected by the model
    img = np.expand_dims(img, axis=0)

    # Normalize the image data
    img = img / 255.0

    # Use the model to predict the class of the image
    prediction = model.predict(img)[0]

    # Get the index of the highest probability class
    confidence = round(prediction.max() * 100, 2)
    class_index = np.argmax(prediction)

    # Print the predicted class
    pred_class = class_names[class_index]

    return pred_class, confidence


def custom_image_sk(img_path, model):
    class_names = ['Adenocarcinoma', 'Large cell carcinoma', 'Normal', 'Squamous cell carcinoma']
    # Load the image
    img = load_img(img_path, target_size=(64, 64))

    # Convert the image to a NumPy array
    img_array = img_to_array(img)

    # Reshape the image array to a 2D array
    img_array = img_array.reshape(1, -1)

    # Normalize the image array
    img_array = img_array / 255

    # Predict the confidence score for the image
    confidence = model.predict_proba(img_array)
    pred_class = class_names[confidence.argmax()]

    return pred_class, round(confidence.max() * 100, 2)


# # Test
# CNN_model = keras.models.load_model('models/CNN.h5')
# KNN_model = pickle.load(open('models/KNN.sav', 'rb'))
# SVM_model = pickle.load(open('models/SVM.sav', 'rb'))
# DTC_model = pickle.load(open('models/DTC.sav', 'rb'))
# MLP_model = pickle.load(open('models/MLP.sav', 'rb'))
# Hybrid1_model = pickle.load(open('models/Hybrid1.sav', 'wb'))
# Hybrid2_model = pickle.load(open('models/Hybrid2.sav', 'wb'))
#
# image_path = "testing_input/sccarcinoma2.png"
# class_CNN, confidence_CNN = custom_image_tf(image_path, CNN_model)
# class_SVM, confidence_SVM = custom_image_sk(image_path, SVM_model)
# class_KNN, confidence_KNN = custom_image_sk(image_path, KNN_model)
# class_DTC, confidence_DTC = custom_image_sk(image_path, DTC_model)
# class_MLP, confidence_MLP = custom_image_sk(image_path, MLP_model)
#
# print(class_CNN, confidence_CNN)
# print(class_SVM, confidence_SVM)
# print(class_KNN, confidence_KNN)
# print(class_DTC, confidence_DTC)
# print(class_MLP, confidence_MLP)
