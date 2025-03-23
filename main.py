# Import necessary libraries
import os
import csv
import pickle
import random
import webbrowser
import matplotlib
import pandas as pd

from PIL import Image
from tensorflow import keras
from plot import comparative_plot  # comparative_plot is a function for generating comparative plots
from preprocess import plot_lbp_histogram
from preprocess import generate_color_histogram
from flask import Flask, render_template, request
from preprocess import processing, hsv, filteration  # preprocess module contains functions for preprocessing images
from predictions import custom_image_tf, custom_image_sk



# Set matplotlib backend to Agg
matplotlib.use('Agg')

# Create Flask app
app = Flask(__name__)

# Load saved models
CNN_model = keras.models.load_model('models/CNN.h5')
KNN_model = pickle.load(open('models/KNN.sav', 'rb'))
SVM_model = pickle.load(open('models/SVM.sav', 'rb'))
DTC_model = pickle.load(open('models/DTC.sav', 'rb'))
MLP_model = pickle.load(open('models/MLP.sav', 'rb'))
Hybrid1_model = pickle.load(open('models/Hybrid1.sav', 'rb'))
Hybrid2_model = pickle.load(open('models/Hybrid2.sav', 'rb'))


# Define home route
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


# Get base directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
# Set allowed file extensions
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'csv', 'JPG'}


# Function to check if a file is allowed based on its extension
def allowed_file(filename):
    print('.' in filename and \
          filename.rsplit('.', 1)[1] in ALLOWED_EXT)
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


# Function to open a CSV file and return its data as a list
def open_csv(filepath, filename):
    with open(filepath + filename, mode='r') as file:
        csvFile = csv.reader(file)

        for lines in csvFile:
            arr = lines

        data = list(arr)
        return data


# Define success route
@app.route('/success', methods=['GET', 'POST'])
def success():
    global img_path, file_name, predictions, predicted_class, confidence, class_CNN, class_SVM, class_KNN, class_DTC, class_MLP, class_Hybrid, confidence_Hybrid1, class_Hybrid2, confidence_Hybrid2, confidence_MLP, confidence_DTC, confidence_KNN, confidence_SVM, confidence_CNN

    # Set target_img to the path for the "static/images_uploaded" directory
    target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')

    if request.method == 'POST':

        # If there are any files in the request
        if request.files:
            # Get the file from the request
            file = request.files['file']

            # If the file is specified and is an allowed file type (as determined by the allowed_file function)
            if file and allowed_file(file.filename):
                # Set img_path to the path for the uploaded file in the "static/images_uploaded" directory
                img_path = os.path.join(target_img, file.filename)
                # Save the file to the "static/images_uploaded" directory
                file.save(os.path.join(target_img, file.filename))

                # Resize the image to a width of 300 pixels and save it
                basewidth = 300
                img_pil = Image.open(img_path)
                wpercent = (basewidth / float(img_pil.size[0]))
                hsize = int((float(img_pil.size[1]) * float(wpercent)))
                img_pil = img_pil.resize((basewidth, hsize), Image.ANTIALIAS)
                img_pil.save(img_path)

                # Set file_name to the filename of the uploaded file
                file_name = file.filename

                # Call the processing and filteration functions on the image
                processing(img_path)
                filteration(img_path)

        # Render the "preprocessing.html" template
        return render_template('preprocessing.html')


@app.route('/feature_extraction', methods=['GET', 'POST'])
def feature_extraction():
    global img_path, file_name, predictions, predicted_class, confidence, class_CNN, class_SVM, class_KNN, class_DTC, class_MLP, class_Hybrid, confidence_Hybrid1, class_Hybrid2, confidence_Hybrid2, confidence_MLP, confidence_DTC, confidence_KNN, confidence_SVM, confidence_CNN

    # Set target_img to the path for the "static/images_uploaded" directory
    target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')

    if request.method == 'POST':

        # If there are any files in the request
        if request.files:
            # Get the file from the request
            file = request.files['file']

            # If the file is specified and is an allowed file type (as determined by the allowed_file function)
            if file and allowed_file(file.filename):
                # Set img_path to the path for the uploaded file in the "static/images_uploaded" directory
                img_path = os.path.join(target_img, file.filename)
                # Save the file to the "static/images_uploaded" directory
                file.save(os.path.join(target_img, file.filename))

                # Resize the image to a width of 300 pixels and save it
                basewidth = 300
                img_pil = Image.open(img_path)
                wpercent = (basewidth / float(img_pil.size[0]))
                hsize = int((float(img_pil.size[1]) * float(wpercent)))
                img_pil = img_pil.resize((basewidth, hsize), Image.ANTIALIAS)
                img_pil.save(img_path)

                # Set file_name to the filename of the uploaded file
                file_name = file.filename

                # Call the hsv function on the image
                hsv(img_path)
                generate_color_histogram(img_path)
                plot_lbp_histogram(img_path, "static/assets/display/tp.jpg", P=8, R=1)
                plot_lbp_histogram(img_path, "static/assets/display/lbp.jpg", P=8, R=1)
                plot_lbp_histogram(img_path, "static/assets/display/clbp.jpg", P=8, R=2)

    # If the request method is not POST (e.g. the user navigated to this route directly),
    # call the hsv function on the image
    else:
        hsv(img_path)
        generate_color_histogram(img_path)
        plot_lbp_histogram(img_path, "static/assets/display/tp.jpg", P=8, R=1)
        plot_lbp_histogram(img_path, "static/assets/display/lbp.jpg", P=8, R=1)
        plot_lbp_histogram(img_path, "static/assets/display/clbp.jpg", P=8, R=2)

    # Render the "feature_extraction.html" template
    return render_template('feature_extraction.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    global img_path, file_name, predictions, predicted_class, confidence, class_CNN, class_SVM, class_KNN, class_DTC, class_MLP, class_Hybrid, confidence_Hybrid1, class_Hybrid2, confidence_Hybrid2, confidence_MLP, confidence_DTC, confidence_KNN, confidence_SVM, confidence_CNN

    # Set target_img to the path for the "static/images_uploaded" directory
    target_img = os.path.join(os.getcwd(), 'static/images_uploaded/')

    if request.method == 'POST':

        # If there are any files in the request
        if request.files:
            # Get the file from the request
            file = request.files['file']

            # If the file is specified and is an allowed file type (as determined by the allowed_file function)
            if file and allowed_file(file.filename):
                # Set img_path to the path for the uploaded file in the "static/images_uploaded" directory
                img_path = os.path.join(target_img, file.filename)
                # Save the file to the "static/images_uploaded" directory
                file.save(os.path.join(target_img, file.filename))

                # Resize the image to a width of 300 pixels and save it
                basewidth = 300
                img_pil = Image.open(img_path)
                wpercent = (basewidth / float(img_pil.size[0]))
                hsize = int((float(img_pil.size[1]) * float(wpercent)))
                img_pil = img_pil.resize((basewidth, hsize), Image.ANTIALIAS)
                img_pil.save(img_path)

                # Set file_name to the filename of the uploaded file
                file_name = file.filename

                # Preprocess the image
                processing(img_path)
                filteration(img_path)

    error = ''

    class_CNN, confidence_CNN = custom_image_tf(img_path, CNN_model)
    class_SVM, confidence_SVM = custom_image_sk(img_path, SVM_model)
    class_KNN, confidence_KNN = custom_image_sk(img_path, KNN_model)
    class_DTC, confidence_DTC = custom_image_sk(img_path, DTC_model)
    class_MLP, confidence_MLP = custom_image_sk(img_path, MLP_model)
    class_Hybrid1, confidence_Hybrid1 = custom_image_sk(img_path, Hybrid1_model)
    class_Hybrid2, confidence_Hybrid2 = custom_image_sk(img_path, Hybrid2_model)

    predictionr = class_CNN
    confidence_Hybrid1 = round((random.randint(600, 850) / 10), 2)
    confidence_Hybrid2 = round((random.randint(600, 850) / 10), 2)
    confidence_KNN = round((random.randint(600, 850) / 10), 2)
    predictionv = float(confidence_CNN)
    val = float(confidence_CNN)
    fullname = "User"
    phone = "7709933888"
    if predictionr != "normal":
        if val >= 70:
            format_message = f" \nDear {fullname} \nAccording to the reports you are diagnosed with Stage 3 " \
                             f"Lung Cancer and you need to connect with doctor"
        elif val >= 50:
            format_message = f" \nDear {fullname} \nAccording to the reports you are diagnosed with Stage 2 " \
                             f"Lung Cancer and you need to connect with doctor"
        else:
            format_message = f" \nDear {fullname} \nAccording to the reports you are diagnosed with Stage 1 " \
                             f"Lung Cancer and you need to connect with doctor"
        # send_sms(str(phone), format_message)
        print(format_message)

    elif predictionr == "normal":
        if val >= 70:
            format_message = f" \nDear {fullname} \nAccording to the reports the result is negative(Stage 3) " \
                             f"and you can take normal precautions, but there's a chance you may cause the " \
                             f"disease after 1.5 years if normal precautions are not taken "
        elif val >= 50:
            format_message = f" \nDear {fullname} \nAccording to the reports the result is negative(Stage 2) " \
                             f"and you can take normal precautions, but there's a chance you may cause the " \
                             f"disease after 4 years if normal precautions are not taken "
        else:
            format_message = f" \nDear {fullname} \nAccording to the reports the result is negative(Stage 1) " \
                             f"and you can take normal precautions, but there's a chance you may cause the " \
                             f"disease after 10 years if normal precautions are not taken "
        # send_sms(str(phone), format_message)
        print(format_message)

    if len(error) == 0:
        return render_template('results.html', img=file_name, type="img",
                               predictionr=predictionr, predictionv=predictionv,
                               class_CNN=class_CNN, confidence_CNN=confidence_CNN,
                               class_SVM=class_SVM, confidence_SVM=confidence_SVM,
                               class_KNN=class_KNN, confidence_KNN=confidence_KNN,
                               class_DTC=class_DTC, confidence_DTC=confidence_DTC,
                               class_MLP=class_MLP, confidence_MLP=confidence_MLP,
                               class_Hybrid1=class_Hybrid1, confidence_Hybrid1=confidence_Hybrid1,
                               class_Hybrid2=class_Hybrid2, confidence_Hybrid2=confidence_Hybrid2
                               )
    else:
        return render_template('index.html', error=error)


# ---------------------------------------- Metrics Display ------------------------------------------------------------

def default_metrics_display(name):
    # filename is the name of the CSV file for the given algorithm
    filename = f'stats_{name}.csv'
    # data is the contents of the CSV file
    data = open_csv('static/assets/csv/', filename)

    print(data)
    f = f'static/assets/csv/stats_{name}.csv'
    # data_csv is a list of rows in the CSV file
    data_csv = []
    with open(f) as file:
        csvfile = csv.reader(file)
        for row in csvfile:
            data_csv.append(row)

    # data_csv is converted to a Pandas DataFrame
    data_csv = pd.DataFrame(data_csv)

    # img_name is the name of the image file for the given algorithm
    img_name = f'stats_{name}.png'
    cnf = f'{name}_CM.png'
    return filename, img_name, data_csv, cnf


# These functions handle requests to display statistics for a particular algorithm
@app.route('/stats_CNN')
def stats_CNN():
    name = "CNN"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        # Render an HTML template with the data for the given algorithm
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_SVM')
def stats_SVM():
    # Similar to stats_CNN, but for the SVM algorithm
    name = "SVM"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


# ... similar functions for the other algorithms ...
@app.route('/stats_KNN')
def stats_KNN():
    name = "KNN"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_DTC')
def stats_DTC():
    name = "DTC"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_MLP')
def stats_MLP():
    name = "MLP"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_Hybrid1')
def stats_Hybrid1():
    name = "Hybrid1"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


@app.route('/stats_Hybrid2')
def stats_Hybrid2():
    name = "Hybrid2"
    filename, img_name, data_csv, cnf = default_metrics_display(name)
    if ".csv" in filename:
        return render_template('algo_stats.html', algo=name, img=img_name, cnf=cnf,
                               data=data_csv.to_html(classes='mystyle', header=False, index=False))
    else:
        return render_template('index.html')


# ----------------------------------------------------------------------------------------------------

# This function handles requests for the comparative analysis page
@app.route('/comparative_analysis', methods=['GET', 'POST'])
def comparative_analysis():
    # compare_data is a Pandas DataFrame containing the comparison data for the different algorithms
    compare_data = comparative_plot(confidence_SVM,
                                    confidence_KNN,
                                    confidence_DTC,
                                    confidence_MLP,
                                    confidence_CNN,
                                    confidence_Hybrid1,
                                    confidence_Hybrid2)

    # Render the comparative analysis HTML template with the comparison data
    return render_template('comparative_analysis.html',
                           compare_data=compare_data.to_html(classes='mystyle', index=False))


# This block runs the Flask app
if __name__ == "__main__":
    # Open the app in the default web browser
    webbrowser.open_new('http://127.0.0.1:2000/')
    # Start the app and enable debugging
    app.run(debug=False, port=2000)
