# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 19:21:52 2019

@author: sl
"""

from keras.applications import ResNet50, imagenet_utils
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from flask import Flask, flash, url_for, request, render_template, redirect, send_from_directory
# from flask.ext.session import Session
from werkzeug import secure_filename
import os
import io
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential
import cv2
import tensorflow as tf

UPLOAD = r'static/img'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# initialize Flask application
app = Flask(__name__)
# sess = Session()
app.config['UPLOAD'] = UPLOAD
model = None
dog_model, pretrained_model, graph = None, None, None

DOG_NAMES = ['Affenpinscher',
 'Afghan_hound',
 'Airedale_terrier',
 'Akita',
 'Alaskan_malamute',
 'American_eskimo_dog',
 'American_foxhound',
 'American_staffordshire_terrier',
 'American_water_spaniel',
 'Anatolian_shepherd_dog',
 'Australian_cattle_dog',
 'Australian_shepherd',
 'Australian_terrier',
 'Basenji',
 'Basset_hound',
 'Beagle',
 'Bearded_collie',
 'Beauceron',
 'Bedlington_terrier',
 'Belgian_malinois',
 'Belgian_sheepdog',
 'Belgian_tervuren',
 'Bernese_mountain_dog',
 'Bichon_frise',
 'Black_and_tan_coonhound',
 'Black_russian_terrier',
 'Bloodhound',
 'Bluetick_coonhound',
 'Border_collie',
 'Border_terrier',
 'Borzoi',
 'Boston_terrier',
 'Bouvier_des_flandres',
 'Boxer',
 'Boykin_spaniel',
 'Briard',
 'Brittany',
 'Brussels_griffon',
 'Bull_terrier',
 'Bulldog',
 'Bullmastiff',
 'Cairn_terrier',
 'Canaan_dog',
 'Cane_corso',
 'Cardigan_welsh_corgi',
 'Cavalier_king_charles_spaniel',
 'Chesapeake_bay_retriever',
 'Chihuahua',
 'Chinese_crested',
 'Chinese_shar-pei',
 'Chow_chow',
 'Clumber_spaniel',
 'Cocker_spaniel',
 'Collie',
 'Curly-coated_retriever',
 'Dachshund',
 'Dalmatian',
 'Dandie_dinmont_terrier',
 'Doberman_pinscher',
 'Dogue_de_bordeaux',
 'English_cocker_spaniel',
 'English_setter',
 'English_springer_spaniel',
 'English_toy_spaniel',
 'Entlebucher_mountain_dog',
 'Field_spaniel',
 'Finnish_spitz',
 'Flat-coated_retriever',
 'French_bulldog',
 'German_pinscher',
 'German_shepherd_dog',
 'German_shorthaired_pointer',
 'German_wirehaired_pointer',
 'Giant_schnauzer',
 'Glen_of_imaal_terrier',
 'Golden_retriever',
 'Gordon_setter',
 'Great_dane',
 'Great_pyrenees',
 'Greater_swiss_mountain_dog',
 'Greyhound',
 'Havanese',
 'Ibizan_hound',
 'Icelandic_sheepdog',
 'Irish_red_and_white_setter',
 'Irish_setter',
 'Irish_terrier',
 'Irish_water_spaniel',
 'Irish_wolfhound',
 'Italian_greyhound',
 'Japanese_chin',
 'Keeshond',
 'Kerry_blue_terrier',
 'Komondor',
 'Kuvasz',
 'Labrador_retriever',
 'Lakeland_terrier',
 'Leonberger',
 'Lhasa_apso',
 'Lowchen',
 'Maltese',
 'Manchester_terrier',
 'Mastiff',
 'Miniature_schnauzer',
 'Neapolitan_mastiff',
 'Newfoundland',
 'Norfolk_terrier',
 'Norwegian_buhund',
 'Norwegian_elkhound',
 'Norwegian_lundehund',
 'Norwich_terrier',
 'Nova_scotia_duck_tolling_retriever',
 'Old_english_sheepdog',
 'Otterhound',
 'Papillon',
 'Parson_russell_terrier',
 'Pekingese',
 'Pembroke_welsh_corgi',
 'Petit_basset_griffon_vendeen',
 'Pharaoh_hound',
 'Plott',
 'Pointer',
 'Pomeranian',
 'Poodle',
 'Portuguese_water_dog'
 'Saint_bernard',
 'Silky_terrier',
 'Smooth_fox_terrier',
 'Tibetan_mastiff',
 'Welsh_springer_spaniel',
 'Wirehaired_pointing_griffon',
 'Xoloitzcuintli',
 'Yorkshire_terrier']

def image_location(image):
    """
    Store uploaded image in default
    directory
    Args image - an image filename
    """
    return os.path.join(app.config['UPLOAD'], image) 

def allowed_file(filename):
    """
    Ensure only selected image filename extensions
    are permissible
    Args filename - an image filename
    """
    return '.' in filename and \
            filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS

def load_model():
    """
    Load pretrained model
    """
    global pretrained_model, dog_model
    dog_model = ResNet50(weights='imagenet')
    pretrained_model = ResNet50(weights='imagenet', include_top=False)
    # pretrained_model._make_predict_function()
    # workaround to fix error
    # ValueError: Tensor Tensor("dense_2/Softmax:0", shape=(?, 133), dtype=float32) is not an element of this graph.
    # use the same graph when making prediction
    # https://github.com/tensorflow/tensorflow/issues/14356
    global graph
    graph = tf.get_default_graph()
 
def load_trained_model(path):
     """
     Load user defined model with pretrained weights
     Args: path - a directory path to the saved checkpoint
     """
     global model
     model = Sequential()
     # shape is a tuple from the penultimate fc layer of resnet50
     model.add(Flatten(input_shape=(1, 1, 2048)))
     model.add(Dense(2048, activation='relu'))
     model.add(Dropout(0.85))
     model.add(Dense(133, activation='softmax'))
     model.load_weights(path)

def prepare_image(image, size):
    """
    Preprocess the input image
    Args: image - input image to be preprocessed
          size - int specifying the target size
    Returns a preprocessed image      
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(size)    
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)
    
    return image

def face_detector(img_path):
    """
    Detect whether the input image has a human
    face in it or not
    Args: img_path - a path to an input image
    """
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
#    img = cv2.imread(img_path)
    img = np.round(np.abs(np.array(img_path)[:, :, ::-1]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_detector(img_path):
    """
    Detect whether the input image has a dog
    in it or not
    Args: img_path - a path to an input image
    """
    with graph.as_default():
        prediction = np.argmax(dog_model.predict(img_path))
    return ((prediction <= 268) & (prediction >= 151))    

def predict_breed(path):
    """
    Predict the breed of the dog
    Args: path - a directory path to the input image
    Returns - the predicted dog breed
    """
    with graph.as_default():
        bottleneck_features = pretrained_model.predict(path)
        predictions = model.predict(bottleneck_features)
    return DOG_NAMES[np.argmax(predictions)]

@app.route("/", methods=['GET', 'POST'])
def index():
    """
    Uploads the image file to default location
    and if valid redirects the user to the predict
    url
    """
    if request.method == 'POST':
        # check if the post request has the file part
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url) 
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(image_location(filename))
            return redirect(url_for('predict', img=filename))
        else:    
            flash('Invalid extension')
            return redirect(request.url) 
    return render_template('index.html')

                          
@app.route("/predict/<img>")
def predict(img):
    """
    Predict the breed of the dog

    Returns - the predicted dog breed
    """
    data = {"success": False}
    category = "Dog"
    
    if request.method == 'GET':
        image = Image.open(image_location(img))
        faces = face_detector(image)
        image = prepare_image(image, size=(224, 224))
        dog = dog_detector(image)
        
        if faces:
            breed = predict_breed(image)
            category = "Human"
        else:
            if dog:    
                breed = predict_breed(image)
            else:
                category = "Other"
                breed = "Unknown"
            
        data["predictions"] = None
        data["predictions"]= {"category": category, "breed": breed}
        # data["predictions"].append(
        #     {"category": category,
        #     "breed": breed}
        #     )
        data["success"] = True  

    # os.path.join('img', img) doesn't work on windows as it produces a backward slash
    # instead of forward slash    
    return render_template('predict.html', category=category, data=data, 
        img_file='img/' + img, img_ref='img/ref/' + breed + '.jpg') if category != 'Unknown' else render_template('predict.html', category=category, data=data, 
        img_file='img/' + img, img_ref='img/ref/unknown.png')

if __name__ == '__main__':
    print("Loading pretrained model to extract bottleneck features...")
    load_model()
    print("Load user defined model with pretrained weights...")
    load_trained_model('saved_weights/weights.best.Resnet50.hdf5')
    # app.secret_key = 'secret key'
    # app.config['SESSION_TYPE'] = 'filesystem'

    # sess.init_app(app)
    app.run()                    