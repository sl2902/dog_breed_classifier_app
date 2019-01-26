# Dog breed classifier
In this project we build an application that classifies dog breeds - 1 out of 133 different breeds; the model is built in 2 stages: 1) In stage 1, we use a pretrained model as a feature extractor. In stage2, these features as passed as inputs into user defined classifier and, trained; the best weights during training are saved. In addition to classifying the breed of a dog, if the user inputs an image of a human, it will try to classify the breed of the human, otherwise it will inform the user that the input is unclassifiable

## Instructions:
1) The instructions assume you are on Windows and you have Python 3.6 installed

2) Navigate to the directory you want to setup a new virtual environment in
```
python -m virtualenv dog_breed_env
On windows, run
dog_breed_env\Scripts\activate
```

3) Clone the repository
```
git clone https://github.com/sl2902/dog_breed_classifier_app.git
cd dog_breed_classifier_app
```

4) Install the relevant libraries
```
pip install requirements.txt
```

5) Start the app locally, and wait for the script to load the pretrained model and user-defined model weights. You
should see the following - `Running on http://127.0.0.1:5000/`
```
python run_dog_breed_classifier.py
```

6) Launch the app on Chrome
```
http://localhost:5000/, or
http://127.0.0.1:5000/
```

References:</br>
https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html</br>
http://flask.pocoo.org/docs/1.0/patterns/fileuploads/</br>
https://pymote.readthedocs.io/en/latest/install/windows_virtualenv.html</br>
https://getbootstrap.com/docs/3.3/getting-started/</br>
https://pythonhosted.org/Flask-Bootstrap/basic-usage.html#examples
