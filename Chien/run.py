from flask import Flask, redirect, url_for, render_template, request, session, flash
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from os import path
import librosa
import tensorflow as tf
import numpy as np
import urllib
from flask import send_file
import urllib.parse

app = Flask(__name__)



@app.route('/')
def home_screen():
    return render_template('home.html')
@app.route('/SER')
def predict_screen():
    return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     # Check if a file is included in the form submission
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']

#     # Check if the file is selected
#     if file.filename == '':
#         flash('No selected file')
#         return redirect(request.url)

#     if file:
#         # Preprocess the file (if needed) and pass it to your model for prediction
#         # result = model.predict(file)

#         # Display the result on the webpage
#         # return render_template('result.html', result=result)
#         return render_template('result.html')
def is_valid_url(url):
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def get_file_content_as_string(path):
    url = 'https://github.com/Ksj14-kumar/Speech-emotion-recognition/blob/main/UI/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")

@app.route('/model')
def load_model():
    model = tf.keras.models.load_model('mymodel.h5')
    return model

def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def predict(model, wav_filepath):
    emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
    test_point = extract_mfcc(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 40, 1))
    predictions = model.predict(test_point)
    return emotions[np.argmax(predictions[0]) + 1]

@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return render_template('index.html', emotion="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', emotion="No selected file")
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            wav_filepath = 'uploads/' + file.filename
            file.save(wav_filepath)
            model = load_model()
            emotion = predict(model, wav_filepath)
            # Đoạn mã HTML để phát file âm thanh
            audio_html = f'<audio controls><source src="{wav_filepath}" type="audio/wav">Trình duyệt không hỗ trợ phát audio.</audio>'
            # Trả về template đã được cập nhật
            return render_template('index.html', emotion=emotion, audio_html=audio_html)
    # url = request.values['url']
    # if is_valid_url(url) and url.endswith('.wav'):
    #     filename = url.split("/")[-1]
    #     urllib.request.urlretrieve(url, 'uploads/' + filename)
    #     wav_filepath = 'uploads/' + filename
    #     model = load_model()
    #     emotion = predict(model, wav_filepath)
    #     return render_template('index.html', emotion=emotion)
    else:
        return render_template('index.html', emotion="Invalid input. Please provide a valid .wav file or URL.")
  
if __name__ == "__main__":
    app.run(debug = True)