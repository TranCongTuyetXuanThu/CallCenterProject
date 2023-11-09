from flask import Flask, redirect, url_for, render_template, request, session, flash, send_from_directory, jsonify, Response
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from os import path
import librosa
import tensorflow as tf
import numpy as np
import urllib
from flask import send_file
import urllib.parse
import subprocess
from torch import nn
import torch, torchaudio, statistics
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os

# Model flow
class define_model(nn.Module):

    # Define layers
    def __init__(self, num_emotions):
        super().__init__()

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=40, #####################
            nhead=4,
            dim_feedforward=512,
            dropout=0.4,
            activation='relu'
        )
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=4)

        #maxpool: reshape (width, height)
        #conv: reshape (channel)
        conv2d_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),#######
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),#######
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),#######
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),#######
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(32),#######
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(64),#######
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        self.conv2Dblock2 = conv2d_layer

        self.fc1_layer = nn.Linear(960*2+40, 980)
        self.act1 = nn.ReLU()
        self.fc2_layer = nn.Linear(980, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim = 1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim = 1)

        x_maxpool = self.transformer_maxpool(x)
        x_maxpool_reduced = torch.squeeze(x_maxpool,1) ############
        x = x_maxpool_reduced.permute(2,0,1) ###########
        transformer_output = self.transformer_encoder(x)
        transformer_embedding = torch.mean(transformer_output, dim = 0)

        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2, transformer_embedding], dim = 1)
        fc1 = self.fc1_layer(complete_embedding)
        ac1 = self.act1(fc1)
        output_logits = self.fc2_layer(ac1)
        output_softmax = self.softmax_out(output_logits)
        return output_logits, output_softmax

class preprocess_real_data():
    def __init__(self,file):
      self.file = file
    def mp4_to_wav(self, file):
        # Load the MP4 file
        # video = VideoFileClip(file)
        # # Extract the audio from the video
        # audio = video.audio
        # # Export the audio as a WAV file
        # file = file[:-4]+'.wav'
        # audio.write_audiofile(file)
        output_file = file.split('.')[0] + ".wav"
        subprocess.call(['ffmpeg', '-i', file, output_file])
        return output_file
    def remove_noise(self, file):
        # Detect non-silent parts of the audio
        sound_file = AudioSegment.from_wav(file)
        non_sil_times = detect_nonsilent(sound_file, min_silence_len=400, silence_thresh=sound_file.dBFS * 0.65)

        # Concatenate the non-silent parts of the audio
        if len(non_sil_times) > 0:
            non_sil_times_concat = [non_sil_times[0]]
            if len(non_sil_times) > 1:
                for t in non_sil_times[1:]:
                    if t[0] - non_sil_times_concat[-1][1] < 100:
                        non_sil_times_concat[-1] = (non_sil_times_concat[-1][0], t[1])
                    else:
                        non_sil_times_concat.append(t)
            new_audio = sound_file[non_sil_times_concat[0][0]:non_sil_times_concat[0][1]]
            for t in non_sil_times_concat[1:]:
                new_audio += sound_file[t[0]:t[1]]
        else:
            new_audio = sound_file

        # Export the new audio file
        file_name = file[:-4]+'_denoised.wav'
        new_audio.export(file[:-4]+'_denoised.wav', format="wav")
        return file_name
    def resize(self, file):
        tensor = torch.load(file)
        tensor1 = tensor[0,:,:]
        # tensor2 = tensor[1,:,:]
        n = tensor1.shape[-1]//500 + 1
        sample1 = torch.zeros((40,n * 500))
        sample1[:,:tensor1.shape[-1]] = tensor1
        sample1 = torch.transpose(torch.transpose(sample1.reshape((40,500,n)),dim0=0, dim1=2),dim0=1, dim1=2).reshape((n,1,40,500))
        return sample1
    def save_tensor_file(self, file):
        waveform, sample_rate = torchaudio.load(file)
        transform = torchaudio.transforms.MFCC(sample_rate=sample_rate)
        mfcc = transform(waveform)
        file_name = file[:-4]+'.pt'
        torch.save(mfcc,file_name)
        return file_name
    def complete_preprocessing(self, file):
      if file[-4:] != '.wav':
          file = self.mp4_to_wav(file)
      file = self.remove_noise(file)
      file = self.save_tensor_file(file)
      tensor = self.resize(file)
      return file, tensor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


@app.route('/')
def home_screen():
    return render_template('home.html')
@app.route('/demo')
def predict_screen():
    return render_template('demo.html')

@app.route('/test', methods=['POST','GET'])
def test_screen():
    audio_file = "../uploads/5.wav"
    
    return render_template('test.html', audio_url = audio_file)

@app.route('/model')
def load_model(model_path):
    model = define_model(3)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    # model.load_state_dict(torch.load(model_path))
    return model

# def extract_mfcc(wav_file_name):
#     y, sr = librosa.load(wav_file_name)
#     mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
#     return mfccs

# def predict(model, wav_filepath):
#     emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}
#     test_point = extract_mfcc(wav_filepath)
#     test_point = np.reshape(test_point, newshape=(1, 40, 1))
#     predictions = model.predict(test_point)
#     return emotions[np.argmax(predictions[0]) + 1]
def predict_1_sample(file_sample, model, preprocesser):
    file_sample, tensor = preprocesser.complete_preprocessing(file_sample)
    output_logit, output_softmax = model(tensor)
    output_softmax = torch.argmax(output_softmax, dim=1)
    final_output = max(set(output_softmax.tolist()), key=output_softmax.tolist().count)
    emotion_dict = {0: 'positive',
                    1: 'neutral',
                    2: 'negative'}
    label = emotion_dict[final_output]
    return final_output, label


@app.route('/upload', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return render_template('demo.html', emotion="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('demo.html', emotion="No selected file")
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            wav_filepath = 'static/uploads/' + file.filename
            file.save(wav_filepath)
            model = load_model('model95.pt')
            preprocess = preprocess_real_data('ok')
            final_output, label = predict_1_sample(wav_filepath, model, preprocess)
            # Trả về template đã được cập nhật
            if label=="positive":
                label="(Positive) Good job!"
            elif label=="negative":
                label= "(Negative) Alert! Your customer is not satified"
            elif label=="neutral":
                label="Neutral"
            return render_template('demo.html', emotion=label, audio = True, file=file.filename)
    # url = request.values['url']
    # if is_valid_url(url) and url.endswith('.wav'):
    #     filename = url.split("/")[-1]
    #     urllib.request.urlretrieve(url, 'uploads/' + filename)
    #     wav_filepath = 'uploads/' + filename
    #     model = load_model()
    #     emotion = predict(model, wav_filepath)
    #     return render_template('index.html', emotion=emotion)
    else:
        return render_template('demo.html', emotion="Invalid input. Please provide a valid .wav file or URL.")


@app.route('/predict', methods=['POST'])
def predict_audio():
    audio_data = request.files['file']
    # Save the received audio data as a WAV file
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_audio.wav')
    audio_data.save(audio_path)

    # Load your model and perform the prediction
    model = load_model('model95.pt')
    preprocess = preprocess_real_data('ok')
    final_output, label = predict_1_sample(audio_path, model, preprocess)
    if label=="positive":
        label="(Positive) Good job!"
    elif label=="negative":
        label= "(Negative) Alert! Your customer is not satified"
    elif label=="neutral":
        label="Neutral" 
    return render_template('demo.html', emotions = label, audio2 = True, file = 'recorded_audio.wav')

if __name__ == "__main__":
    app.run(debug = True)
    audio = False
    audio2 = False