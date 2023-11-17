from flask import Flask, redirect, url_for, render_template, request, session, flash, send_from_directory, jsonify, Response, send_file
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm 
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy  import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from os import path
import subprocess
from torch import nn
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Model flow
import subprocess
import torch, torchaudio
import datetime


# Create Flask app
app = Flask(__name__)

# App configuration settings
app.config['SECRET_KEY'] = 'Daylasecretkey!'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///user.db"

# Ensure UPLOAD_FOLDER exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Flask extensions
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Define User class for SQLAlchemy model
class User(UserMixin, db.Model):
    # Define User model columns
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))
#Define login form
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('Remember me')

#Define RegisterForm
class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])

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
        conv2d_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
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
                in_channels=2,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(16),
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
            nn.BatchNorm2d(32),
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
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
        )
        self.conv2Dblock2 = conv2d_layer

        self.fc1_layer = nn.Linear(1792*2+40, 1800)
        self.act1 = nn.ReLU()
        self.fc2_layer = nn.Linear(1800, num_emotions)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        conv2d_embedding1 = self.conv2Dblock1(x)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim = 1)

        conv2d_embedding2 = self.conv2Dblock2(x)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim = 1)

        x_maxpool = self.transformer_maxpool(x)
        x_maxpool_reduced = x_maxpool.resize(x_maxpool.shape[0], 40, 450) 
        x = x_maxpool_reduced.permute(2,0,1) #rearrange dims
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
        output_file = file.replace('input/dpl-project/test_data/raw_datas', 'working/wav_datas')
        output_file = output_file.split('.')[0] + ".wav"
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
        file_name = file.replace('wav_datas', 'denoised_datas')
        file_name = file_name[:-4]+'_denoised.wav'
        new_audio.export(file_name, format="wav")
        return file_name
    
    def resize(self, file):
        tensor = torch.load(file)
        n = tensor.shape[-1]//900 + 1
        sample = torch.zeros((2,40,n * 900))
        sample[:,:,:tensor.shape[-1]] = tensor
        sample = sample.reshape((2,40,900,n)).permute(3,0,1,2)
        torch.save(sample,file)
        return sample
    
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
    
def waveform(path, save_path):
    waveform, sr = torchaudio.load(path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(waveform.numpy().T)
    ax.set_title('Waveform')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    fig.savefig(save_path, format='png')

def plot_waveform_async(audio_path):
    waveform_image = waveform(audio_path)
    return waveform_image


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if user.password == form.password.data:
                login_user(user, remember=form.remember.data)
                return redirect(url_for('access_demo'))
        flash('Invalid username or password', 'error')  
    return render_template('login.html', form=form)

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        new_user = User(username=form.username.data, email=form.email.data, password=form.password.data)
        db.session.add(new_user)
        db.session.commit()
        return '<h1>New user has been created!</h1>'

    return render_template('signup.html', form=form)
@app.route('/demo')
@login_required
def access_demo():
    return render_template('demo.html', name=current_user.username)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home_screen'))

@app.route('/')
def home_screen():
    return render_template('home.html')
@app.route('/demo')
@login_required
def predict_screen():
    return render_template('demo.html')

@app.route('/model')
def load_model(model_path):
    model = define_model(3)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model

import wave

def get_wav_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        # Get the frame rate (samples per second) and total number of frames
        frame_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()
        # Calculate the duration in seconds
        duration = total_frames / float(frame_rate)
    return duration

def predict_real_time(file_sample, model, preprocesser):
    _, tensor = preprocesser.complete_preprocessing(file_sample)
    output_logit, output_softmax = model(tensor)
    output_softmax = torch.argmax(output_softmax, dim=1)
    emotion_dict = {0: 'positive',
                    1: 'neutral',
                    2: 'negative'}
    labels = [emotion_dict[int(value)] for value in output_softmax]
    
    final_output = max(set(output_softmax.tolist()), key=output_softmax.tolist().count)
    label = emotion_dict[final_output]
    
    duration_seconds = get_wav_duration(file_sample)
    range_ = duration_seconds/ len(labels)
    a = datetime.time(0, 0)
    list_labels = list()
    delta = datetime.timedelta(seconds=round(range_, 5))
    for label in labels:
        new_time = datetime.datetime.combine(datetime.datetime.today(), a) + delta
        list_labels.append(str(a)[3:8]+'-'+str(new_time.time())[3:8]+'_'+str(label))
        a = new_time.time()
    return(list_labels)


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

            model = load_model('model78.pt')
            tmp_dir = "static/tmp"
            waveform_image_path = os.path.join(tmp_dir, "waveform.png")
            preprocess = preprocess_real_data(0)
            waveform(wav_filepath, waveform_image_path)
            list_labels = predict_real_time(wav_filepath, model, preprocess)

            history=[]
            for x in list_labels:
                if 'neutral' in x:
                    y = x.replace('neutral', 'Predicted Emotion: Neutral')
                elif 'positive' in x:
                    y = x.replace('positive', 'Predicted Emotion: Positive')
                elif 'negative' in x:
                    y = x.replace('negative', 'Predicted Emotion: Negative')
                history.append(y)

            print(list_labels)

            for i in range(len(list_labels)):
                if 'neutral' in list_labels[i]:
                    list_labels[i] = list_labels[i].replace('neutral', 'Predicted Emotion: Neutral')
                elif 'positive' in list_labels[i]:
                    list_labels[i] = list_labels[i].replace('positive', 'Predicted Emotion: (Positive) Good job!')
                elif 'negative' in list_labels[i]:
                    list_labels[i] = list_labels[i].replace('negative', 'Predicted Emotion: (Negative) Alert! Your customer is not satified')

            return render_template('demo.html', emotion=list_labels,waveform_image_path="waveform.png", audio = True, history=history, file=file.filename, login_stat = True)
    else:
        return render_template('demo.html', emotion="Invalid input. Please provide a valid .wav file or URL.")

@app.route('/predict', methods=['POST'])
def predict_audio():
    audio_data = request.files['file']
    # Save the received audio data as a WAV file
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_audio.wav')
    audio_data.save(audio_path)
    tmp_dir = "static/tmp"
    waveform_image_path = os.path.join(tmp_dir, "waveform_record.png")
    # Load your model and perform the prediction
    model = load_model('model78.pt')
    preprocess = preprocess_real_data(0)
    waveform(audio_path, waveform_image_path)
    #waveform_image = waveform(audio_path)
    list_labels = predict_real_time(audio_path, model, preprocess)
    histories=[]
    for x in list_labels:
        if 'neutral' in x:
            y = x.replace('neutral', 'Predicted Emotion: Neutral')
        elif 'positive' in x:
            y = x.replace('positive', 'Predicted Emotion: Positive')
        elif 'negative' in x:
            y = x.replace('negative', 'Predicted Emotion: Negative')
        histories.append(y)
    print(list_labels)
    for i in range(len(list_labels)):
        if 'neutral' in list_labels[i]:
            list_labels[i] = list_labels[i].replace('neutral', 'Predicted Emotion: Neutral')
        elif 'positive' in list_labels[i]:
            list_labels[i] = list_labels[i].replace('positive', 'Predicted Emotion: (Positive) Good job!')
        elif 'negative' in list_labels[i]:
            list_labels[i] = list_labels[i].replace('negative', 'Predicted Emotion: (Negative) Alert! Your customer is not satified')
    return render_template('demo.html', emotions = list_labels, waveform_image="waveform_record.png", audio2 = True, histories=histories, file = 'recorded_audio.wav', login_stat = True)

@app.route('/result')
def result_answer():
    return render_template('result.html', login_stat = True)

if __name__ == "__main__":
    with app.app_context():
        if not path.exists("user.db"):
            db.create_all()
            print("Created database")
    model = define_model(3)
    model.eval()
    model.load_state_dict(torch.load("model78.pt", map_location='cpu'))
    app.run(debug=True)
    audio = False
    audio2 = False
    login_stat = False