import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, send_from_directory

app = Flask(__name__)

# Thư mục lưu trữ tệp tải lên và thư mục lưu trữ biểu đồ
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST' and 'audio_file' in request.files:
        file = request.files['audio_file']
        if file.filename == '':
            return redirect(request.url)

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        y, sr = librosa.load(file_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.colorbar()
        plt.title('MFCC')
        plt.xlabel('Thời gian')
        plt.ylabel('MFCC Coefficients')

        image_path = os.path.join(app.config['STATIC_FOLDER'], 'mfcc.png')
        plt.savefig(image_path)

        return render_template('index1.html', mfcc_image='mfcc.png')

    return render_template('index1.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug = True)