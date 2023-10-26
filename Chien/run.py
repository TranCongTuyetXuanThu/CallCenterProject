from flask import Flask, redirect, url_for, render_template, request, session, flash
from datetime import timedelta
from flask_sqlalchemy import SQLAlchemy
from os import path

app = Flask(__name__)



@app.route('/')
def home_screen():
    return render_template('home.html')
@app.route('/SER')
def predict_screen():
    return render_template('index1.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Check if a file is included in the form submission
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # Check if the file is selected
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Preprocess the file (if needed) and pass it to your model for prediction
        # result = model.predict(file)

        # Display the result on the webpage
        # return render_template('result.html', result=result)
        return render_template('result.html')
    

if __name__ == "__main__":
    app.run(debug = True)