from flask import Flask, redirect, url_for, render_template, request, session, flash


app = Flask(__name__)



@app.route('/')
def home_screen():
    return render_template('home.html')
@app.route('/demo')
def predict_screen():
    return render_template('demo.html')




if __name__ == "__main__":
    app.run(debug = True)