from flask import Flask ,redirect, url_for, render_template, request
import os

app = Flask(__name__)

# @app.route('/index') 
# def hello_world():
#     return "<h1>Chien</h1>"

# @app.route('/Blog/<int:blog_id>') 
# def hello_blog(blog_id):
#     return f"<h1> Blog {blog_id}!</h1>"

# @app.route('/user/<name>') 
# def hello_user(name):
#     if name == 'admin':
#         return redirect(url_for('hello_world'))
#     return f"<h1> Hello {name}!</h1>"

@app.route('/') 
def index():
    return render_template("index.html")
@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        file_path = os.path.join('uploads', uploaded_file.filename)
        #uploaded_file.save(file_path)
        # Xử lý file âm thanh và triết xuất kết quả từ mô hình máy học ở đây
        # Sau đó trả kết quả cho người dùng
        return "result: ..."
    return "Không có file được tải lên."

if __name__ == "__main__":
    app.run(debug = True)