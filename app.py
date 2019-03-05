from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image_label():
    if request.method == 'POST':
        print (request.files)
    return render_template('index.html')

@app.route('/top_images')
def view_top_images():
    