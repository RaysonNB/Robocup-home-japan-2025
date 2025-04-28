from flask import Flask, jsonify, request, send_from_directory
import os
from werkzeug.utils import secure_filename
#dates
#catkinws/src/begginner/turorial
#sever
#gemini code

app = Flask(__name__)

questions = {
    "Question1": "Question1",
    "Question2": "Question2",
    "Question3": "Question3",
    "Steps" : "-1",
    "Voice" : "Voice",
    "Questionasking":"None",
    "answer":"None"
}

@app.route("/Fambot", methods=['GET', 'POST'])
def handle_questions():
    if request.method == 'GET':
        return jsonify(questions)
    elif request.method == 'POST':
        data = request.get_json()

        # Update existing questions with new values
        for key in data:
            if key in questions:
                questions[key] = data[key]

        return jsonify(questions), 200

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Return the URL where the image can be accessed
        return jsonify({
            "message": "Image uploaded successfully",
            "filename": filename,
            "url": f"/uploads/{filename}"  # This is the GET path
        }), 200
    else:
        return jsonify({"error": "Invalid file type"}), 400


# Add this endpoint to allow image downloads
@app.route('/uploads/<filename>', methods=['GET'])
def get_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
