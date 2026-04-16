import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from compute_R import compute_R_from_image

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': '没有上传图片'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    try:
        R_value = compute_R_from_image(filepath)
        quality_score = 85.0 if R_value > 1000 else 70.0
        message = f"曲率半径 {R_value:.2f} mm，"
        if quality_score > 80:
            message += "图片质量良好。"
        else:
            message += "图片质量一般，建议调整对焦或光线。"
        return jsonify({'radius': round(R_value, 2), 'quality': quality_score, 'message': message})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)