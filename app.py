import requests
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

# ==================== 扣子API代理配置 ====================
COZE_BOT_ID = "7629205057501823027"
COZE_API_TOKEN = "pat_CNcyRV6qnMmJ6MfxBMUovl5e8r7lmG7C3EHjYnwWb9bonEetkc8mWcHkr5Cf0K0R"
COZE_API_URL = "https://api.coze.cn/open_api/v2/chat"

# ==================== 图片上传与R计算接口 ====================
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

# ==================== 扣子API代理接口（苏格拉底式对话） ====================
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': '没有消息内容'}), 400

    headers = {
        'Authorization': f'Bearer {COZE_API_TOKEN}',
        'Content-Type': 'application/json'
    }
    payload = {
        "bot_id": COZE_BOT_ID,
        "user": "student_001",
        "query": user_message,
        "stream": False
    }

    try:
        resp = requests.post(COZE_API_URL, headers=headers, json=payload, timeout=30)
        resp_data = resp.json()
        if resp_data.get('code') == 0:
            messages = resp_data.get('messages', [])
            bot_reply = next((m['content'] for m in messages if m['role'] == 'assistant'), "嗯，这个问题有点意思...")
            return jsonify({'reply': bot_reply})
        else:
            return jsonify({'error': resp_data.get('msg', '扣子API出错')}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ==================== 启动入口 ====================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
