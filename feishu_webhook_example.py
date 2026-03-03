
# 飞书Webhook验证示例 (Python Flask)

from flask import Flask, request, jsonify
import hashlib
import json

app = Flask(__name__)

# 配置信息
VERIFICATION_TOKEN = "your_verification_token_10aec1e89ce6b99a"

@app.route('/api/feishu/webhook', methods=['POST'])
def feishu_webhook():
    # 验证签名
    timestamp = request.headers.get('X-Lark-Request-Timestamp', '')
    nonce = request.headers.get('X-Lark-Request-Nonce', '')
    signature = request.headers.get('X-Lark-Signature', '')

    # 计算签名
    string_to_sign = f"{timestamp}{nonce}{VERIFICATION_TOKEN}"
    expected_signature = hashlib.sha256(string_to_sign.encode()).hexdigest()

    if expected_signature != signature:
        return jsonify({"error": "Invalid signature"}), 403

    # 处理事件
    data = request.json

    if data.get("type") == "url_verification":
        # URL验证
        return jsonify({"challenge": data.get("challenge")})

    # 处理其他事件...
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(port=5656, debug=True)
