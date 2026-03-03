import json
import hashlib
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class FeishuBot:
    """飞书机器人集成（简化版）"""

    def __init__(self, app_id: str, app_secret: str, verification_token: str = "", encrypt_key: str = ""):
        self.app_id = app_id
        self.app_secret = app_secret
        self.verification_token = verification_token
        self.encrypt_key = encrypt_key
        logger.info(f"初始化飞书机器人，App ID: {app_id[:6]}...")

    def verify_signature(self, timestamp: str, nonce: str, signature: str) -> bool:
        """验证签名"""
        if not self.verification_token:
            logger.warning("未配置验证令牌，跳过签名验证")
            return True

        # 按字典序排序
        sorted_list = sorted([timestamp, nonce, self.verification_token])
        combined = ''.join(sorted_list)

        # SHA1加密
        sha1 = hashlib.sha1()
        sha1.update(combined.encode('utf-8'))
        calculated = sha1.hexdigest()

        return calculated == signature

    def handle_event(self, data: Dict) -> Optional[Dict]:
        """处理飞书事件"""
        try:
            # 处理URL验证
            if data.get("type") == "url_verification":
                return {"challenge": data.get("challenge", "")}

            # 处理消息事件
            if data.get("type") == "event_callback":
                event = data.get("event", {})

                if event.get("type") == "message":
                    message_type = event.get("msg_type")
                    content = event.get("content", "{}")

                    try:
                        content_data = json.loads(content)
                        text = content_data.get("text", "").strip()
                    except:
                        text = str(content)

                    logger.info(f"收到飞书消息: {text[:100]}...")

                    # 简单回复
                    response = self._generate_simple_response(text)

                    return {
                        "success": True,
                        "response": response,
                        "message": "消息处理完成"
                    }

            return {"success": True, "message": "事件已忽略"}

        except Exception as e:
            logger.error(f"处理事件失败: {str(e)}")
            return {"success": False, "error": str(e)}

    def _generate_simple_response(self, text: str) -> str:
        """生成简单回复"""
        if not text:
            return "收到空消息"

        text_lower = text.lower()

        if any(word in text_lower for word in ['你好', 'hello', 'hi', '嗨']):
            return "你好！我是高数辅导助手，可以帮助您解答高等数学问题。请直接发送数学问题。"
        elif any(word in text_lower for word in ['高数', '微积分', 'calculus', '数学']):
            return "我专注于高等数学辅导。您可以发送数学问题，我会尽力为您解答。"
        elif any(word in text_lower for word in ['帮助', 'help', '功能']):
            return """高数辅导助手功能：
📚 数学问题解答
🧮 表达式计算
📁 文件上传解析
📝 练习题目生成
⚡ 实时互动解答

直接发送问题开始使用吧！"""
        else:
            return f"收到您的消息：{text}\n\n我专注于高数辅导，请发送数学相关问题，我会尽力帮助您解答。"

    def send_message(self, message: str, receive_id: str, msg_type: str = "text") -> Dict:
        """发送消息（简化）"""
        logger.info(f"发送消息到 {receive_id}: {message[:50]}...")
        return {
            "success": True,
            "message": "消息发送成功（演示模式）",
            "data": {"message_id": "demo_message_id"}
        }

    def reply_message(self, message_id: str, content: str, msg_type: str = "text") -> Dict:
        """回复消息（简化）"""
        logger.info(f"回复消息到 {message_id}: {content[:50]}...")
        return {
            "success": True,
            "message": "消息回复成功（演示模式）",
            "data": {"message_id": "demo_reply_id"}
        }

    def get_tenant_access_token(self) -> str:
        """获取租户访问令牌（简化）"""
        return "demo_token"
