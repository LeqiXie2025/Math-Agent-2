from flask import Flask, render_template, request, jsonify, session, send_from_directory, Response
from flask_cors import CORS
import os
from dotenv import load_dotenv
import logging
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime
import traceback
import time
import uuid
import re
from werkzeug.utils import secure_filename
import math

# 加载环境变量
load_dotenv()

# 配置日志
def setup_logging():
    """配置日志系统"""
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logging()

# ========== Tesseract OCR 配置 ==========
TESSERACT_AVAILABLE = False
TESSERACT_PATH = r'D:\Program Files\Tesseract-OCR\tesseract.exe'  # 请根据实际路径修改

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np

    if os.path.exists(TESSERACT_PATH):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        logger.info(f"✅ Tesseract路径已设置: {TESSERACT_PATH}")

        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ Tesseract版本: {version}")
            TESSERACT_AVAILABLE = True

            try:
                languages = pytesseract.get_languages(config='')
                logger.info(f"✅ 已安装语言包: {languages}")
                if 'chi_sim' in languages:
                    logger.info("✅ 中文语言包已安装")
                if 'eng' in languages:
                    logger.info("✅ 英文语言包已安装")
                if 'equ' in languages:
                    logger.info("✅ 数学符号包已安装")
            except Exception as e:
                logger.warning(f"⚠️ 无法获取语言包列表: {str(e)}")

        except Exception as e:
            logger.error(f"❌ Tesseract测试失败: {str(e)}")
    else:
        logger.error(f"❌ Tesseract路径不存在: {TESSERACT_PATH}")

except ImportError:
    logger.error("❌ pytesseract模块未安装，请运行: pip install pytesseract")
except Exception as e:
    logger.error(f"❌ Tesseract初始化失败: {str(e)}")


# ========== LaTeX-OCR 配置 ==========
try:
    from pix2tex.cli import LatexOCR
    LATEX_OCR_AVAILABLE = True
except ImportError:
    LATEX_OCR_AVAILABLE = False
    logger.warning("⚠️ pix2tex未安装，LaTeX-OCR功能将不可用")


# ========== DeepSeek 配置 ==========
try:
    import openai
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    logger.warning("⚠️ openai库未安装，DeepSeek增强功能将不可用")


# ========== 专门针对高等数学复杂选择题的OCR处理器（增强版：集成LaTeX-OCR + DeepSeek融合） ==========
class AdvancedMathOCRProcessor:
    """专门处理高等数学复杂公式和选择题的OCR处理器（增强版）"""

    # 扩展的数学符号白名单（用于Tesseract限制字符集，可选）
    MATH_WHITELIST = "0123456789.+-*/=()[]{}<>xyzuvwPQRSTUVWXYZ∂∫∑∏√∞πθαβγδεζηλμσφψωlim∬∭∇⋅×∩∪⊂⊃∈∉∀∃¬∧∨→⇔≤≥≠≈≡⊥∥∠△□∎ΣΓΔΛΞΠΦΨΩ"

    # 常见OCR错误映射表（用于基础后处理）
    ERROR_MAP = {
        'a': 'α', 'b': 'β', 'g': 'γ', 'd': 'δ', 'e': 'ε',
        'th': 'θ', 'l': 'λ', 'm': 'μ', 's': 'σ', 'f': 'φ',
        'w': 'ω', 'p': 'π',
        'J': '∫', 'JJ': '∬', 'S': '∑', 'P': '∏',
        '0': '∂', 'V': '∇', 'V': '√', '00': '∞',
        '->': '→', '=>': '⇒', '<=>': '⇔',
        '<=': '≤', '>=': '≥', '!=': '≠', '~=': '≈', '==': '≡',
        '_|_': '⊥', '||': '∥', '/_': '∠', 'Δ': '△', '[]': '□',
        'lim ': 'lim ',  # 保留空格
        'dy dz': 'dy dz',
        'dz dx': 'dz dx',
        'dx dy': 'dx dy',
        'Sigma': 'Σ',
        'sigma': 'Σ',
        'iint': '∬',
        'iiint': '∭',
    }

    def __init__(self):
        self.tesseract_available = TESSERACT_AVAILABLE

        # 初始化LaTeX-OCR
        self.latex_ocr_available = LATEX_OCR_AVAILABLE
        self.latex_model = None
        if self.latex_ocr_available:
            try:
                self.latex_model = LatexOCR()
                logger.info("✅ LaTeX-OCR模型初始化成功")
            except Exception as e:
                self.latex_ocr_available = False
                logger.error(f"❌ LaTeX-OCR模型初始化失败: {str(e)}")

        # 初始化DeepSeek客户端
        self.deepseek_available = False
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if api_key and api_key != 'your_deepseek_api_key_here' and DEEPSEEK_AVAILABLE:
            try:
                self.deepseek_client = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                self.deepseek_available = True
                logger.info("✅ DeepSeek客户端初始化成功")
            except Exception as e:
                logger.error(f"❌ DeepSeek客户端初始化失败: {str(e)}")
        else:
            logger.info("ℹ️ DeepSeek未配置或库未安装，将使用传统后处理")

        # 定义多个预处理函数，用于不同策略
        self.preprocessors = [
            self._preprocess_ultra,
            self._preprocess_high_contrast,
            self._preprocess_sharpen,
            self._preprocess_simple,
        ]

    # ---------- 预处理方法 ----------
    def _preprocess_ultra(self, image_path):
        """极致预处理：自适应二值化 + 形态学连接 + 放大 + 反转"""
        try:
            img = Image.open(image_path).convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(3.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(3.0)

            width, height = img.size
            img = img.resize((width * 4, height * 4), Image.Resampling.LANCZOS)

            img_array = np.array(img)
            try:
                from scipy.ndimage import uniform_filter
                block_size = 25
                local_mean = uniform_filter(img_array.astype(float), size=block_size)
                local_std = uniform_filter((img_array - local_mean) ** 2, size=block_size) ** 0.5
                threshold = local_mean - 0.15 * local_std
                binary = (img_array > threshold).astype(np.uint8) * 255
                img = Image.fromarray(binary)
            except ImportError:
                img = img.point(lambda x: 0 if x < 180 else 255, '1')

            try:
                from scipy.ndimage import binary_dilation, binary_erosion
                img_array = np.array(img)
                structure = np.ones((1, 3))
                dilated = binary_dilation(img_array > 128, structure=structure)
                eroded = binary_erosion(dilated, structure=structure)
                img = Image.fromarray((eroded * 255).astype(np.uint8))
            except ImportError:
                img = img.filter(ImageFilter.MaxFilter(3))
                img = img.filter(ImageFilter.MinFilter(3))

            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = img.filter(ImageFilter.EDGE_ENHANCE_MORE)
            img = ImageOps.invert(img)
            return img
        except Exception as e:
            logger.warning(f"极致预处理失败: {str(e)}")
            return None

    def _preprocess_high_contrast(self, image_path):
        """高对比度预处理"""
        try:
            img = Image.open(image_path).convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(3.0)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(2.5)
            img = img.point(lambda x: 0 if x < 200 else 255, '1')
            width, height = img.size
            img = img.resize((width * 3, height * 3), Image.Resampling.LANCZOS)
            img = img.filter(ImageFilter.MedianFilter(size=3))
            img = ImageOps.invert(img)
            return img
        except Exception as e:
            return None

    def _preprocess_sharpen(self, image_path):
        """锐化增强预处理"""
        try:
            img = Image.open(image_path).convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.5)
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(4.0)
            img = img.filter(ImageFilter.SHARPEN)
            width, height = img.size
            img = img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
            img = ImageOps.invert(img)
            return img
        except Exception as e:
            return None

    def _preprocess_simple(self, image_path):
        """简单预处理（后备）"""
        try:
            img = Image.open(image_path).convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.point(lambda x: 0 if x < 200 else 255, '1')
            width, height = img.size
            img = img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
            img = ImageOps.invert(img)
            return img
        except Exception as e:
            return None

    # ---------- Tesseract识别 ----------
    def _recognize_with_tesseract(self, image, lang, config, psm=None):
        """使用Tesseract进行OCR识别"""
        if not self.tesseract_available:
            return ""
        try:
            if isinstance(image, str):
                img = Image.open(image)
            else:
                img = image

            full_config = config
            if psm is not None:
                full_config = f"--psm {psm} " + full_config

            text = pytesseract.image_to_string(img, lang=lang, config=full_config)
            return text.strip()
        except Exception as e:
            logger.debug(f"Tesseract识别失败 (lang={lang}, psm={psm}): {str(e)}")
            return ""

    # ---------- LaTeX-OCR识别 ----------
    def _recognize_with_latex_ocr(self, image_path):
        """使用LaTeX-OCR识别数学公式，返回LaTeX代码"""
        if not self.latex_ocr_available or self.latex_model is None:
            return ""
        try:
            img = Image.open(image_path).convert('RGB')
            latex = self.latex_model(img)
            return latex.strip()
        except Exception as e:
            logger.error(f"LaTeX-OCR识别失败: {str(e)}")
            return ""

    # ---------- DeepSeek增强后处理 ----------
    def _enhance_with_deepseek(self, tesseract_candidates, latex_result):
        """
        使用DeepSeek融合Tesseract文本候选和LaTeX-OCR公式结果
        """
        if not self.deepseek_available:
            return None

        prompt = """你是一个专业的数学OCR后处理助手。下面是从同一张数学题图片中获得的两种识别结果：

1. Tesseract OCR（通用OCR）识别出的多个文本候选（可能存在错误）：
"""
        for i, text in enumerate(tesseract_candidates, 1):
            preview = text[:500] + ("..." if len(text) > 500 else "")
            prompt += f"候选{i}: {preview}\n\n"

        if latex_result:
            prompt += f"2. LaTeX-OCR（专用公式识别）识别出的LaTeX代码：\n{latex_result}\n\n"
        else:
            prompt += "2. LaTeX-OCR未能识别出有效公式。\n\n"

        prompt += """请综合以上信息，还原出最准确的原始数学题目。要求：
- 保留题目中所有的数学公式，并用正确的LaTeX语法表示（如 \\( ... \\) 或 \[ ... \]）。
- 如果题目包含选择题选项，请保持A. B. C. D. 的格式。
- 修复常见的OCR错误（如将数字0误认为字母O，希腊字母混淆等）。
- 输出最终结果，不要包含任何解释或额外标记。

原始题目示例格式：
2. 设 \\( P = P(x, y, z) \\), \\( Q = Q(x, y, z) \\) 均为连续函数，\\(\\Sigma\\) 为以 \\(z = \\sqrt{1 - x^2 - y^2} (x \\leq 0, y \\geq 0)\\) 的上侧，则 \\[ \\iint_{\\Sigma} P \\, dy \\, dz + Q \\, dz \\, dx = (\\quad ) \\]
A. \\[\\iint_{\\Sigma} \\left( \\frac{x}{z} P + \\frac{y}{z} Q \\right) \\, dx \\, dy\\]
B. \\[\\iint_{\\Sigma} \\left( -\\frac{x}{z} P + \\frac{y}{z} Q \\right) \\, dx \\, dy\\]
C. \\[\\iint_{\\Sigma} \\left( \\frac{x}{z} P - \\frac{y}{z} Q \\right) \\, dx \\, dy\\]
D. \\[\\iint_{\\Sigma} \\left( -\\frac{x}{z} P - \\frac{y}{z} Q \\right) \\, dx \\, dy\\]

请直接输出还原后的题目："""

        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "你是一个专业的数学OCR后处理助手，能够准确还原数学题目。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            enhanced_text = response.choices[0].message.content.strip()
            logger.info(f"✅ DeepSeek增强成功，输出长度: {len(enhanced_text)}")
            return enhanced_text
        except Exception as e:
            logger.error(f"❌ DeepSeek调用失败: {str(e)}")
            return None

    # ---------- 主识别函数 ----------
    def recognize_complex_math(self, image_path):
        """专门识别复杂数学公式：Tesseract多策略 + LaTeX-OCR + DeepSeek融合"""
        if not self.tesseract_available:
            return None

        # ---- 1. 收集Tesseract候选文本 ----
        tesseract_candidates = []  # 存储文本

        # 定义要尝试的参数组合
        lang_combinations = [
            'equ+eng',
            'chi_sim+equ+eng',
            'eng',
            'chi_sim',
        ]
        psm_modes = [6, 8, 13, 3, 4]
        base_config = '-c tessedit_do_invert=0'

        for preproc in self.preprocessors:
            try:
                processed_img = preproc(image_path)
                if processed_img is None:
                    continue

                for lang in lang_combinations:
                    for psm in psm_modes:
                        text = self._recognize_with_tesseract(processed_img, lang, base_config, psm)
                        if text and len(text) > 10:
                            tesseract_candidates.append(text)

                # 带白名单的配置
                white_config = f'-c tessedit_char_whitelist="{self.MATH_WHITELIST}"'
                for lang in lang_combinations:
                    for psm in psm_modes:
                        text = self._recognize_with_tesseract(processed_img, lang, white_config, psm)
                        if text and len(text) > 10:
                            tesseract_candidates.append(text)

            except Exception as e:
                logger.warning(f"预处理方法 {preproc.__name__} 失败: {str(e)}")
                continue

        # 去重
        unique_candidates = []
        seen = set()
        for text in tesseract_candidates:
            fingerprint = text[:100]
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_candidates.append(text)

        # ---- 2. 使用LaTeX-OCR识别公式 ----
        latex_result = self._recognize_with_latex_ocr(image_path)

        # ---- 3. 尝试DeepSeek融合 ----
        enhanced = None
        if self.deepseek_available and (unique_candidates or latex_result):
            enhanced = self._enhance_with_deepseek(unique_candidates, latex_result)

        if enhanced and len(enhanced) > 20:
            cleaned = self.post_process_math_text(enhanced)
            return cleaned
        else:
            # 回退到最佳Tesseract候选（按长度简单选）
            if unique_candidates:
                best_text = max(unique_candidates, key=len)
                cleaned = self.post_process_math_text(best_text)
                return cleaned
            elif latex_result:
                return latex_result
            else:
                return None

    # ---------- 基础后处理 ----------
    def post_process_math_text(self, text):
        """后处理数学文本，修复常见错误"""
        if not text:
            return text

        for wrong, correct in self.ERROR_MAP.items():
            text = text.replace(wrong, correct)

        # 修复极限格式
        text = re.sub(r'lim\s*\(', 'lim (', text)
        text = re.sub(r'lim_([^{])', r'lim_{\1}', text)

        # 上下标转换（示例，可根据需要扩展）
        def superscript_replace(match):
            char = match.group(1)
            sup_map = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            }
            return sup_map.get(char, '^' + char)
        text = re.sub(r'\^([0-9])', superscript_replace, text)

        def subscript_replace(match):
            char = match.group(1)
            sub_map = {
                '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
                '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
            }
            return sub_map.get(char, '_' + char)
        text = re.sub(r'_([0-9])', subscript_replace, text)

        # 括号统一
        text = text.replace('（', '(').replace('）', ')')
        text = text.replace('［', '[').replace('］', ']')
        text = text.replace('｛', '{').replace('｝', '}')
        text = text.replace('〈', '<').replace('〉', '>')

        # 分数简化
        text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', text)

        # 平方根
        text = re.sub(r'sqrt\s*\(', '√(', text)

        # 清理多余空格
        text = re.sub(r'\s+', ' ', text)

        # 微分符号
        text = re.sub(r'd\s*x', 'dx', text)
        text = re.sub(r'd\s*y', 'dy', text)
        text = re.sub(r'd\s*z', 'dz', text)

        return text.strip()

    def format_choice_question(self, text):
        """格式化选择题（保持原样）"""
        if not text:
            return text

        lines = text.split('\n')
        formatted_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if re.match(r'^\d+\.', line):
                formatted_lines.append(f"\n{line}")
            elif re.match(r'^[A-D]\.', line) or re.match(r'^[A-D]\)', line):
                formatted_lines.append(f"\n{line}")
            elif any(c in line for c in '=∫∂Σ√∞πθαβγ'):
                formatted_lines.append(f"\n{line}")
            else:
                formatted_lines.append(line)
        return '\n'.join(formatted_lines)


# ========== 文件处理器 ==========
class ImprovedFileHandler:
    """改进的文件处理器 - 增强版OCR识别"""

    def __init__(self, upload_folder):
        self.upload_folder = upload_folder
        self.allowed_extensions = app.config['ALLOWED_EXTENSIONS']
        self.math_ocr = AdvancedMathOCRProcessor()

    def allowed_file(self, filename):
        if '.' not in filename:
            return False
        ext = filename.rsplit('.', 1)[1].lower()
        return ext in self.allowed_extensions

    def save_file(self, file):
        try:
            original_filename = file.filename
            logger.info(f"原始文件名: {original_filename}")

            ext = ''
            if '.' in original_filename:
                ext = original_filename.rsplit('.', 1)[1].lower()

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = str(uuid.uuid4())[:8]
            new_filename = f"{timestamp}_{unique_id}.{ext}" if ext else f"{timestamp}_{unique_id}"

            filepath = os.path.join(self.upload_folder, new_filename)
            file.save(filepath)

            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                logger.info(f"✅ 文件保存成功: {new_filename}, 大小: {os.path.getsize(filepath)} 字节")
                return True, new_filename, filepath, original_filename
            else:
                logger.error(f"❌ 文件保存失败: {filepath}")
                return False, None, None, original_filename

        except Exception as e:
            logger.error(f"❌ 保存文件异常: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None, None, file.filename if file else 'unknown'

    def preprocess_image_simple(self, image_path):
        try:
            img = Image.open(image_path)
            if img.mode != 'L':
                img = img.convert('L')
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(2.0)
            img = img.point(lambda x: 0 if x < 200 else 255, '1')
            width, height = img.size
            img = img.resize((width * 2, height * 2), Image.Resampling.LANCZOS)
            return img
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            return None

    def ocr_with_all_languages(self, image_path):
        """后备方案：简单多语言OCR"""
        try:
            if not TESSERACT_AVAILABLE:
                return None

            img = self.preprocess_image_simple(image_path)
            if img is None:
                return None

            results = []
            try:
                text = pytesseract.image_to_string(img, lang='equ+eng', config='--oem 3 --psm 6')
                if text and text.strip():
                    results.append(("【数学公式识别】", text.strip()))
            except:
                pass
            try:
                text = pytesseract.image_to_string(img, lang='chi_sim+eng', config='--oem 3 --psm 6')
                if text and text.strip():
                    results.append(("【中英文混合识别】", text.strip()))
            except:
                pass

            if results:
                combined = ""
                for title, content in results:
                    combined += f"{title}\n{content}\n\n"
                return combined
            return None
        except Exception as e:
            logger.error(f"OCR识别失败: {str(e)}")
            return None

    def extract_text_from_image_enhanced(self, image_path):
        """增强的图片文本提取 - 使用改进的OCR处理器"""
        try:
            logger.info(f"开始增强OCR识别: {image_path}")

            if not TESSERACT_AVAILABLE:
                return self._get_manual_input_prompt(image_path)

            math_text = self.math_ocr.recognize_complex_math(image_path)

            if math_text and len(math_text.strip()) > 30:
                cleaned_text = self.math_ocr.post_process_math_text(math_text)
                formatted_text = self.math_ocr.format_choice_question(cleaned_text)
                logger.info(f"✅ 复杂数学OCR识别成功: {len(formatted_text)}字符")
                logger.info(f"识别结果预览: {formatted_text[:300]}")
                return formatted_text

            logger.warning("复杂数学OCR识别结果不理想，使用原方法")
            return self.ocr_with_all_languages(image_path)

        except Exception as e:
            logger.error(f"增强OCR识别失败: {str(e)}")
            return self.ocr_with_all_languages(image_path)

    def extract_text_from_image(self, image_path):
        return self.extract_text_from_image_enhanced(image_path)

    def _get_manual_input_prompt(self, image_path):
        filename = os.path.basename(image_path)
        tesseract_status = "✅ 已安装" if TESSERACT_AVAILABLE else "❌ 未安装"
        language_status = ""
        if TESSERACT_AVAILABLE:
            try:
                languages = pytesseract.get_languages(config='')
                has_chinese = 'chi_sim' in languages
                has_equ = 'equ' in languages
                language_status = f"中文: {'✅' if has_chinese else '❌'} 数学: {'✅' if has_equ else '❌'}"
            except:
                language_status = "中文: ⚠️ 数学: ⚠️"

        return f"""【请手动输入数学问题】

📁 文件名: {filename}
🔍 Tesseract: {tesseract_status} {language_status}
⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'⚠️ OCR未能识别出有效文字，请直接在下方输入框中输入您的问题' if TESSERACT_AVAILABLE else '⚠️ Tesseract未安装，请直接在下方输入框中输入您的问题'}

📝 输入格式示例:

1. 极限: lim(x→0) sin(x)/x
2. 导数: derivative of x^2 + 2x
3. 积分: integrate x^2 dx from 0 to 1
4. 方程: solve x^2 - 5x + 6 = 0
5. 选择题: 设 P = P(x,y,z), Q = Q(x,y,z) 均为连续函数，Σ 为以 z = √(1-x²-y²) (x≤0, y≥0) 的上侧，则 ∬_Σ P dy dz + Q dz dx = ( )

您的问题:"""

    def extract_text_from_docx(self, docx_path):
        try:
            import docx
            doc = docx.Document(docx_path)
            text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            return text if text else "[Word文档] 文档内容为空"
        except ImportError:
            return "[Word解析错误] 请安装python-docx: pip install python-docx"
        except Exception as e:
            return f"[Word解析错误] {str(e)}"

    def extract_text_from_pdf(self, pdf_path):
        try:
            import PyPDF2
            text = ""
            with open(pdf_path, 'rb') as f:
                pdf = PyPDF2.PdfReader(f)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip() if text else "[PDF文档] 未提取到文本内容"
        except ImportError:
            return "[PDF解析错误] 请安装PyPDF2: pip install PyPDF2"
        except Exception as e:
            return f"[PDF解析错误] {str(e)}"

    def extract_text_from_txt(self, txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except:
            try:
                with open(txt_path, 'r', encoding='gbk') as f:
                    return f.read().strip()
            except:
                return "[文本文件] 无法读取"

    def extract_text(self, filepath):
        try:
            filename = os.path.basename(filepath)
            ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
            logger.info(f"开始提取文件文本: {filename}, 类型: {ext}")

            if ext in ['docx']:
                text = self.extract_text_from_docx(filepath)
            elif ext in ['doc']:
                text = "[Word文档] 旧版.doc格式支持有限"
            elif ext in ['pdf']:
                text = self.extract_text_from_pdf(filepath)
            elif ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']:
                text = self.extract_text_from_image(filepath)
            elif ext in ['txt']:
                text = self.extract_text_from_txt(filepath)
            else:
                text = f"[不支持的文件类型] {ext}"

            return text
        except Exception as e:
            logger.error(f"文本提取失败: {str(e)}")
            return f"[文件处理错误] {str(e)}"


# ========== 创建Flask应用 ==========
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-for-flask-app-2024')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 32 * 1024 * 1024))  # 32MB
app.config['ALLOWED_EXTENSIONS'] = set(
    os.getenv('ALLOWED_EXTENSIONS', 'png,jpg,jpeg,gif,bmp,tiff,pdf,doc,docx,txt').split(','))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1小时

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)

CORS(app, supports_credentials=True)

# 全局服务实例
tutor = None
file_handler = None
feishu_bot = None


def initialize_services():
    """初始化所有服务"""
    global tutor, file_handler, feishu_bot

    try:
        logger.info("正在初始化服务...")

        file_handler = ImprovedFileHandler(app.config['UPLOAD_FOLDER'])
        logger.info("✅ 文件处理器初始化成功")

        try:
            from services import CalculusTutor
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if api_key and api_key != 'your_deepseek_api_key_here':
                tutor = CalculusTutor(api_key)
                logger.info("✅ DeepSeek AI助手初始化成功")
            else:
                logger.warning("⚠️ DeepSeek API密钥未配置，使用模拟模式")
                from services import MockTutor
                tutor = MockTutor()
        except Exception as e:
            logger.error(f"❌ AI助手初始化失败: {str(e)}")
            from services import MockTutor
            tutor = MockTutor()

        feishu_app_id = os.getenv('FEISHU_APP_ID')
        feishu_app_secret = os.getenv('FEISHU_APP_SECRET')
        if feishu_app_id and feishu_app_secret and feishu_app_id != 'your_feishu_app_id':
            try:
                from feishu_integration import FeishuBot
                feishu_bot = FeishuBot(
                    app_id=feishu_app_id,
                    app_secret=feishu_app_secret,
                    verification_token=os.getenv('FEISHU_VERIFICATION_TOKEN', ''),
                    encrypt_key=os.getenv('FEISHU_ENCRYPT_KEY', '')
                )
                logger.info("✅ 飞书机器人初始化成功")
            except Exception as e:
                logger.error(f"❌ 飞书机器人初始化失败: {str(e)}")
                feishu_bot = None
        else:
            logger.info("ℹ️ 飞书机器人未配置")
            feishu_bot = None

        logger.info("✅ 所有服务初始化完成")
    except Exception as e:
        logger.error(f"❌ 服务初始化失败: {str(e)}")
        raise


# ========== API路由 ==========
@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'GET':
        challenge = request.args.get('challenge', '')
        if challenge:
            return jsonify({'challenge': challenge})
        return render_template('index.html')
    return jsonify({'success': True})


@app.route('/feishu-webhook', methods=['GET', 'POST'])
def feishu_webhook():
    if request.method == 'GET':
        challenge = request.args.get('challenge', '')
        if challenge:
            return jsonify({'challenge': challenge})
    return jsonify({'success': True})


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'tesseract': 'available' if TESSERACT_AVAILABLE else 'unavailable',
        'latex_ocr': 'available' if LATEX_OCR_AVAILABLE else 'unavailable',
        'deepseek': 'available' if DEEPSEEK_AVAILABLE and os.getenv('DEEPSEEK_API_KEY') else 'unavailable',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/status')
def status():
    return jsonify({
        'success': True,
        'status': 'online',
        'ocr_available': TESSERACT_AVAILABLE
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        if file_handler is None:
            initialize_services()

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未提供文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400

        if not file_handler.allowed_file(file.filename):
            return jsonify({'success': False, 'error': '不支持的文件类型'}), 400

        success, filename, filepath, original_name = file_handler.save_file(file)
        if not success:
            return jsonify({'success': False, 'error': '保存文件失败'}), 500

        text = file_handler.extract_text(filepath)

        ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        is_image = ext in ['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff']

        ocr_success = not (text.startswith('【请手动输入数学问题】'))

        return jsonify({
            'success': True,
            'filename': filename,
            'original_filename': original_name,
            'text': text,
            'is_image': is_image,
            'ocr_success': ocr_success,
            'tesseract_available': TESSERACT_AVAILABLE
        })

    except Exception as e:
        logger.error(f"上传失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/upload-and-ask', methods=['POST'])
def upload_and_ask():
    """上传文件 -> OCR识别 -> 自动提问 -> 返回答案"""
    try:
        if file_handler is None or tutor is None:
            initialize_services()

        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '未提供文件'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400

        if not file_handler.allowed_file(file.filename):
            return jsonify({'success': False, 'error': '不支持的文件类型'}), 400

        # 保存文件
        success, filename, filepath, original_name = file_handler.save_file(file)
        if not success:
            return jsonify({'success': False, 'error': '保存文件失败'}), 500

        # 提取文本（OCR）
        text = file_handler.extract_text(filepath)

        # 如果OCR结果是手动输入提示，则返回引导
        if text.startswith('【请手动输入数学问题】'):
            return jsonify({
                'success': True,
                'manual_input': True,
                'text': text,
                'answer': None,
                'filename': filename
            })

        # 调用AI助手解答问题（默认使用中文）
        answer = tutor.ask_question(text, language='zh')

        return jsonify({
            'success': True,
            'filename': filename,
            'original_filename': original_name,
            'text': text,
            'answer': answer,
            'manual_input': False
        })

    except Exception as e:
        logger.error(f"上传并提问失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question():
    try:
        if tutor is None:
            initialize_services()

        data = request.json
        question = data.get('question', '').strip()
        language = data.get('language', 'zh')

        if not question:
            return jsonify({'success': False, 'error': '问题不能为空'}), 400

        if question.startswith('【请手动输入数学问题】'):
            return jsonify({
                'success': True,
                'answer': '📝 请在上方输入框中直接输入您的数学问题。\n\n📌 输入格式示例：\n\n1. 极限: lim(x→0) sin(x)/x\n2. 导数: derivative of x^2 + 2x\n3. 积分: integrate x^2 dx from 0 to 1\n4. 方程: solve x^2 - 5x + 6 = 0\n5. 选择题: 设 P = P(x,y,z), Q = Q(x,y,z) 均为连续函数，Σ 为以 z = √(1-x²-y²) (x≤0, y≥0) 的上侧，则 ∬_Σ P dy dz + Q dz dx = ( )\n6. 中文: 求函数f(x)=x²+2x的导数',
                'question': question,
                'language': language
            })

        answer = tutor.ask_question(question, language)

        return jsonify({
            'success': True,
            'answer': answer,
            'question': question,
            'language': language
        })

    except Exception as e:
        logger.error(f"处理问题失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        if tutor is None:
            initialize_services()

        data = request.json
        expression = data.get('expression', '').strip()

        if not expression:
            return jsonify({'success': False, 'error': '表达式不能为空'}), 400

        result = tutor.calculate_expression(expression)

        return jsonify({
            'success': True,
            'result': result.get('result', ''),
            'steps': result.get('steps', ''),
            'numerical': result.get('numerical', ''),
            'expression': expression
        })

    except Exception as e:
        logger.error(f"计算失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/generate-practice', methods=['POST'])
def generate_practice():
    """生成练习题"""
    try:
        if tutor is None:
            initialize_services()

        data = request.json
        topic = data.get('topic', 'limits')
        difficulty = int(data.get('difficulty', 3))
        count = int(data.get('count', 3))

        questions = []

        # 根据主题生成不同的练习题
        for i in range(count):
            # 为每个题目生成不同的随机参数
            import random
            import math

            if topic == 'limits':
                # 极限题目 - 生成不同的极限表达式
                limit_types = ['polynomial', 'trigonometric', 'rational', 'exponential']
                limit_type = random.choice(limit_types)

                if limit_type == 'polynomial':
                    # 多项式极限
                    a = random.randint(1, 5)
                    b = random.randint(1, 5)
                    c = random.randint(0, 3)
                    x_val = random.randint(0, 3)
                    answer_value = a * x_val ** 2 + b * x_val + c
                    questions.append({
                        'id': i + 1,
                        'question': f'计算极限：lim(x→{x_val}) ({a}x² + {b}x + {c})',
                        'answer': f'= {answer_value}',
                        'detailed_answer': f'直接将 x = {x_val} 代入表达式：\n\n' +
                                           f'{a}×{x_val}² + {b}×{x_val} + {c}\n' +
                                           f'= {a}×{x_val ** 2} + {b * x_val} + {c}\n' +
                                           f'= {a * x_val ** 2} + {b * x_val} + {c}\n' +
                                           f'= {answer_value}',
                        'hint': '直接代入法',
                        'difficulty': difficulty,
                        'topic': '极限'
                    })
                elif limit_type == 'trigonometric':
                    # 三角函数极限
                    x_val = 0
                    a = random.randint(1, 3)
                    answer_value = a
                    questions.append({
                        'id': i + 1,
                        'question': f'计算极限：lim(x→{x_val}) sin({a}x)/x',
                        'answer': f'= {answer_value}',
                        'detailed_answer': f'使用重要极限公式：lim(x→0) sin(ax)/x = a\n\n' +
                                           f'lim(x→0) sin({a}x)/x = {a}',
                        'hint': '重要极限公式：lim(x→0) sin(ax)/x = a',
                        'difficulty': difficulty,
                        'topic': '极限'
                    })
                elif limit_type == 'rational':
                    # 有理函数极限
                    a = random.randint(1, 4)
                    b = random.randint(1, 4)
                    c = random.randint(1, 3)
                    answer_value = f'{a}/{b}'
                    questions.append({
                        'id': i + 1,
                        'question': f'计算极限：lim(x→∞) ({a}x² + {b}x + {c})/({b}x² + {c}x + {a})',
                        'answer': f'= {answer_value}',
                        'detailed_answer': f'当 x→∞ 时，比较分子和分母的最高次项：\n\n' +
                                           f'分子最高次项：{a}x²\n' +
                                           f'分母最高次项：{b}x²\n\n' +
                                           f'因此极限 = {a}/{b}',
                        'hint': '当x→∞时，比较最高次项系数',
                        'difficulty': difficulty,
                        'topic': '极限'
                    })
                else:
                    # 指数函数极限
                    a = random.randint(1, 3)
                    answer_value = a
                    questions.append({
                        'id': i + 1,
                        'question': f'计算极限：lim(x→0) (e^{a}x - 1)/x',
                        'answer': f'= {answer_value}',
                        'detailed_answer': f'使用重要极限公式：lim(x→0) (e^ax - 1)/x = a\n\n' +
                                           f'lim(x→0) (e^{a}x - 1)/x = {a}',
                        'hint': '重要极限：lim(x→0) (e^ax - 1)/x = a',
                        'difficulty': difficulty,
                        'topic': '极限'
                    })

            elif topic == 'derivatives':
                # 导数题目 - 生成不同的函数
                derivative_types = ['polynomial', 'trigonometric', 'exponential', 'composite']
                derivative_type = random.choice(derivative_types)

                if derivative_type == 'polynomial':
                    a = random.randint(1, 4)
                    b = random.randint(2, 4)
                    c = random.randint(1, 3)
                    answer_coef1 = a * b
                    answer_coef2 = c * (b - 1)
                    questions.append({
                        'id': i + 1,
                        'question': f'求导数：f(x) = {a}x^{b} + {c}x^{b - 1}',
                        'answer': f"f'(x) = {answer_coef1}x^{b - 1} + {answer_coef2}x^{b - 2}",
                        'detailed_answer': f'使用幂函数求导公式：d/dx(x^n) = nx^(n-1)\n\n' +
                                           f'第一项：d/dx({a}x^{b}) = {a}×{b}x^{b - 1} = {answer_coef1}x^{b - 1}\n' +
                                           f'第二项：d/dx({c}x^{b - 1}) = {c}×{b - 1}x^{b - 2} = {answer_coef2}x^{b - 2}\n\n' +
                                           f'因此：f\'(x) = {answer_coef1}x^{b - 1} + {answer_coef2}x^{b - 2}',
                        'hint': '幂函数求导公式：d/dx(x^n) = nx^(n-1)',
                        'difficulty': difficulty,
                        'topic': '导数'
                    })
                elif derivative_type == 'trigonometric':
                    a = random.randint(1, 3)
                    func = random.choice(['sin', 'cos', 'tan'])
                    if func == 'sin':
                        questions.append({
                            'id': i + 1,
                            'question': f'求导数：f(x) = sin({a}x)',
                            'answer': f"f'(x) = {a}cos({a}x)",
                            'detailed_answer': f'使用三角函数求导公式：d/dx[sin(ax)] = a·cos(ax)\n\n' +
                                               f'因此：f\'(x) = {a}cos({a}x)',
                            'hint': 'sin(ax)的导数为a·cos(ax)',
                            'difficulty': difficulty,
                            'topic': '导数'
                        })
                    elif func == 'cos':
                        questions.append({
                            'id': i + 1,
                            'question': f'求导数：f(x) = cos({a}x)',
                            'answer': f"f'(x) = -{a}sin({a}x)",
                            'detailed_answer': f'使用三角函数求导公式：d/dx[cos(ax)] = -a·sin(ax)\n\n' +
                                               f'因此：f\'(x) = -{a}sin({a}x)',
                            'hint': 'cos(ax)的导数为-a·sin(ax)',
                            'difficulty': difficulty,
                            'topic': '导数'
                        })
                    else:
                        questions.append({
                            'id': i + 1,
                            'question': f'求导数：f(x) = tan({a}x)',
                            'answer': f"f'(x) = {a}sec²({a}x)",
                            'detailed_answer': f'使用三角函数求导公式：d/dx[tan(ax)] = a·sec²(ax)\n\n' +
                                               f'因此：f\'(x) = {a}sec²({a}x)',
                            'hint': 'tan(ax)的导数为a·sec²(ax)',
                            'difficulty': difficulty,
                            'topic': '导数'
                        })
                elif derivative_type == 'exponential':
                    a = random.randint(1, 3)
                    base = random.choice(['e', '2', '3'])
                    if base == 'e':
                        questions.append({
                            'id': i + 1,
                            'question': f'求导数：f(x) = e^{a}x',
                            'answer': f"f'(x) = {a}e^{a}x",
                            'detailed_answer': f'使用指数函数求导公式：d/dx[e^(ax)] = a·e^(ax)\n\n' +
                                               f'因此：f\'(x) = {a}e^{a}x',
                            'hint': 'e^(ax)的导数为a·e^(ax)',
                            'difficulty': difficulty,
                            'topic': '导数'
                        })
                    else:
                        b = int(base)
                        ln_val = round(math.log(b), 4)
                        questions.append({
                            'id': i + 1,
                            'question': f'求导数：f(x) = {b}^{a}x',
                            'answer': f"f'(x) = {a}·{b}^{a}x·ln({b})",
                            'detailed_answer': f'使用指数函数求导公式：d/dx[a^(bx)] = b·a^(bx)·ln(a)\n\n' +
                                               f'因此：f\'(x) = {a}·{b}^{a}x·ln({b})\n' +
                                               f'其中 ln({b}) ≈ {ln_val}',
                            'hint': 'a^(bx)的导数为b·a^(bx)·ln(a)',
                            'difficulty': difficulty,
                            'topic': '导数'
                        })
                else:
                    # 复合函数
                    a = random.randint(1, 3)
                    b = random.randint(1, 3)
                    answer_coef = 3 * a
                    questions.append({
                        'id': i + 1,
                        'question': f'求导数：f(x) = ({a}x + {b})³',
                        'answer': f"f'(x) = {answer_coef}({a}x + {b})²",
                        'detailed_answer': f'使用链式法则：d/dx[(ax+b)³] = 3(ax+b)² × a\n\n' +
                                           f'= 3a(ax+b)²\n' +
                                           f'= {answer_coef}({a}x + {b})²',
                        'hint': '链式法则：d/dx[(ax+b)³] = 3a(ax+b)²',
                        'difficulty': difficulty,
                        'topic': '导数'
                    })

            elif topic == 'integrals':
                # 积分题目 - 生成不同的积分表达式
                integral_types = ['polynomial', 'trigonometric', 'exponential', 'rational']
                integral_type = random.choice(integral_types)

                if integral_type == 'polynomial':
                    a = random.randint(1, 4)
                    b = random.randint(1, 4)
                    c = random.randint(0, 3)
                    coef = round(a / (b + 1), 2)
                    questions.append({
                        'id': i + 1,
                        'question': f'计算不定积分：∫({a}x^{b} + {c}) dx',
                        'answer': f'= {coef}x^{b + 1} + {c}x + C',
                        'detailed_answer': f'使用幂函数积分公式：∫x^n dx = x^(n+1)/(n+1) + C\n\n' +
                                           f'第一项：∫{a}x^{b} dx = {a} × x^{b + 1}/{b + 1} = {coef}x^{b + 1}\n' +
                                           f'第二项：∫{c} dx = {c}x\n\n' +
                                           f'因此：∫({a}x^{b} + {c}) dx = {coef}x^{b + 1} + {c}x + C',
                        'hint': '幂函数积分公式：∫x^n dx = x^(n+1)/(n+1) + C',
                        'difficulty': difficulty,
                        'topic': '积分'
                    })
                elif integral_type == 'trigonometric':
                    a = random.randint(1, 3)
                    func = random.choice(['sin', 'cos'])
                    if func == 'sin':
                        coef = round(-1 / a, 2)
                        questions.append({
                            'id': i + 1,
                            'question': f'计算不定积分：∫ sin({a}x) dx',
                            'answer': f'= {coef} cos({a}x) + C',
                            'detailed_answer': f'使用三角函数积分公式：∫ sin(ax) dx = -1/a·cos(ax) + C\n\n' +
                                               f'∫ sin({a}x) dx = -1/{a}·cos({a}x) + C = {coef} cos({a}x) + C',
                            'hint': '∫ sin(ax) dx = -1/a·cos(ax) + C',
                            'difficulty': difficulty,
                            'topic': '积分'
                        })
                    else:
                        coef = round(1 / a, 2)
                        questions.append({
                            'id': i + 1,
                            'question': f'计算不定积分：∫ cos({a}x) dx',
                            'answer': f'= {coef} sin({a}x) + C',
                            'detailed_answer': f'使用三角函数积分公式：∫ cos(ax) dx = 1/a·sin(ax) + C\n\n' +
                                               f'∫ cos({a}x) dx = 1/{a}·sin({a}x) + C = {coef} sin({a}x) + C',
                            'hint': '∫ cos(ax) dx = 1/a·sin(ax) + C',
                            'difficulty': difficulty,
                            'topic': '积分'
                        })
                elif integral_type == 'exponential':
                    a = random.randint(1, 3)
                    coef = round(1 / a, 2)
                    questions.append({
                        'id': i + 1,
                        'question': f'计算不定积分：∫ e^{a}x dx',
                        'answer': f'= {coef} e^{a}x + C',
                        'detailed_answer': f'使用指数函数积分公式：∫ e^(ax) dx = 1/a·e^(ax) + C\n\n' +
                                           f'∫ e^{a}x dx = 1/{a}·e^{a}x + C = {coef} e^{a}x + C',
                        'hint': '∫ e^(ax) dx = 1/a·e^(ax) + C',
                        'difficulty': difficulty,
                        'topic': '积分'
                    })
                else:
                    # 有理函数积分
                    a = random.randint(1, 3)
                    coef = round(1 / a, 2)
                    questions.append({
                        'id': i + 1,
                        'question': f'计算不定积分：∫ 1/({a}x + 1) dx',
                        'answer': f'= {coef} ln|{a}x + 1| + C',
                        'detailed_answer': f'使用有理函数积分公式：∫ 1/(ax+b) dx = 1/a·ln|ax+b| + C\n\n' +
                                           f'∫ 1/({a}x + 1) dx = 1/{a}·ln|{a}x + 1| + C = {coef} ln|{a}x + 1| + C',
                        'hint': '∫ 1/(ax+b) dx = 1/a·ln|ax+b| + C',
                        'difficulty': difficulty,
                        'topic': '积分'
                    })

            else:  # differential_equations
                # 微分方程题目
                de_types = ['separable', 'first_order', 'second_order']
                de_type = random.choice(de_types)

                if de_type == 'separable':
                    a = random.randint(1, 3)
                    questions.append({
                        'id': i + 1,
                        'question': f'解微分方程：dy/dx = {a}xy',
                        'answer': f'y = Ce^({a}x²/2)',
                        'detailed_answer': f'1. 分离变量：dy/y = {a}x dx\n\n' +
                                           f'2. 两边积分：∫ dy/y = ∫ {a}x dx\n\n' +
                                           f'3. 得到：ln|y| = {a}x²/2 + C₁\n\n' +
                                           f'4. 解出：y = Ce^({a}x²/2)，其中 C = ±e^C₁',
                        'hint': '分离变量法：dy/y = ax dx，两边积分',
                        'difficulty': difficulty,
                        'topic': '微分方程'
                    })
                elif de_type == 'first_order':
                    a = random.randint(1, 3)
                    questions.append({
                        'id': i + 1,
                        'question': f'解微分方程：dy/dx + {a}y = 0',
                        'answer': f'y = Ce^(-{a}x)',
                        'detailed_answer': f'1. 这是一阶线性齐次微分方程\n\n' +
                                           f'2. 分离变量：dy/y = -{a} dx\n\n' +
                                           f'3. 两边积分：ln|y| = -{a}x + C₁\n\n' +
                                           f'4. 解出：y = Ce^(-{a}x)',
                        'hint': '一阶线性齐次微分方程',
                        'difficulty': difficulty,
                        'topic': '微分方程'
                    })
                else:
                    a = random.randint(2, 3)
                    b = random.randint(1, 2)
                    # 计算特征方程的根
                    discriminant = a * a - 4 * b
                    if discriminant > 0:
                        r1 = round((-a + math.sqrt(discriminant)) / 2, 2)
                        r2 = round((-a - math.sqrt(discriminant)) / 2, 2)
                        solution = f'y = C₁e^{r1}x + C₂e^{r2}x'
                    elif discriminant == 0:
                        r = round(-a / 2, 2)
                        solution = f'y = (C₁ + C₂x)e^{r}x'
                    else:
                        real = round(-a / 2, 2)
                        imag = round(math.sqrt(-discriminant) / 2, 2)
                        solution = f'y = e^{real}x(C₁cos({imag}x) + C₂sin({imag}x))'

                    questions.append({
                        'id': i + 1,
                        'question': f'解微分方程：d²y/dx² + {a}dy/dx + {b}y = 0',
                        'answer': solution,
                        'detailed_answer': f'1. 写出特征方程：r² + {a}r + {b} = 0\n\n' +
                                           f'2. 求解特征方程：\n' +
                                           f'   r = [-{a} ± √({a}² - 4×{b})]/2\n' +
                                           f'   = [-{a} ± √({a * a - 4 * b})]/2\n\n' +
                                           f'3. 根据根的情况得到通解：\n' +
                                           f'   {solution}',
                        'hint': '特征方程法：r² + ar + b = 0',
                        'difficulty': difficulty,
                        'topic': '微分方程'
                    })

        return jsonify({
            'success': True,
            'questions': questions,
            'count': len(questions)
        })

    except Exception as e:
        logger.error(f"生成练习题失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/check-tesseract', methods=['GET'])
def check_tesseract():
    status = {
        'success': True,
        'tesseract_available': TESSERACT_AVAILABLE,
        'latex_ocr_available': LATEX_OCR_AVAILABLE,
        'deepseek_available': DEEPSEEK_AVAILABLE and os.getenv('DEEPSEEK_API_KEY') is not None,
        'path': TESSERACT_PATH if os.path.exists(TESSERACT_PATH) else None
    }
    if TESSERACT_AVAILABLE:
        try:
            status['version'] = str(pytesseract.get_tesseract_version())
            status['languages'] = pytesseract.get_languages(config='')
        except:
            pass
    return jsonify(status)


# 初始化服务
try:
    initialize_services()
except Exception as e:
    logger.error(f"启动失败: {str(e)}")


if __name__ == '__main__':
    port = int(os.getenv('APP_PORT', 5656))

    print("\n" + "=" * 60)
    print("🚀 高数辅导助手 - 复杂数学OCR增强版（集成LaTeX-OCR + DeepSeek）")
    print("=" * 60)
    print(f"📁 上传目录: {app.config['UPLOAD_FOLDER']}")
    print(f"🔍 Tesseract: {'✅ 已安装' if TESSERACT_AVAILABLE else '❌ 未安装'}")
    print(f"   📍 路径: {TESSERACT_PATH}")
    print(f"   📍 路径存在: {'✅' if os.path.exists(TESSERACT_PATH) else '❌'}")
    print(f"🔬 LaTeX-OCR: {'✅ 已安装' if LATEX_OCR_AVAILABLE else '❌ 未安装'}")
    print(f"🤖 DeepSeek: {'✅ 已配置' if DEEPSEEK_AVAILABLE and os.getenv('DEEPSEEK_API_KEY') else '❌ 未配置'}")

    if TESSERACT_AVAILABLE:
        try:
            languages = pytesseract.get_languages(config='')
            print(f"   📍 语言包: {len(languages)} 个")
            print(f"   📍 中文: {'✅' if 'chi_sim' in languages else '❌'}")
            print(f"   📍 英文: {'✅' if 'eng' in languages else '❌'}")
            print(f"   📍 数学: {'✅' if 'equ' in languages else '❌'}")
        except:
            pass

    print("-" * 60)
    print("🔥 专门优化: 高等数学选择题识别")
    print("   - 4倍图像放大 + 自适应阈值二值化 + 形态学连接")
    print("   - 多预处理 + 多PSM + 多语言组合 (Tesseract)")
    print("   - 专用公式识别 (LaTeX-OCR)")
    print("   - 🤖 DeepSeek大模型智能融合与修复")
    print("   - 增强后处理（上下标、符号修复）")
    print("-" * 60)
    print(f"🌐 访问地址: http://localhost:{port}")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
