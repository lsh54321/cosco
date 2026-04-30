import fitz  # PyMuPDF
import os
from paddleocr import PaddleOCR

# 全局实例化一次，避免重复加载模型
ocr = PaddleOCR(use_angle_cls=True, lang="ch")

def extract_text(pdf_path: str) -> str:
    """
    提取PDF文本内容：
    - 优先使用PyMuPDF的文字层
    - 若页面无文字（扫描件），则调用PaddleOCR识别
    """
    doc = fitz.open(pdf_path)
    full_text = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            full_text.append(text)
        else:
            # 扫描版页面，转为图片OCR
            pix = page.get_pixmap(dpi=200)
            img_path = f"temp_page_{page_num}.png"
            pix.save(img_path)
            result = ocr.ocr(img_path, cls=True)
            if result and result[0]:
                page_text = " ".join([line[1][0] for line in result[0]])
            else:
                page_text = ""
            full_text.append(page_text)
            os.remove(img_path)

    return "\n".join(full_text)