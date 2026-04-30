import os
from typing import Dict, List

def scan_business_pdfs(base_dir: str = "data") -> Dict[str, List[str]]:
    """
    扫描data目录，返回 业务线名 -> [PDF文件路径列表] 的字典。
    目录结构：
        data/
        ├── container_shipping/   (放该业务线所有PDF)
        ├── port_logistics/
        └── ...
    """
    business_pdfs = {}
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            pdf_files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(".pdf")
            ]
            if pdf_files:
                business_pdfs[folder] = pdf_files
    return business_pdfs