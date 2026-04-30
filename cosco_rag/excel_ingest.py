"""
读取6业务线Excel表格，将每行转为语义描述文本，
调用已封装的EmbeddingService存储至Milvus。
"""
import re
from pymilvus import utility
import pandas as pd
from storage.connection import get_connection
from storage.schema import create_collections
from storage.index import create_indexes
from retrieval.loader import load_all_collections
from ingestion.vectorizer import EmbeddingService
from config import COMPLIANCE_INFO

# ──────────────────────────────────────────────
# 1. 定义每个业务线的“行 → 自然语言”模板
#    模板中的 {字段} 会被实际数据替换
# ──────────────────────────────────────────────
TEMPLATES = {
    "sensitive_goods": (
        "敏感品名“{品名}”属于【{风险等级}】风险等级。"
        "订舱前必须提供：{所需文件}。{备注}"
    ),
    "port_codes": (
        "中文港口“{中文港口名}”（标准名称：{标准名称}）的 UN/LOCODE 代码为 {UN/LOCODE}，"
        "所属国家 {所属国家}，常用箱型有 {常用箱型}。"
    ),
    "historical_bookings": (
        "历史提单 {提单号}：发货人 {发货人}，收货人 {收货人}。"
        "品名：{品名}，HS编码：{HS编码}。"
        "起运港 {起运港代码} 至 目的港 {目的港代码}，船名航次 {船名航次}。"
        "毛重 {毛重(kg)} kg，箱型 {箱型}。提交日期 {提交日期}，当前状态 {状态}。"
    ),
    "sop_rules": (
        "【{规则ID}】适用场景“{适用场景}”：规则描述：{规则描述} "
        "优先级：{优先级}（涉及品名/HS：{相关品名/HS}）"
    ),
    "hscodes": (
        "HS编码 {HS编码} 对应商品“{商品描述}”。"
        "监管条件：{监管条件}，危险品类别：{危险品类别}。备注：{备注}"
    ),
}
def clean_column_name(col: str) -> str:
    """
    清洗列名：
    1. 去掉首尾空格
    2. 中文括号转英文括号
    3. 去除左括号前、左括号后、右括号前的所有空格
    """
    col = col.strip()
    col = col.replace("（", "(").replace("）", ")")
    # 去掉 '(' 左边的空格
    col = re.sub(r'\s*\(\s*', '(', col)
    # 去掉 ')' 左边的空格
    col = re.sub(r'\s*\)', ')', col)
    return col

def row_to_text(row: pd.Series, template: str) -> str:
    """将一行Series格式化为自然语言描述"""
    return template.format(**row.to_dict())

def process_excel_single_file(filepath: str, biz: str, embed_service: EmbeddingService):
    """从单个Excel文件（一个Sheet）读取并入库"""
    df = pd.read_excel(filepath)
    df = df.fillna("")  # 空白单元格填 ""
    df.columns = df.columns.str.strip()  # 仅去掉首尾空格，清洗一次
    # 用清洗函数重命名所有列
    df.rename(columns=lambda x: clean_column_name(x), inplace=True)
    texts = []
    template = TEMPLATES[biz]
    for _, row in df.iterrows():
        texts.append(row_to_text(row, template))
    if texts:
        embed_service.embed_and_insert(texts, biz, source_file=filepath)
        print(f"✅ {biz}：已插入 {len(texts)} 条记录")

def process_excel_multi_sheet(filepath: str, embed_service: EmbeddingService):
    """从一个Excel文件中的多个Sheet读取（Sheet名=业务线名）"""
    xls = pd.ExcelFile(filepath)
    for sheet_name in xls.sheet_names:
        if sheet_name not in TEMPLATES:
            print(f"⚠️ 跳过未知业务线 Sheet: {sheet_name}")
            continue
        df = pd.read_excel(filepath, sheet_name=sheet_name).fillna("")
        texts = []
        template = TEMPLATES[sheet_name]
        for _, row in df.iterrows():
            texts.append(row_to_text(row, template))
        if texts:
            embed_service.embed_and_insert(texts, sheet_name, source_file=filepath)
            print(f"✅ {sheet_name}：已插入 {len(texts)} 条记录")

def main():
    # 1. 连接 Milvus + 确保 Collection 存在
    get_connection()
    for biz in COMPLIANCE_INFO:
        if utility.has_collection(biz):
            utility.drop_collection(biz)
            print(f"已删除 Collection: {biz}")
    create_collections()

    # 2. 初始化嵌入服务（这里用的 bge-small-zh-v1.5，已在 .vectorizer 中使用 HuggingFaceEmbeddings）
    embed_service = EmbeddingService()

    # 3. 根据实际情况选择读取方式
    # 方式A：每个业务线一个Excel文件（假设放在 data_excel/ 目录下，文件名=业务线名.xlsx）
    import os
    excel_dir = "../doc"
    for biz in COMPLIANCE_INFO:
        file_path = os.path.join(excel_dir, f"{biz}.xlsx")
        if os.path.exists(file_path):
            process_excel_single_file(file_path, biz, embed_service)
        else:
            print(f"⚠️ 未找到 {biz}.xlsx，跳过")

    # 方式B：单个Excel文件含多个Sheet（Sheet名即业务线名）
    # single_file = "所有业务线模拟数据.xlsx"
    # if os.path.exists(single_file):
    #    process_excel_multi_sheet(single_file, embed_service)

    # 4. 创建索引 + 加载
    create_indexes()
    load_all_collections()
    print("🎉 全部Excel数据入库完成！")

if __name__ == "__main__":
    main()