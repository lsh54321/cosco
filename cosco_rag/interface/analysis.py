import os
from http import client

import torch
from paddlex import create_pipeline
from pydantic import SecretStr
from pymilvus import MilvusClient, connections, Collection, FieldSchema, DataType, CollectionSchema

from cosco_rag.utils.configManager import get_config

cfg = get_config()

api_url = cfg.get('ocr.api_url')
model = cfg.get('llm.model')
log_level = cfg.get('logging.level')

# 图片解析
# pipeline = create_pipeline(pipeline="OCR")
# pipeline.device=torch.device("cpu")
# output = pipeline.predict("./doc/部门甘特图")
# for res in output:
#     res.print()
#     res.save_to_img("./output/")
#     res.save_to_json("./output/")

# 表格解析
import pandas as pd

# 最简单的用法：读取Excel文件的第一个工作表
df = pd.read_excel('../../doc/container_shipping.xlsx')
# 合并单元格
df_cleaned = df.ffill()
# 单元格集合
column_names = df_cleaned.columns.tolist()

for idx, row in df_cleaned.iterrows():
    # 1. 生成“行级”信息流，以“列名: 值”的形式呈现，并融入表描述
    details = [f"{col}: {value}" for col, value in zip(column_names, row)]
    content_for_llm = {
        "type": "table_row",
        "source": "container_shipping.xlsx",
        "row_index": idx,
        "data_str": f"这是一份港口运价数据，其中一条记录为：{'，'.join(details)}"
    }
    print(content_for_llm)
    # 向量化存储

import pandas as pd
import sqlite3
import re

api_key = os.environ["DASHSCOPE_API_KEY"]


def excel_to_sqlite(excel_path: str, db_path: str = "cargo.db", table_name: str = "rates"):
    """
    读取Excel（支持多级表头、合并单元格），清洗后存入SQLite。
    """
    # 读取Excel，指定多级表头（例如前两行是列头）
    df = pd.read_excel(excel_path, header=[0, 1])  # 根据实际层级调整

    # 展开多级列头（例如将 ("航线","起运港") 合并为 "航线_起运港"）
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # 处理合并单元格造成的NaN – 向前填充
    df = df.ffill()

    # 列名规范化（去掉特殊字符，转小写）
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col).lower() for col in df.columns]

    # 连接数据库
    conn = sqlite3.connect(db_path)

    # 导入数据
    df.to_sql(table_name, conn, if_exists='replace', index=False)

    # 打印表结构示例
    print("导入成功！表结构如下：")
    print(pd.read_sql(f"PRAGMA table_info({table_name})", conn))

    return conn


def get_schema(conn, table_name):
    """获取表的列名、类型以及示例数据（帮助 LLM 理解字段含义）"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()  # (cid, name, type, notnull, dflt_value, pk)
    col_names = [col[1] for col in columns]

    # 取前3行作为示例数据
    sample_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 3", conn)
    sample_str = sample_df.to_string(index=False)

    schema_text = f"表名: {table_name}\n列: {', '.join(col_names)}\n示例数据:\n{sample_str}"
    return schema_text


def build_sql_prompt(user_question: str, schema_text: str) -> str:
    prompt = f"""你是一个专业的航运数据分析师。请根据用户的问题，生成一个能够从下方数据库中查出答案的SQL查询。

数据库schema：
{schema_text}

用户问题：{user_question}

要求：
- 只输出SQL语句，不要输出多余解释。
- 如果问题不明确，返回一个最合理的推测性查询。
- 使用标准SQLite语法。

SQL：
"""
    return prompt


import requests
import json


def generate_sql(user_question, schema_text, model="qwen2.5-72b-instruct"):
    prompt = build_sql_prompt(user_question, schema_text)

    # 示例使用阿里云 DashScope API（通义千问）
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"

    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": prompt}]},
        "parameters": {"result_format": "message", "temperature": 0}
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()["output"]["choices"][0]["message"]["content"]
    else:
        raise Exception(f"LLM调用失败: {response.text}")


def answer_from_sql(conn, user_question, schema_text):
    # 1. 生成 SQL
    raw_sql = generate_sql(user_question, schema_text)

    # 可选：清洗 SQL（去掉 markdown 代码块标记）
    if raw_sql.startswith("```sql"):
        raw_sql = raw_sql.replace("```sql", "").replace("```", "").strip()

    print(f"[生成的SQL] {raw_sql}")

    # 2. 执行查询
    try:
        result_df = pd.read_sql(raw_sql, conn)
        return result_df
    except Exception as e:
        # 如果 SQL 执行失败，可以重新让 LLM 修正（此处简化）
        return f"SQL执行错误: {e}\n生成的SQL: {raw_sql}"


# 使用
# conn = excel_to_sqlite("../../doc/部门OKR与周报.xlsx", "cargo.db", "freight_rates")
#
# schema = get_schema(conn, "freight_rates")
#
# # 3. 用户提问
# question = "从上海到洛杉矶，COSCO的40HQ集装箱运价是多少？"
#
# # 4. 执行 Text-to-SQL 流程
# answer_df = answer_from_sql(conn, question, schema)
#
# print("\n====== 查询结果 ======")
# print(answer_df)

# 文档解析
# import numpy as np
# from langchain_paddleocr import PaddleOCRVLLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_milvus import Milvus
#
# loader = PaddleOCRVLLoader(file_path="../../doc/面.pdf",
#                            api_url="https://lbgdi0w0p3x5h4xa.aistudio-app.com/layout-parsing",
#                            access_token=SecretStr("cf078cc027025fb478b3700e433265c1355a87bf"))
# docs = loader.load()
#
# # 智能分块：根据版面，优先保证段落、表格完整
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
# chunks = splitter.split_documents(docs)
# embeddings = OpenAIEmbeddings()
#
# # 1. 手动建立连接
# connections.connect(alias="default", host='192.168.1.37', port='19530')
# # 步骤 2: 创建 Schema 和 Collection（这会多次与服务器通信，因此必须先连接）
# fields = [
#     FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
#     FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
#     FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536)
# ]
# schema = CollectionSchema(fields, "OCR文本块")
# collection = Collection("pdf_chunks3", schema)
#
# texts = [chunk.page_content for chunk in chunks]
#
# vectors = embeddings.embed_documents(texts)
# collection.insert([
#     texts,  # ✅ 文本字段是字符串列表
#     vectors                               # ✅ 向量字段是浮点数列表的列表
# ])
#
# index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
# collection.create_index("vector", index_params)
# collection.load()
# print(f"成功将 {len(chunks)} 个文本块存入 Milvus")
