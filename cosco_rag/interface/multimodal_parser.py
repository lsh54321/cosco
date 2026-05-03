import base64
import json
import re
from datetime import datetime
from typing import Dict, Any
from PIL import Image

from pymilvus.orm import utility

from cosco_rag import config
from cosco_rag.config import SMALL_EMBEDDING_DIM

# ---------- 配置 ----------
USE_OLLAMA = True  # 设为 False 则使用 Transformers
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL_NAME = "Qwen3-VL:2B-Instruct"  # Ollama 中的模型名, 确保已安装

# ---------- Milvus 相关导入 ----------
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer

# ---------- 2.1 Ollama 调用 ----------
async def parse_image_ollama(image_path: str) -> Dict[str, Any]:
    """通过 Ollama 的 OpenAI 兼容 API 解析图片"""
    import httpx
    # 读取图片并转为 base64
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    prompt = """
    你是中远海运订舱单证识别专家。请仔细分析这张托书图片，提取以下信息，输出 JSON 格式：
    {
        "shipper": "发货人名称",
        "consignee": "收货人名称",
        "goods_name": "货物名称",
        "hs_code": "HS编码（10位数字）",
        "weight_kg": 毛重（数字，单位公斤）,
        "container_type": "箱型，如 40HQ",
        "port_of_loading": "起运港",
        "destination": "目的港",
        "confidence": 0.0~1.0  # 整体可信度
    }
    如果某个字段找不到，设为 null。不要输出额外解释。
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


# ---------- 2.2 Transformers 调用（备用） ----------
def load_transformers_model():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    return model, processor


def parse_image_transformers(image_path: str) -> Dict[str, Any]:
    model, processor = load_transformers_model()
    image = Image.open(image_path)
    prompt = "提取托书中的发货人、收货人、品名、HS编码、毛重、箱型、起运港、目的港，输出JSON。"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text, images=image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    # 提取 JSON 部分
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    return json.loads(json_match.group())


def ensure_milvus_collection():
    """确保历史提单集合存在，若不存在则创建"""
    try:
        connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        if not utility.has_collection("historical_bookings"):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=SMALL_EMBEDDING_DIM)
            ]
            schema = CollectionSchema(fields, description="历史订舱托书解析记录")
            collection = Collection("historical_bookings", schema)
            # 创建向量索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("embedding", index_params)
            print(f"Milvus 集合 historical_bookings 创建成功")
        else:
            collection = Collection("historical_bookings")
        return collection
    except Exception as e:
        print(f"Milvus 连接/创建失败: {e}")
        return None

def insert_parsed_to_milvus(parsed: Dict[str, Any], image_path: str, content_text: str = None):
    """将解析结果存入 Milvus"""
    if content_text is None:
        # 生成自然语言描述
        content_text = (f"发货人：{parsed.get('shipper', '')}，"
                        f"收货人：{parsed.get('consignee', '')}，"
                        f"品名：{parsed.get('goods_name', '')}，"
                        f"HS编码：{parsed.get('hs_code', '')}，"
                        f"毛重：{parsed.get('weight_kg', '')} kg，"
                        f"箱型：{parsed.get('container_type', '')}，"
                        f"起运港：{parsed.get('port_of_loading', '')}，"
                        f"目的港：{parsed.get('destination', '')}")
    try:
        embedding = config.embeddings
        # 将文本转为向量（返回 list of float）
        vector = embedding.embed_query(content_text)
        collection = Collection("historical_bookings")
        collection.insert([
            [content_text],
            [image_path],
            [vector]
        ])
        collection.flush()
        print(f"✅ 解析结果已存入 Milvus (collection: historical_bookings)")
    except Exception as e:
        print(f"⚠️ Milvus 入库失败: {e}")


# ---------- 2.1 Ollama 调用 ----------
async def parse_image_ollama(image_path: str) -> Dict[str, Any]:
    """通过 Ollama 的 OpenAI 兼容 API 解析图片"""
    import httpx
    # 读取图片并转为 base64
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    prompt = """
    你是中远海运订舱单证识别专家。请仔细分析这张托书图片，提取以下信息，输出 JSON 格式：
    {
        "shipper": "发货人名称",
        "consignee": "收货人名称",
        "goods_name": "货物名称",
        "hs_code": "HS编码（10位数字）",
        "weight_kg": 毛重（数字，单位公斤）,
        "container_type": "箱型，如 40HQ",
        "port_of_loading": "起运港",
        "destination": "目的港",
        "confidence": 0.0~1.0  # 整体可信度
    }
    如果某个字段找不到，设为 null。不要输出额外解释。
    """
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_data}"}}
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 1024,
        "response_format": {"type": "json_object"}
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


# ---------- 2.2 Transformers 调用（备用） ----------
def load_transformers_model():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    return model, processor


def parse_image_transformers(image_path: str) -> Dict[str, Any]:
    model, processor = load_transformers_model()
    image = Image.open(image_path)
    prompt = "提取托书中的发货人、收货人、品名、HS编码、毛重、箱型、起运港、目的港，输出JSON。"
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text, images=image, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    result = processor.decode(outputs[0], skip_special_tokens=True)
    # 提取 JSON 部分
    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    return json.loads(json_match.group())

async def extract_booking_from_image(image_path: str, save_to_milvus: bool = True) -> Dict[str, Any]:
    """统一入口，解析并可选存入 Milvus"""
    if USE_OLLAMA:
        parsed = await parse_image_ollama(image_path)
    else:
        parsed = parse_image_transformers(image_path)
    if save_to_milvus:
        # 确保 Milvus 集合存在
        ensure_milvus_collection()
        insert_parsed_to_milvus(parsed, image_path)
    return parsed