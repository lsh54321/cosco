from pymilvus import Collection
from cosco_rag.config import COMPLIANCE_INFO

def create_indexes():
    """为所有业务线的向量字段创建IVF_FLAT索引"""
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    for line in COMPLIANCE_INFO:
        collection = Collection(line)
        if not collection.has_index():
            collection.create_index(field_name="vector", index_params=index_params)
            print(f"✅ 索引创建完成: {line}")