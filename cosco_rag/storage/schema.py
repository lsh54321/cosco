from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
from cosco_rag.config import COMPLIANCE_INFO, EMBEDDING_DIM,SMALL_EMBEDDING_DIM

def create_collections():
    """为每条业务线创建独立的Milvus Collection"""
    for line in COMPLIANCE_INFO:
        if not utility.has_collection(line):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=SMALL_EMBEDDING_DIM)
            ]
            schema = CollectionSchema(fields, description=f"中远海运知识库 - {line}")
            Collection(name=line, schema=schema)
            print(f"✅ 创建Collection: {line}")
        else:
            print(f"⚠️ Collection已存在: {line}")