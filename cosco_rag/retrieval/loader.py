from pymilvus import Collection
from cosco_rag.config import COMPLIANCE_INFO

def load_all_collections():
    """将所有业务线Collection加载到内存，用于检索"""
    for line in COMPLIANCE_INFO:
        collection = Collection(line)
        collection.load()
        print(f"🚀 已加载Collection: {line}")