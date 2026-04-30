from pymilvus import connections
from cosco_rag.config import MILVUS_HOST, MILVUS_PORT

_connected = False

def get_connection(alias="default"):
    """建立Milvus连接（单例），解决'ConnectionNotExistException'"""
    global _connected
    if not _connected:
        connections.connect(
            alias=alias,
            host=MILVUS_HOST,
            port=MILVUS_PORT
        )
        _connected = True
        print(f"✅ Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")
    return connections