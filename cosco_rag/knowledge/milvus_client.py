from pymilvus import connections, Collection
from cosco_rag import config
def get_collection(name: str):
    connections.connect(host=config.MILVUS_HOST, port=config.MILVUS_PORT)
    return Collection(name)

# ------------------- 5 个 RAG 知识库对应的检索函数 -------------------
def search_sensitive_goods(product_name: str):

    collection = get_collection("sensitive_goods")
    vec = config.embeddings.embed_query(product_name)

    # 使用实际字段名进行搜索
    res = collection.search(
        data=[vec],
        anns_field='vector',  # 关键修改点
        param={"metric_type": "COSINE"},
        limit=10
    )
    if res and res[0]:
        hit = res[0][0]
        return {
            "is_sensitive": hit.score > 0.8,
            "risk_level": hit.entity.get("风险等级"),
            "required_docs": hit.entity.get("所需文件")
        }
    return {"is_sensitive": False}

def search_port_code(port_name: str):
    # 此处简化，实际可向量检索或精确匹配
    mapping = {"上海": "CNSHA", "宁波": "CNNGB", "洛杉矶": "USLAX"}
    return mapping.get(port_name, port_name)

def search_hs_code(hs_code: str):
    # 从 hscodes 集合查询商品描述
    coll = get_collection("hscodes")
    # 实际可用向量检索或过滤查询，这里简单返回固定值
    desc_map = {"85076000": "锂离子蓄电池", "38021000": "活性炭"}
    return desc_map.get(hs_code, "未知商品")