from cosco_rag.retrieval.search import RAGService
import numpy as np

rag = RAGService()  # 会加载您的 Qwen3-VL-Embedding 模型

texts = [
    "SHANGHAI",
    "中国香港",
    "COSCO SHIPPING STAR",
    "5G通讯基站设备",
    "集装箱船",
    "散货船"
]

vecs = [rag.embeddings.embed_query(t) for t in texts]

# 检查不同文本向量的长度和相似度
for i in range(len(vecs)):
    for j in range(i+1, len(vecs)):
        cosine = np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]))
        print(f"「{texts[i]}」 vs 「{texts[j]}」: cosine = {cosine:.4f}")

vecs2 = [rag.embeddings.embed_query("船名 COSCO BELGIUM，IMO编号 9512345，船籍 中国香港，船型 集装箱船，建造于 2018 年，总吨位 140000 吨，载重吨 150000 吨，船舶尺寸：长 366 米，宽 48 米，吃水 15.2 米，主机型号 MAN B&W 10S90ME，所有人 COSCO Shipping，管理公司 COSCO Shipping Mgmt，下次坞修日期为 2025-06-30 00:00:00，船级社 CCS，证书到期日 2025-09-01 00:00:00。"),
        rag.embeddings.embed_query("新加坡"),
         rag.embeddings.embed_query("船名 COSCO HOPE，IMO编号 9423457，船籍 新加坡，船型 散货船，建造于 2015 年，总吨位 43000 吨，载重吨 82000 吨，船舶尺寸：长 229 米，宽 32 米，吃水 13.8 米，主机型号 Daihatsu 8DKM-36，所有人 COSCO Bulk，管理公司 COSCO Bulk Mgmt，下次坞修日期为 2025-03-15 00:00:00，船级社 LR，证书到期日 2025-07-20 00:00:00。"),
         rag.embeddings.embed_query("新加坡"),]

cosine = np.dot(vecs2[0], vecs2[1])
print(f"cosine = {cosine:.4f}")

cosine = np.dot(vecs2[2], vecs2[1])
print(f"cosine = {cosine:.4f}")

cosine = np.dot(vecs2[3], vecs2[1])
print(f"cosine = {cosine:.4f}")