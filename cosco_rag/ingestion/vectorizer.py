from typing import List

from cosco_rag import config
from pymilvus import Collection

class EmbeddingService:
    def __init__(self):
        self.embeddings = config.embeddings

    def embed_and_insert(self, chunks: List[str], business_line: str, source_file: str):
        if not chunks:
            return

        # 注意：chunks 必须是纯字符串列表，不能是 Document 对象
        vectors = self.embeddings.embed_documents(chunks)

        insert_data = [
            chunks,
            [source_file] * len(chunks),
            vectors
        ]

        collection = Collection(business_line)
        collection.insert(insert_data)
        collection.flush()
        print(f"📥 插入 {len(chunks)} 条向量至 {business_line}（来源：{source_file}）")