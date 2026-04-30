import os
from http import HTTPStatus

import dashscope
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient
import numpy as np


# 连接到本地 Milvus 实例
# embedding = OpenAIEmbeddings(model="text-embedding-3-small")
# client = MilvusClient(alias="default", host="localhost", port="19530", token="root:fridge_zzl")
# client.create_collection(collection_name="test", dimension=1536, consistency_level="Strong", metric_type="COSINE",auto_id=True)
# res = client.list_indexes(collection_name="test")
# print(res)

text = """
春天像一个美丽的仙女，悄无声息地来到我身边，她给了我一个青春的梦，一个理想的梦，一个未来的梦。

小草从土地里钻出来，嫩绿的叶片在阳光下闪闪发亮。它迎着微风轻轻摇摆，好像在告诉我，春天是充满希望的季节。大树伯伯伸展着枝叶繁茂的枝头，守护着新生的嫩芽，无论风雨多大，他都坚定地保护它们成长。抬头望去，白云妹妹在蓝天的怀抱中轻盈飘动，她不再留恋冬天的寒冷，而是欢快地迎接春天的温暖。大地爸爸默默地为万物提供养分，让一切生机勃勃。春仙女看到这春意盎然的景象，脸上露出欣慰的笑容，她决定多停留些日子，和大家一起创造一个多彩缤纷的世界。

春仙女的到来，也提醒我新学期开始了。我已经初二了，却还漫不经心地过日子。但春天的景象让我醒悟：小草教会我乐观向上，大树教我坚强守护，白云教我拥抱变化，大地教我默默奉献。春仙女轻声对我说：“春天就像青春，你要珍惜它，有目标，有理想，努力向前。”

春天带给了我一个青春的梦，一个理想的梦，一个未来的梦。我仿佛看到自己努力学习的身影，未来在向我招手。小草冲我微笑：“加油，我相信你！”大树竖起大拇指：“成功就在前方。”白云拉着我的手：“青春短暂，要努力抓住。”大地严肃地说：“青春需要干劲！”春仙女温柔地鼓励：“相信自己，未来并不遥远。”

春天带给了我一个青春的梦，一个理想的梦，一个未来的梦，一个努力的梦，一个充实的梦。
"""
api_key = os.environ["DASHSCOPE_API_KEY"]
class MilvusOperation(object):
    def __init__(self):
        self.client = MilvusClient(host="localhost", port="19530", token="root:fridge_zzl")
        self.embedding = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed_with_str(self,text):
        resp = dashscope.TextEmbedding.call(
            model=dashscope.TextEmbedding.Models.text_embedding_v1,
            api_key=api_key,
            input=text)
        if resp.status_code == HTTPStatus.OK:
            return resp["output"]["embeddings"][0]["embedding"]

    def create_table(self):
        self.client.create_collection(collection_name="test", dimension=1536, consistency_level="Strong", metric_type="COSINE",auto_id=True)
        data_list = []
        for txt in text.split("\n"):
            vector = self.embed_with_str(txt)
            data = {"vector": vector,"text": txt}
            data_list.append(data)
            print(data)

        self.client.insert(collection_name="test", data=data_list)

    def search(self):
        # 假设你已经有一个查询向量vector_query

        # results将包含最相似的记录的ID
        # print(results)
        pass
if __name__ == "__main__":
    a = MilvusOperation()
    a.search()