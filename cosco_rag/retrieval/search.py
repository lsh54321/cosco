from cosco_rag import config
from openai import OpenAI
from langchain_milvus import Milvus
from openai.types.chat import ChatCompletionUserMessageParam
from sentence_transformers import CrossEncoder


from cosco_rag.storage.connection import get_connection

class RAGService:
    def __init__(self):
        # 加载重排序模型（轻量）
        self.reranker = CrossEncoder('D:/modelscope/hub/models/cross-encoder/ms-marco-TinyBERT-L2-v2', max_length=512)
        # 嵌入模型同样使用 Qwen3-Embedding-8B
        self.embeddings = config.embeddings
        # 答案生成 LLM（可继续用 OpenAI 或换成 Qwen 的对话模型，这里以保持简单为例）
        self.llm_client = OpenAI(
            base_url="http://localhost:11434/v1",  # Ollama 的 OpenAI 兼容端点
            api_key="ollama"  # 任意非空字符串
        )
        # 如果生成模型和嵌入模型不同，请自行分离配置；此处为演示统一用硅基流动的 key

    def rerank(self, query, docs_with_score):
        # 构造(问题, 文档)对
        pairs = [(query, doc.page_content) for doc, _ in docs_with_score]
        # 预测相关性分数
        scores = self.reranker.predict(pairs)
        # 按分数降序排序，取索引
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        reranked_docs = [docs_with_score[i][0] for i in sorted_indices]
        return reranked_docs

    def answer(self, question: str, business_line: str, top_k: int = 15) -> str:
        print(business_line)

        # 1. 向量化问题
        # Create Milvus vector store
        vector_store = Milvus(
            embedding_function=self.embeddings,
            connection_args={"uri": "http://localhost:19530"},
            vector_field="vector",                  # 修正：字符串，不是列表
            collection_name=business_line,
            # consistency_level="Strong",    # Strong consistency level, default is "Session"
            auto_id=True,
            # drop_old=True,  # If you want to drop old collection and create a new one
        )

        # 2. 直接进行相似度搜索
        docs_with_score = vector_store.similarity_search_with_score(question, k=2)
        # 重排序
        reranked_docs = self.rerank(question, docs_with_score)
        top_docs = reranked_docs[:3]
        context = "\n".join([doc.page_content for doc in top_docs])\

        # 3. 组装 Prompt 并生成答案
        prompt = f"""你是一个中远海运船舶管理助手。请严格根据以下参考资料回答用户问题。

        用户问题：{question}

        要求：
        - 只回答与问题直接相关的内容，不要列出无关数据。
        - 如果参考资料中没有符合条件的船，请明确说明“未找到符合条件的船”。

        参考资料：
        {context}

        回答："""

        print(prompt)
        messages: ChatCompletionUserMessageParam = {"role": "user", "content": prompt}
        response = self.llm_client.chat.completions.create(
            model='Qwen3-VL:2B-Instruct',  # 你本地的 Ollama 模型名
            messages=[messages],
        ) # type: ignore
        # return response.message.content
        print(response.choices[0].message.content)

get_connection()
rag = RAGService()
rag.answer("船籍是利比里亚船有哪些","ship_management")

