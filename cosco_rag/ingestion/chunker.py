from langchain_text_splitters import RecursiveCharacterTextSplitter

from cosco_rag.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_parent_child(text: str):
    """
    对长文本进行父子切片：
    - 父块：较大的章节级块（chunk_size*2）
    - 子块：普通段落级块（chunk_size），用于精细检索
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 2,
        chunk_overlap=CHUNK_OVERLAP * 2,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    parent_chunks = parent_splitter.split_text(text)
    child_chunks = []
    for p_chunk in parent_chunks:
        child_chunks.extend(child_splitter.split_text(p_chunk))
    return parent_chunks, child_chunks