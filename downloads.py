#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen3-VL-Embedding-2B',cache_dir="D:\modelscope\hub\models")