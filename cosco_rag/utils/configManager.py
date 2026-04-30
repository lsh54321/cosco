import os
import yaml
from dotenv import load_dotenv

# 加载顶层的 .env 文件，该文件存放真实的 API Key
load_dotenv()

class ConfigManager:
    def __init__(self, env: str = "development"):
        """env: development, staging, production"""
        self.env = env
        self.config = {}

        # 1. 加载 base.yaml
        base_path = "../../configs/base.yaml"
        self._load_yaml(base_path)

        # 2. 加载环境特定配置文件
        env_path = f"../../configs/{env}.yaml"
        self._load_yaml(env_path)

        # 3. 解析环境变量占位符 (例如 'env:DEV_LANGSMITH_API_KEY')
        self._resolve_env_placeholders(self.config)

    def _load_yaml(self, path: str):
        """加载 YAML 文件并合并到主配置中"""
        with open(path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            if config_data:
                self._deep_merge(config_data, self.config)

    def _deep_merge(self, src, dest):
        """递归合并配置，实现配置继承"""
        for key, value in src.items():
            if key in dest and isinstance(dest[key], dict) and isinstance(value, dict):
                self._deep_merge(value, dest[key])
            else:
                dest[key] = value

    def _resolve_env_placeholders(self, obj):
        """将配置中的 'env:VAR_NAME' 替换为实际的环境变量值"""
        if isinstance(obj, dict):
            for k, v in obj.items():
                obj[k] = self._resolve_env_placeholders(v)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                obj[i] = self._resolve_env_placeholders(item)
        elif isinstance(obj, str) and obj.startswith('env:'):
            env_var = obj[4:]
            return os.getenv(env_var, '')
        return obj

    def get(self, key_path: str, default=None):
        """通过点分隔的路径获取配置，如 'langsmith.api_key'"""
        keys = key_path.split('.')
        val = self.config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
                if val is None:
                    return default
            else:
                return default
        return val

# 全局配置实例，在程序初始化时调用
CONFIG = ConfigManager(env=os.getenv("APP_ENV", "development"))