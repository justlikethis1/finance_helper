import asyncio
import concurrent.futures
import time
from typing import Optional, List, Dict
import logging
import os
import requests

# 导入自定义日志配置
from backend.logging_config import get_logger

# 获取日志记录器
logger = get_logger("backend.model")

# DeepSeek API配置
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量中读取DeepSeek API密钥
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

class ModelManager:
    def __init__(self, model_name: str = "deepseek-chat", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.inference_lock = asyncio.Lock()  # 推理锁，避免并发冲突
        # 创建线程池执行器，用于处理API调用
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=5,  # API调用可以并行处理
            thread_name_prefix="model_thread"
        )
    
    def load_model(self):
        """加载模型（DeepSeek API无需本地加载）"""
        logger.info("使用DeepSeek API，无需本地模型加载")
        return True
    
    def _warmup(self):
        """预热模型（DeepSeek API无需预热）"""
        logger.info("DeepSeek API无需预热")
    
    def _ensure_model_loaded(self):
        """确保模型已加载（DeepSeek API始终可用）"""
        logger.info("DeepSeek API已就绪")

    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """同步生成模型响应，使用DeepSeek API"""
        # 记录请求开始时间
        start_time = time.time()
        logger.info(f"开始处理生成请求，提示词长度: {len(prompt)} 字符")
        # 详细记录传入的提示词内容
        logger.info(f"传入模型的提示词: {prompt}")
        
        try:
            # 构建API请求参数
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": 0.95
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            }
            
            # 发送API请求
            api_start_time = time.time()
            response = requests.post(
                DEEPSEEK_API_URL,
                json=payload,
                headers=headers,
                timeout=60
            )
            
            # 记录API调用时间
            api_time = time.time() - api_start_time
            logger.info(f"API调用完成，耗时: {api_time:.2f} 秒")
            
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"API调用失败，状态码: {response.status_code}")
                logger.error(f"错误信息: {response.text}")
                return f"API调用失败: {response.status_code}"
            
            # 解析响应
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                generated_response = response_data["choices"][0]["message"]["content"]
                
                # 记录生成的响应
                logger.info(f"模型生成的部分: {generated_response}")
                
                # 记录总请求处理时间
                total_time = time.time() - start_time
                logger.info(f"生成请求处理完成，总耗时: {total_time:.2f} 秒，生成响应长度: {len(generated_response)} 字符")
                
                return generated_response
            else:
                logger.error("API响应格式错误")
                return "API响应格式错误"
            
        except Exception as e:
            # 记录请求处理失败
            total_time = time.time() - start_time
            logger.error(f"生成请求处理失败，总耗时: {total_time:.2f} 秒，错误: {e}")
            return f"生成失败: {str(e)}"
    
    async def async_generate_response(self, prompt: str, max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
        """异步生成模型响应，处理并发请求"""
        # 使用锁确保同时只有一个请求在处理
        async with self.inference_lock:
            # 使用自定义线程池执行器运行同步生成函数
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                self.executor,  # 使用自定义线程池执行器
                self.generate_response, 
                prompt, 
                max_new_tokens, 
                temperature
            )
            return response

# 全局模型管理器实例
model_manager = ModelManager()

# 初始化函数，供FastAPI启动时调用
def initialize_model():
    """初始化并加载模型"""
    return model_manager.load_model()
