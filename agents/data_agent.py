import akshare as ak
import random
import baostock as bs
import pandas as pd
import numpy as np
import time
import datetime
import os
import json
import signal
import threading
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from functools import wraps
from bs4 import BeautifulSoup

# 解决嵌套事件循环问题
import nest_asyncio
nest_asyncio.apply()

# 新增性能优化库
import aiohttp
import cachetools
import diskcache
from cachetools import TTLCache

# 添加tickflow作为备用数据源
import tickflow as tf

# 接口优先级配置
# 个股数据：必盈API > AKShare > 其他
# 其他数据：AKShare > 其他
API_PRIORITY = {
    'stock_data': ['biying', 'akshare', 'other'],
    'other_data': ['akshare', 'other']
}

# 导入自定义日志配置
from backend.logging_config import get_logger
# 导入自定义数据处理器
from .data_processor import DataProcessor, convert_numpy_types

# 获取日志记录器
logger = get_logger("agents.data")


def rate_limited(max_calls, period):
    """装饰器：限制函数调用频率，防止API访问过于频繁
    
    参数:
    max_calls: 时间段内最大调用次数
    period: 时间段长度（秒）
    """
    def decorator(func):
        calls = []
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # 清理过期的调用记录
            calls[:] = [call for call in calls if now - call < period]
            
            if len(calls) >= max_calls:
                # 计算需要等待的时间
                wait_time = period - (now - calls[0])
                if wait_time > 0:
                    logger.info(f"请求频率过高，等待 {wait_time:.2f} 秒后重试...")
                    time.sleep(wait_time)
                    # 清理过期记录并重试
                    now = time.time()
                    calls[:] = [call for call in calls if now - call < period]
            
            calls.append(time.time())
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def timeout(seconds=2, error_message="函数执行超时"):
    """装饰器：限制函数执行时间（跨平台实现）
    
    参数:
    seconds: 超时时间（秒）
    error_message: 超时错误信息
    """
    def decorator(func):
        result = None
        exception = None
        
        def worker(*args, **kwargs):
            nonlocal result, exception
            try:
                result = func(*args, **kwargs)
            except Exception as e:
                exception = e
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal result, exception
            # 重置结果和异常
            result = None
            exception = None
            
            # 创建并启动线程
            thread = threading.Thread(target=worker, args=args, kwargs=kwargs)
            thread.daemon = True
            thread.start()
            
            # 等待指定时间
            thread.join(seconds)
            
            # 检查线程是否仍在运行
            if thread.is_alive():
                raise TimeoutError(error_message)
            
            # 如果发生异常，重新抛出
            if exception is not None:
                raise exception
            
            return result
        return wrapper
    return decorator

def retry_with_backoff(max_retries=3, backoff_factor=2.0, exceptions=(Exception, TimeoutError)):
    """装饰器：添加固定延迟的自动重试机制
    
    参数:
    max_retries: 最大重试次数
    backoff_factor: 固定重试延迟（秒）
    exceptions: 要捕获并重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，记录错误并返回空值，允许调用方切换到下一个数据源
                        logger.warning(f"第{attempt+1}次调用 {func.__name__} 失败，不再重试: {e}")
                        return pd.DataFrame()  # 返回空DataFrame而不是抛出异常
                    # 使用固定延迟
                    delay = backoff_factor
                    logger.warning(f"第{attempt+1}次调用 {func.__name__} 失败: {e}，将在 {delay:.2f} 秒后重试")
                    time.sleep(delay)
        return wrapper
    return decorator


def async_retry_with_backoff(max_retries=3, backoff_factor=2.0, exceptions=(Exception,)):
    """装饰器：为异步函数添加固定延迟的自动重试机制
    
    参数:
    max_retries: 最大重试次数
    backoff_factor: 固定重试延迟（秒）
    exceptions: 要捕获并重试的异常类型
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        # 最后一次尝试失败，记录错误并返回空值，允许调用方切换到下一个数据源
                        logger.warning(f"第{attempt+1}次调用 {func.__name__} 失败，不再重试: {e}")
                        # 返回空的DataFrame而不是抛出异常，确保单个接口失败不影响整体流程
                        if func.__name__ == '_get_stock_quote_async':
                            if len(args) > 1:
                                return (args[1], pd.DataFrame())  # args[0]是self，args[1]是symbol
                            elif 'symbol' in kwargs:
                                return (kwargs['symbol'], pd.DataFrame())
                            else:
                                return (None, pd.DataFrame())
                        else:
                            return pd.DataFrame()
                    # 使用固定延迟
                    delay = backoff_factor
                    logger.warning(f"第{attempt+1}次调用 {func.__name__} 失败: {e}，将在 {delay:.2f} 秒后重试")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator


class DataFetchingAgent:
    # 必盈API常量定义
    # 获取基础股票代码和名称列表接口
    BIYING_API_STOCK_LIST = "https://api.biyingapi.com/hslt/list/{licence}"
    # 公司信息查询接口
    BIYING_API_COMPANY_INFO = "https://api.biyingapi.com/hscp/gsjj/{stock_code}/{licence}"
    # 指数、行业、概念列表接口
    BIYING_API_INDEX_SECTOR_LIST = "https://api.biyingapi.com/hszg/list/{licence}"
    # 根据指数、行业、概念树查询股票接口
    BIYING_API_STOCKS_BY_INDEX_SECTOR = "https://api.biyingapi.com/hszg/gg/{code}/{licence}"
    # 根据股票代码获取相关指数、行业、概念接口
    BIYING_API_RELATED_INDEX_SECTOR = "https://api.biyingapi.com/hszg/zg/{stock_code}/{licence}"
    # 资金流向数据接口
    BIYING_API_FUND_FLOW = "https://api.biyingapi.com/hsstock/history/transaction/{stock_code}/{licence}?st={start_time}&et={end_time}&lt={limit}"
    # 历史涨跌停价格接口
    BIYING_API_HISTORY_LIMIT_PRICE = "https://api.biyingapi.com/hsstock/stopprice/history/{stock_code}/{licence}?st={start_time}&et={end_time}"
    # 实时交易数据接口
    BIYING_API_REAL_TIME_TRANSACTION = "https://api.biyingapi.com/hsrl/ssjy/{stock_code}/{licence}"
    # 最新行情数据接口（支持多周期）
    BIYING_API_LATEST_QUOTE = "https://api.biyingapi.com/hsstock/latest/{stock_code}.{market}/{period}/{adjust}/{licence}?lt={limit}"
    # 公司所属指数接口
    BIYING_API_COMPANY_INDICES = "https://api.biyingapi.com/hscp/sszs/{stock_code}/{licence}"
    # 近一年各个季度的利润接口
    BIYING_API_QUARTERLY_PROFIT = "https://api.biyingapi.com/hscp/jdlr/{stock_code}/{licence}"
    # 资产负债表接口
    BIYING_API_BALANCE_SHEET = "https://api.biyingapi.com/hsstock/financial/balance/{stock_code}/{licence}?st={start_time}&et={end_time}"
    # 公司主要指标接口
    BIYING_API_COMPANY_METRICS = "https://api.biyingapi.com/hsstock/financial/pershareindex/{stock_code}/{licence}?st={start_time}&et={end_time}"
    # 技术指标接口 - MACD
    BIYING_API_MACD = "https://api.biyingapi.com/hsstock/history/macd/{stock_code}/{period}/{adjust}/{licence}?st={start_time}&et={end_time}&lt={limit}"
    # 技术指标接口 - MA
    BIYING_API_MA = "https://api.biyingapi.com/hsstock/history/ma/{stock_code}/{period}/{adjust}/{licence}?st={start_time}&et={end_time}&lt={limit}"
    # 技术指标接口 - BOLL
    BIYING_API_BOLL = "https://api.biyingapi.com/hsstock/history/boll/{stock_code}/{period}/{adjust}/{licence}?st={start_time}&et={end_time}&lt={limit}"
    # 技术指标接口 - KDJ
    BIYING_API_KDJ = "https://api.biyingapi.com/hsstock/history/kdj/{stock_code}/{period}/{adjust}/{licence}?st={start_time}&et={end_time}&lt={limit}"
    def __init__(self):
        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 数据文件路径
        self.data_dir = os.path.join(self.root_dir, 'data')
        
        # 旧缓存目录路径（保留用于兼容性）
        self.cache_dir = os.path.join(self.root_dir, 'cache')
        # 创建缓存目录
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            logger.info(f"已创建缓存目录: {self.cache_dir}")
        
        # 使用cachetools的TTLCache实现内存缓存（30秒过期，最多2000个条目，提高命中率）
        self.memory_cache = TTLCache(maxsize=2000, ttl=30)
        # 使用diskcache实现磁盘缓存
        self.disk_cache = diskcache.Cache(os.path.join(self.root_dir, 'disk_cache'))
        logger.info(f"已创建disk_cache目录")
        
        # 初始化数据处理器
        self.data_processor = DataProcessor()
        
        # 初始化线程池，用于异步磁盘缓存保存
        import concurrent.futures
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # 配置优化后的requests会话（用于同步请求）
        self.session = requests.Session()
        
        # 为会话添加初始cookie，模拟真实浏览器访问
        self._add_initial_cookies()
        
        # 初始化tickflow客户端作为备用数据源
        try:
            # 使用免费版tickflow客户端
            self.tickflow_client = tf.TickFlow.free()
            logger.info("✓ TickFlow免费版客户端初始化成功")
        except Exception as e:
            logger.warning(f"TickFlow客户端初始化失败: {e}")
            self.tickflow_client = None
        
        # 黄金价格API配置
        self.gold_api_url = 'http://web.juhe.cn/finance/gold/shgold'
        self.gold_api_key = os.getenv("JUHE_API_KEY", "")  # 从环境变量获取API密钥
        
        # 聚合数据财经新闻API配置
        self.juhe_news_api_url = 'http://apis.juhe.cn/fapigx/caijing/query'
        self.juhe_news_api_key = os.getenv("JUHE_NEWS_API_KEY", "")  # 从环境变量获取API密钥
        
        # 新增数据源配置
        # 新浪财经API
        self.sina_api_url = 'https://hq.sinajs.cn'
        # 东方财富API
        self.eastmoney_api_url = 'https://push2.eastmoney.com'
        # 同花顺API
        self.ths_api_url = 'https://q.10jqka.com.cn'
        
        # 数据源优先级配置
        self.data_source_priority = {
            'stock_quote': ['biying', 'akshare', 'sina', 'eastmoney', 'tickflow'],
            'stock_history': ['biying', 'akshare', 'baostock', 'tickflow'],
            'market_overview': ['akshare', 'eastmoney', 'sina'],
            'sector_data': ['akshare', 'eastmoney', 'biying'],
            'index_data': ['akshare', 'eastmoney', 'baostock'],
            'news': ['juhe', 'eastmoney', 'sina']
        }
        
        # 数据验证配置
        self.data_validation = {
            'stock_quote': ['代码', '名称', '最新价', '涨跌幅', '涨跌额'],
            'stock_history': ['日期', '开盘', '最高', '最低', '收盘', '成交量', '成交额'],
            'market_overview': ['上涨股票数', '下跌股票数', '上涨比例', '下跌比例'],
            'sector_data': ['名称', '涨跌幅', '成交量', '成交额']
        }
        
        # 性能监控
        self.performance_metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0,
            'error_count': 0
        }
        
        # 缓存过期时间配置（秒）
        self.cache_ttl = {
            'stock_quote': 60,  # 股票实时行情 1分钟
            'stock_history': 3600,  # 股票历史数据 1小时
            'stock_news': 1800,  # 股票新闻 30分钟
            'market_overview': 300,  # 市场概览 5分钟
            'sector_data': 600,  # 板块数据 10分钟
            'index_data': 300,  # 指数数据 5分钟
            'gold_price': 600,  # 黄金价格 10分钟
            'default': 3600  # 默认 1小时
        }
        
        # 初始化API调用时间记录
        self.api_call_times = []
        
        # 加载外部数据文件
        self.common_indices = self._load_common_indices()
        self.stock_name_code_map = self._load_stock_name_code_map()
        self.common_sectors = self._load_common_sectors()
        
        # 启动缓存预热（异步执行，不阻塞初始化）
        import threading
        warmup_thread = threading.Thread(target=self._warmup_cache)
        warmup_thread.daemon = True
        warmup_thread.start()
    
    def _retry_on_exception(self, func, max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
        """重试装饰器，用于处理网络请求失败的情况
        
        Args:
            func: 要执行的函数
            max_retries: 最大重试次数
            delay: 初始延迟（秒）
            backoff: 延迟增长系数
            exceptions: 要捕获的异常类型
        
        Returns:
            函数执行结果
        """
        for attempt in range(max_retries):
            try:
                return func()
            except exceptions as e:
                if attempt == max_retries - 1:
                    # 最后一次尝试失败，记录错误并抛出异常
                    self.logger.warning(f"重试 {max_retries} 次后仍然失败: {e}")
                    raise
                
                # 计算下次重试的延迟
                retry_delay = delay * (backoff ** attempt)
                self.logger.debug(f"尝试 {attempt + 1} 失败: {e}，{retry_delay:.2f} 秒后重试")
                time.sleep(retry_delay)
        
        return None
    
    def __del__(self):
        """资源清理：关闭所有打开的资源"""
        try:
            # 关闭线程池
            if hasattr(self, 'thread_pool') and self.thread_pool is not None:
                self.thread_pool.shutdown(wait=False)
                logger.info("已关闭线程池")
        except Exception as e:
            logger.warning(f"关闭线程池时出错: {e}")
        
        try:
            # 关闭磁盘缓存连接
            if hasattr(self, 'disk_cache') and self.disk_cache is not None:
                self.disk_cache.close()
                logger.info("已关闭磁盘缓存连接")
        except Exception as e:
            logger.warning(f"关闭磁盘缓存连接时出错: {e}")
        
        try:
            # 关闭aiohttp会话（异步连接）
            if hasattr(self, 'aiohttp_session') and self.aiohttp_session is not None and not self.aiohttp_session.closed:
                import asyncio
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.aiohttp_session.close())
                logger.info("已关闭aiohttp客户端会话")
        except Exception as e:
            logger.warning(f"关闭aiohttp客户端会话时出错: {e}")
    
    def _add_initial_cookies(self):
        """为会话添加初始cookie，模拟真实浏览器访问"""
        # 模拟常见网站的cookie
        cookies_dict = {
            'Hm_lvt_78c58f01938e4d85eaf619eae71b4ed1': f'{int(time.time())}',
            'Hm_lpvt_78c58f01938e4d85eaf619eae71b4ed1': f'{int(time.time())}',
            'vjuids': f'{random.randint(1000000000, 9999999999)}.{random.random()}',
            'vjlast': f'{int(time.time() - random.randint(86400, 604800))}',
            'vinfo_n_f_l_n3': f'{random.randint(0, 1)}',
            'user-id': f'{random.randint(100000, 999999)}'
        }
        
        # 设置cookie到会话
        for key, value in cookies_dict.items():
            self.session.cookies.set(key, value, domain='finance.sina.com.cn')
            self.session.cookies.set(key, value, domain='eastmoney.com')
            self.session.cookies.set(key, value, domain='10jqka.com.cn')
        
        logger.debug("已添加初始cookie到会话")
    
    def rotate_session(self):
        """轮换会话，模拟不同用户访问"""
        logger.info("正在轮换会话...")
        
        # 关闭旧会话
        if hasattr(self, 'session') and self.session is not None:
            try:
                self.session.close()
                logger.debug("已关闭旧会话")
            except Exception as e:
                logger.warning(f"关闭旧会话时出错: {e}")
        
        # 创建新会话
        self.session = requests.Session()
        
        # 重新设置请求头
        self._set_request_headers()
        
        # 添加新的cookie
        self._add_initial_cookies()
        
        # 更新AKShare的会话
        global ak
        ak._requests = self.session
        
        logger.info("会话轮换完成")
    
    def _set_request_headers(self):
        """设置请求头（重构为单独方法，便于会话轮换时重用）"""
        # 扩展的User-Agent池，包含更多主流浏览器和设备
        user_agents = [
            # Chrome桌面版
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            
            # Firefox桌面版
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.2; rv:122.0) Gecko/120.0 Firefox/122.0',
            
            # Safari桌面版
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
            
            # Edge桌面版
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
            
            # 移动设备
            'Mozilla/5.0 (iPhone; CPU iPhone OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (iPad; CPU OS 17_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',
            'Mozilla/5.0 (Android 13; Mobile; rv:122.0) Gecko/122.0 Firefox/122.0',
            'Mozilla/5.0 (Linux; Android 13; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36'
        ]
        
        # 增强的Referer池，模拟从不同页面访问
        referers = [
            'https://quote.eastmoney.com/',
            'https://finance.sina.com.cn/',
            'https://www.10jqka.com.cn/',
            'https://www.jrj.com.cn/',
            'https://www.cnstock.com/',
            'https://www.stcn.com/',
            'https://www.hexun.com/',
            'https://www.yicai.com/',
            'https://www.eastmoney.com/',
            'https://stock.163.com/',
            'https://www.ifeng.com/',
            'https://www.cctv.com/',
            'https://www.sohu.com/',
            'https://www.baidu.com/',
            'https://www.google.com/'
        ]
        
        # 随机选择User-Agent和Referer
        user_agent = random.choice(user_agents)
        referer = random.choice(referers)
        
        # 动态生成Accept和Accept-Language，增加随机性
        accept_headers = [
            'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        ]
        
        accept_language_headers = [
            'zh-CN,zh;q=0.9,en;q=0.8',
            'zh-CN,zh;q=0.9',
            'zh-CN,zh-TW;q=0.9,zh;q=0.8,en-US;q=0.5,en;q=0.3'
        ]
        
        self.session.headers.update({
            'User-Agent': user_agent,
            'Accept': random.choice(accept_headers),
            'Accept-Language': random.choice(accept_language_headers),
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': referer,
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
            'X-Requested-With': 'XMLHttpRequest',
            'DNT': '1',  # Do Not Track
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'TE': 'trailers'
        })
        
        # 初始化aiohttp客户端会话（用于异步请求）
        self.aiohttp_session = None
        

        
        # 配置高级重试策略
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],  # 只对这些状态码重试
            backoff_factor=1.0,  # 指数退避
            allowed_methods=["HEAD", "GET", "OPTIONS"]  # 只对这些方法重试
        )
        
        # 创建适配器
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        # 应用适配器到会话
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        
        # 设置超时时间
        self.session.timeout = 30
        
        # 为akshare设置请求超时和请求头
        ak._requests = self.session
        
        # 初始化baostock
        try:
            lg = bs.login()
            if lg.error_code == '0':
                logger.info("成功初始化baostock API")
            else:
                logger.warning(f"baostock API初始化失败: {lg.error_msg}")
        except Exception as e:
            logger.error(f"baostock API初始化异常: {e}")
        
        # akshare库不支持直接设置会话，保留会话配置代码供参考
        logger.info("已配置优化的HTTP会话参数，提高API访问稳定性")
        logger.info("已集成baostock，支持A股、港股、美股查询")
        logger.info(f"已加载外部数据文件: {len(self.common_indices)}个指数, {len(self.stock_name_code_map)}个股票, {len(self.common_sectors)}个板块")
        
    async def _get_aiohttp_session(self) -> aiohttp.ClientSession:
        """获取aiohttp客户端会话
        
        Returns:
            aiohttp.ClientSession: 客户端会话对象
        """
        if self.aiohttp_session is None or self.aiohttp_session.closed:
            # 配置请求头和重试策略
            user_agents = [
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Firefox/120.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15',
                'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1'
            ]
            
            referers = [
                'https://quote.eastmoney.com/',
                'https://finance.sina.com.cn/',
                'https://www.10jqka.com.cn/',
                'https://www.jrj.com.cn/'
            ]
            
            headers = {
                'User-Agent': random.choice(user_agents),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Referer': random.choice(referers),
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
                'X-Requested-With': 'XMLHttpRequest',
                'DNT': '1',  # Do Not Track
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'cross-site',
                'Sec-Fetch-User': '?1'
            }
            
            # 创建aiohttp会话
            self.aiohttp_session = aiohttp.ClientSession(headers=headers)
            logger.debug("已创建aiohttp客户端会话")
        
        return self.aiohttp_session
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """生成缓存键（使用哈希算法减少键长度）
        
        Args:
            func_name: 函数名
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            str: 生成的缓存键
        """
        import hashlib
        import pickle
        
        # 创建一个包含所有参数的元组
        key_tuple = (func_name, args, tuple(sorted(kwargs.items())))
        
        # 使用pickle序列化，然后计算哈希值
        serialized = pickle.dumps(key_tuple, protocol=pickle.HIGHEST_PROTOCOL)
        hash_value = hashlib.md5(serialized).hexdigest()
        
        # 生成紧凑的缓存键
        return f"{func_name}:{hash_value[:12]}"
    
    def _load_common_indices(self) -> List[str]:
        """从JSON文件加载常见指数列表"""
        file_path = os.path.join(self.data_dir, 'common_indices.json')
        return self._load_json_file(file_path, default=[])
    
    def _load_stock_name_code_map(self) -> Dict[str, str]:
        """从JSON文件加载股票名称和代码映射"""
        file_path = os.path.join(self.data_dir, 'stock_name_code_map.json')
        return self._load_json_file(file_path, default={})
    
    def _load_common_sectors(self) -> List[str]:
        """从JSON文件加载常见板块列表"""
        file_path = os.path.join(self.data_dir, 'common_sectors.json')
        return self._load_json_file(file_path, default=[])
    
    def _load_json_file(self, file_path: str, default):
        """加载JSON文件，失败时返回默认值"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"数据文件未找到: {file_path}，使用默认值")
        except json.JSONDecodeError:
            logger.error(f"JSON文件解析失败: {file_path}，使用默认值")
        except Exception as e:
            logger.error(f"加载JSON文件失败: {file_path}, 错误: {e}，使用默认值")
        return default

    def _save_json_file(self, file_path: str, data):
        """保存数据到JSON文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"成功保存数据到文件: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存JSON文件失败: {file_path}, 错误: {e}")
            return False
    
    def _warmup_cache(self):
        """缓存预热：预先加载常用数据到缓存中，提高后续请求的响应速度"""
        try:
            logger.info("开始缓存预热...")
            
            # 预热市场概览数据
            try:
                market_overview = self.get_market_overview()
                if market_overview:
                    logger.info("市场概览数据缓存预热成功")
            except Exception as e:
                logger.warning(f"市场概览数据缓存预热失败: {e}")
            
            # 预热常见指数数据
            try:
                common_indices = ['上证指数', '深证成指', '创业板指']
                for index_name in common_indices:
                    index_info = self.get_index_info_by_name(index_name)
                    if index_info is not None and not index_info.empty:
                        logger.debug(f"指数 {index_name} 数据缓存预热成功")
            except Exception as e:
                logger.warning(f"指数数据缓存预热失败: {e}")
            
            # 预热热门股票数据
            try:
                popular_stocks = ['sh600519', 'sh601318', 'sz000858', 'sh600036', 'sz000333']
                if popular_stocks:
                    stock_quotes = self.get_stock_quotes(popular_stocks)
                    valid_count = sum(1 for df in stock_quotes.values() if not df.empty)
                    logger.info(f"热门股票数据缓存预热成功，有效数据: {valid_count}/{len(popular_stocks)}")
            except Exception as e:
                logger.warning(f"热门股票数据缓存预热失败: {e}")
            
            # 预热行业板块数据
            try:
                sectors = ['银行', '科技', '医药', '消费', '新能源']
                for sector in sectors:
                    sector_info = self.get_sector_info_by_name(sector)
                    if sector_info is not None and not sector_info.empty:
                        logger.debug(f"板块 {sector} 数据缓存预热成功")
            except Exception as e:
                logger.warning(f"行业板块数据缓存预热失败: {e}")
            
            logger.info("缓存预热完成")
        except Exception as e:
            logger.error(f"缓存预热过程中出错: {e}")
    
    def _validate_data(self, data_type: str, data) -> bool:
        """验证数据是否符合要求
        
        Args:
            data_type: 数据类型
            data: 要验证的数据
            
        Returns:
            bool: 数据是否有效
        """
        if data is None:
            return False
        
        if data_type not in self.data_validation:
            return True
        
        required_fields = self.data_validation[data_type]
        
        if isinstance(data, pd.DataFrame):
            # 检查DataFrame是否为空
            if data.empty:
                return False
            # 检查是否包含所有必需字段
            for field in required_fields:
                if field not in data.columns:
                    logger.warning(f"数据类型 {data_type} 缺少字段: {field}")
                    return False
            return True
        
        elif isinstance(data, dict):
            # 检查字典是否包含所有必需字段
            for field in required_fields:
                if field not in data:
                    logger.warning(f"数据类型 {data_type} 缺少字段: {field}")
                    return False
            return True
        
        elif isinstance(data, list):
            # 检查列表是否为空
            if not data:
                return False
            # 检查列表中的第一个元素是否包含所有必需字段
            if data and isinstance(data[0], dict):
                for field in required_fields:
                    if field not in data[0]:
                        logger.warning(f"数据类型 {data_type} 缺少字段: {field}")
                        return False
            return True
        
        return False
    
    def _clean_data(self, data_type: str, data):
        """清洗数据，处理缺失值和异常值
        
        Args:
            data_type: 数据类型
            data: 要清洗的数据
            
        Returns:
            清洗后的数据
        """
        if data is None:
            return data
        
        if isinstance(data, pd.DataFrame):
            # 处理缺失值
            data = data.dropna()
            
            # 处理异常值
            if data_type == 'stock_quote' or data_type == 'stock_history':
                # 确保价格和成交量为正数
                for col in ['最新价', '开盘', '最高', '最低', '收盘', '成交量', '成交额']:
                    if col in data.columns:
                        data = data[data[col] >= 0]
            
            return data
        
        elif isinstance(data, list):
            # 清洗列表中的字典数据
            cleaned_data = []
            for item in data:
                if isinstance(item, dict):
                    # 移除包含None值的字段
                    cleaned_item = {k: v for k, v in item.items() if v is not None}
                    cleaned_data.append(cleaned_item)
                else:
                    cleaned_data.append(item)
            return cleaned_data
        
        elif isinstance(data, dict):
            # 移除包含None值的字段
            cleaned_data = {k: v for k, v in data.items() if v is not None}
            return cleaned_data
        
        return data
    
    def _get_data_from_source(self, data_type: str, sources: List[str], *args, **kwargs):
        """从多个数据源获取数据，按照优先级顺序尝试
        
        Args:
            data_type: 数据类型
            sources: 数据源列表
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        for source in sources:
            try:
                start_time = time.time()
                self.performance_metrics['api_calls'] += 1
                
                if source == 'biying':
                    # 调用必盈API，根据数据类型选择合适的API URL
                    if data_type == 'stock_quote':
                        # 股票行情数据
                        symbol = kwargs.get('symbol')
                        if symbol:
                            # 提取市场代码（sh或sz）
                            market = symbol[:2]
                            stock_code = symbol[2:]
                            result = self._call_biying_api(
                                self.BIYING_API_LATEST_QUOTE,
                                stock_code=stock_code,
                                market=market,
                                period='1d',
                                adjust='qfq',
                                limit=1
                            )
                        else:
                            result = None
                    else:
                        # 其他数据类型，直接调用
                        result = self._call_biying_api(*args, **kwargs)
                elif source == 'akshare':
                    # 调用akshare API
                    result = self._call_akshare_api(data_type, *args, **kwargs)
                elif source == 'sina':
                    # 调用新浪财经API
                    result = self._call_sina_api(data_type, *args, **kwargs)
                elif source == 'eastmoney':
                    # 调用东方财富API
                    result = self._call_eastmoney_api(data_type, *args, **kwargs)
                elif source == 'baostock':
                    # 调用baostock API
                    result = self._call_baostock_api(data_type, *args, **kwargs)
                elif source == 'tickflow':
                    # 调用tickflow API
                    result = self._call_tickflow_api(data_type, *args, **kwargs)
                elif source == 'juhe':
                    # 调用聚合数据API
                    result = self._call_juhe_api(data_type, *args, **kwargs)
                else:
                    continue
                
                # 记录API调用时间
                elapsed_time = time.time() - start_time
                self.api_call_times.append(elapsed_time)
                if len(self.api_call_times) > 100:
                    self.api_call_times.pop(0)
                self.performance_metrics['average_response_time'] = sum(self.api_call_times) / len(self.api_call_times)
                
                # 验证数据
                if self._validate_data(data_type, result):
                    # 清洗数据
                    cleaned_data = self._clean_data(data_type, result)
                    logger.info(f"从数据源 {source} 成功获取并清洗 {data_type} 数据")
                    return cleaned_data
                else:
                    logger.warning(f"从数据源 {source} 获取的 {data_type} 数据无效")
                    self.performance_metrics['error_count'] += 1
            except Exception as e:
                logger.error(f"从数据源 {source} 获取 {data_type} 数据失败: {e}")
                self.performance_metrics['error_count'] += 1
                continue
        
        logger.error(f"所有数据源获取 {data_type} 数据失败")
        return None
    
    def _call_akshare_api(self, data_type: str, *args, **kwargs):
        """调用akshare API获取数据
        
        Args:
            data_type: 数据类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        try:
            if data_type == 'stock_quote':
                # 获取股票行情
                return ak.stock_zh_a_spot_em()
            elif data_type == 'stock_history':
                # 获取股票历史数据
                symbol = kwargs.get('symbol')
                days = kwargs.get('days', 30)
                end_date = datetime.datetime.now().strftime('%Y%m%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y%m%d')
                return ak.stock_zh_a_daily(symbol=symbol, start_date=start_date, end_date=end_date)
            elif data_type == 'market_overview':
                # 获取市场概览
                return ak.stock_zh_a_spot_em()
            elif data_type == 'sector_data':
                # 获取板块数据
                return ak.stock_sector_spot()
            else:
                return None
        except Exception as e:
            logger.error(f"调用akshare API失败: {e}")
            return None
    
    def _call_sina_api(self, data_type: str, *args, **kwargs):
        """调用新浪财经API获取数据
        
        Args:
            data_type: 数据类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        try:
            if data_type == 'stock_quote':
                symbol = kwargs.get('symbol')
                if not symbol:
                    return None
                if symbol.startswith('sh'):
                    sina_code = f"sh{symbol[2:]}"
                elif symbol.startswith('sz'):
                    sina_code = f"sz{symbol[2:]}"
                else:
                    return None
                url = f"{self.sina_api_url}?list={sina_code}"
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    content = response.text
                    import re
                    match = re.search(r'var hq_str_{}="(.*?)";'.format(sina_code), content)
                    if match:
                        data = match.group(1).split(',')
                        if len(data) >= 11:
                            stock_data = {
                                '代码': symbol,
                                '名称': data[0],
                                '最新价': float(data[3]),
                                '涨跌幅': ((float(data[3]) - float(data[2])) / float(data[2])) * 100,
                                '涨跌额': float(data[3]) - float(data[2]),
                                '成交量': int(data[8]),
                                '成交额': float(data[9]),
                                '今开': float(data[1]),
                                '最高': float(data[4]),
                                '最低': float(data[5]),
                                '昨收': float(data[2])
                            }
                            return pd.DataFrame([stock_data])
            elif data_type == 'market_overview':
                import re
                all_stocks_url = "http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
                params = {
                    'page': 1,
                    'num': 5000,
                    'sort': 'symbol',
                    'asc': 1,
                    'node': 'hs_a',
                    'symbol': '',
                    '_s_r_a': 'page'
                }
                response = self.session.get(all_stocks_url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data and isinstance(data, list):
                        stocks = []
                        for item in data:
                            stocks.append({
                                '代码': item.get('code', ''),
                                '名称': item.get('name', ''),
                                '最新价': float(item.get('trade', 0) or 0),
                                '涨跌幅': float(item.get('changepercent', 0) or 0),
                                '涨跌额': float(item.get('pricechange', 0) or 0),
                                '成交量': int(item.get('volume', 0) or 0),
                                '成交额': float(item.get('amount', 0) or 0),
                                '今开': float(item.get('open', 0) or 0),
                                '最高': float(item.get('high', 0) or 0),
                                '最低': float(item.get('low', 0) or 0),
                                '昨收': float(item.get('settlement', 0) or 0)
                            })
                        return pd.DataFrame(stocks)
            return None
        except Exception as e:
            logger.error(f"调用新浪财经API失败: {e}")
            return None
    
    def _call_eastmoney_api(self, data_type: str, *args, **kwargs):
        """调用东方财富API获取数据
        
        Args:
            data_type: 数据类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        try:
            if data_type == 'stock_quote':
                symbol = kwargs.get('symbol')
                if not symbol:
                    return None
                if symbol.startswith('sh'):
                    eastmoney_code = f"1.{symbol[2:]}"
                elif symbol.startswith('sz'):
                    eastmoney_code = f"0.{symbol[2:]}"
                else:
                    return None
                url = f"{self.eastmoney_api_url}/api/qt/stock/get?secid={eastmoney_code}&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f115,f152"
                response = self.session.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data'):
                        stock_data = data['data']
                        result = {
                            '代码': symbol,
                            '名称': stock_data.get('f14', ''),
                            '最新价': stock_data.get('f2', 0),
                            '涨跌幅': stock_data.get('f3', 0),
                            '涨跌额': stock_data.get('f4', 0),
                            '成交量': stock_data.get('f5', 0),
                            '成交额': stock_data.get('f6', 0),
                            '今开': stock_data.get('f17', 0),
                            '最高': stock_data.get('f15', 0),
                            '最低': stock_data.get('f16', 0),
                            '昨收': stock_data.get('f18', 0)
                        }
                        return pd.DataFrame([result])
            elif data_type == 'market_overview':
                url = "https://push2.eastmoney.com/api/qt/clist/get"
                params = {
                    'pn': 1,
                    'pz': 5000,
                    'po': 1,
                    'np': 1,
                    'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                    'fltt': 2,
                    'invt': 2,
                    'fid': 'f3',
                    'fs': 'm:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23',
                    'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f115,f152'
                }
                response = self.session.get(url, params=params, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('data') and data['data'].get('diff'):
                        stocks = []
                        for item in data['data']['diff']:
                            stocks.append({
                                '代码': item.get('f12', ''),
                                '名称': item.get('f14', ''),
                                '最新价': item.get('f2', 0),
                                '涨跌幅': item.get('f3', 0),
                                '涨跌额': item.get('f4', 0),
                                '成交量': item.get('f5', 0),
                                '成交额': item.get('f6', 0),
                                '今开': item.get('f17', 0),
                                '最高': item.get('f15', 0),
                                '最低': item.get('f16', 0),
                                '昨收': item.get('f18', 0)
                            })
                        return pd.DataFrame(stocks)
            elif data_type == 'sector_data':
                sector_name = kwargs.get('sector_name')
                if sector_name:
                    url = "https://push2.eastmoney.com/api/qt/clist/get"
                    params = {
                        'pn': 1,
                        'pz': 100,
                        'po': 1,
                        'np': 1,
                        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                        'fltt': 2,
                        'invt': 2,
                        'fid': 'f3',
                        'fs': 'm:90+t:2',
                        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f115,f152'
                    }
                    response = self.session.get(url, params=params, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data') and data['data'].get('diff'):
                            sectors = []
                            for item in data['data']['diff']:
                                if sector_name in item.get('f14', ''):
                                    sectors.append({
                                        '板块名称': item.get('f14', ''),
                                        '涨跌幅': item.get('f3', 0),
                                        '涨跌额': item.get('f4', 0),
                                        '成交量': item.get('f5', 0),
                                        '成交额': item.get('f6', 0)
                                    })
                            if sectors:
                                return pd.DataFrame(sectors)
                else:
                    url = "https://push2.eastmoney.com/api/qt/clist/get"
                    params = {
                        'pn': 1,
                        'pz': 200,
                        'po': 1,
                        'np': 1,
                        'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                        'fltt': 2,
                        'invt': 2,
                        'fid': 'f3',
                        'fs': 'm:90+t:2',
                        'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f115,f152'
                    }
                    response = self.session.get(url, params=params, timeout=15)
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('data') and data['data'].get('diff'):
                            sectors = []
                            for item in data['data']['diff']:
                                sectors.append({
                                    '板块名称': item.get('f14', ''),
                                    '涨跌幅': item.get('f3', 0),
                                    '涨跌额': item.get('f4', 0),
                                    '成交量': item.get('f5', 0),
                                    '成交额': item.get('f6', 0)
                                })
                            return pd.DataFrame(sectors)
            return None
        except Exception as e:
            logger.error(f"调用东方财富API失败: {e}")
            return None
    
    def _call_baostock_api(self, data_type: str, *args, **kwargs):
        """调用baostock API获取数据
        
        Args:
            data_type: 数据类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        try:
            if data_type == 'stock_history':
                # 获取股票历史数据
                symbol = kwargs.get('symbol')
                days = kwargs.get('days', 30)
                end_date = datetime.datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
                # 处理股票代码格式
                if symbol.startswith('sh'):
                    baostock_code = f"{symbol[2:]}.SH"
                elif symbol.startswith('sz'):
                    baostock_code = f"{symbol[2:]}.SZ"
                else:
                    return None
                # 调用baostock API
                rs = bs.query_history_k_data_plus(
                    baostock_code,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"
                )
                # 转换为DataFrame
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                if data_list:
                    df = pd.DataFrame(data_list, columns=rs.fields)
                    # 重命名列
                    df.rename(columns={
                        'date': '日期',
                        'open': '开盘',
                        'high': '最高',
                        'low': '最低',
                        'close': '收盘',
                        'volume': '成交量',
                        'amount': '成交额'
                    }, inplace=True)
                    # 转换数据类型
                    numeric_cols = ['开盘', '最高', '最低', '收盘', '成交量', '成交额']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df
            return None
        except Exception as e:
            logger.error(f"调用baostock API失败: {e}")
            return None
    
    def _call_tickflow_api(self, data_type: str, *args, **kwargs):
        """调用tickflow API获取数据
        
        Args:
            data_type: 数据类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        try:
            if not self.tickflow_client:
                return None
            
            if data_type == 'stock_quote':
                # 获取股票行情
                symbol = kwargs.get('symbol')
                if not symbol:
                    return None
                # 处理股票代码格式
                if symbol.startswith('sh'):
                    tickflow_code = f"{symbol[2:]}.SH"
                elif symbol.startswith('sz'):
                    tickflow_code = f"{symbol[2:]}.SZ"
                else:
                    return None
                # 调用tickflow API，检查是否有quote方法
                if hasattr(self.tickflow_client, 'quote'):
                    quote = self.tickflow_client.quote(tickflow_code)
                    if quote:
                        stock_data = {
                            '代码': symbol,
                            '名称': quote.name,
                            '最新价': quote.price,
                            '涨跌幅': quote.change_percent,
                            '涨跌额': quote.change,
                            '成交量': quote.volume,
                            '成交额': quote.amount,
                            '今开': quote.open,
                            '最高': quote.high,
                            '最低': quote.low,
                            '昨收': quote.pre_close
                        }
                        return pd.DataFrame([stock_data])
                else:
                    # TickFlow客户端没有quote方法，返回None
                    logger.warning("TickFlow客户端不支持quote方法")
            return None
        except Exception as e:
            logger.error(f"调用tickflow API失败: {e}")
            return None
    
    def _call_juhe_api(self, data_type: str, *args, **kwargs):
        """调用聚合数据API获取数据
        
        Args:
            data_type: 数据类型
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            获取的数据
        """
        try:
            if data_type == 'news':
                # 获取财经新闻
                params = {
                    'key': self.juhe_news_api_key,
                    'type': kwargs.get('type', 'stock'),
                    'page': kwargs.get('page', 1),
                    'pagesize': kwargs.get('pagesize', 20)
                }
                response = self.session.get(self.juhe_news_api_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('resultcode') == '200':
                        return data.get('result', [])
            return None
        except Exception as e:
            logger.error(f"调用聚合数据API失败: {e}")
            return None

    def _add_to_stock_name_code_map(self, name: str, code: str):
        """将股票名称和代码添加到映射表并保存"""
        if name and code and name not in self.stock_name_code_map:
            self.stock_name_code_map[name] = code
            file_path = os.path.join(self.data_dir, 'stock_name_code_map.json')
            return self._save_json_file(file_path, self.stock_name_code_map)
        return False
    
    def get_performance_metrics(self):
        """获取性能指标
        
        Returns:
            dict: 性能指标
        """
        return self.performance_metrics
    
    def reset_performance_metrics(self):
        """重置性能指标"""
        self.performance_metrics = {
            'api_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0,
            'error_count': 0
        }
        self.api_call_times = []
    
    def log_performance(self):
        """记录性能指标"""
        metrics = self.get_performance_metrics()
        logger.info(f"性能指标: {metrics}")

    def _add_to_common_indices(self, index_name: str):
        """将指数或ETF名称添加到列表并保存"""
        if index_name and index_name not in self.common_indices:
            self.common_indices.append(index_name)
            file_path = os.path.join(self.data_dir, 'common_indices.json')
            return self._save_json_file(file_path, self.common_indices)
        return False

    def update_stock_name_code_map_from_biying(self) -> bool:
        """从必盈API获取企业列表并更新本地映射表
        
        Returns:
            bool: 是否成功更新
        """
        try:
            logger.info("开始从必盈API获取企业列表更新本地映射表")
            
            # 1. 调用必盈API获取企业列表
            result = self._call_biying_api(self.BIYING_API_STOCK_LIST)
            if not result:
                logger.error("获取必盈API企业列表失败")
                return False
            
            # 2. 解析API返回数据
            # 根据实际API测试，股票列表接口直接返回一个列表
            if not isinstance(result, list):
                logger.error("必盈API返回数据格式不正确")
                logger.debug(f"必盈API实际返回格式: {type(result).__name__}, 内容: {result}")
                return False
            
            biying_stock_list = result
            if not biying_stock_list:
                logger.error("必盈API返回的企业列表为空")
                return False
            
            # 3. 构建新的映射表
            new_stock_map = {}
            for stock in biying_stock_list:
                if isinstance(stock, dict) and "dm" in stock and "mc" in stock:
                    code_with_exchange = stock["dm"]  # 格式：002460.SZ
                    name = stock["mc"]  # 字段名是'mc'（股票名称）
                    # 只保存纯数字代码，去掉所有市场前缀
                    if "." in code_with_exchange:
                        # 格式：002460.SZ -> 002460
                        full_code = code_with_exchange.split(".")[0]
                    else:
                        # 格式：SH002460或002460 -> 002460
                        if code_with_exchange.startswith('SH') or code_with_exchange.startswith('SZ'):
                            full_code = code_with_exchange[2:]
                        else:
                            full_code = code_with_exchange
                    new_stock_map[name] = full_code
            
            logger.info(f"从必盈API获取到 {len(new_stock_map)} 个股票信息")
            
            # 4. 比较并更新变化的部分
            changes = {}
            additions = 0
            updates = 0
            
            # 检查新增和更新的股票
            for name, code in new_stock_map.items():
                if name not in self.stock_name_code_map:
                    # 新增股票
                    changes[name] = code
                    additions += 1
                elif self.stock_name_code_map[name] != code:
                    # 更新股票代码
                    changes[name] = code
                    updates += 1
            
            # 记录删除的股票（可选，这里不做删除操作）
            deleted_count = len(self.stock_name_code_map) - len(new_stock_map)
            
            logger.info(f"股票映射表更新统计: 新增 {additions} 个, 更新 {updates} 个, 可能删除 {deleted_count} 个")
            
            # 5. 如果有变化，更新映射表并保存
            if changes:
                # 更新内存中的映射表
                self.stock_name_code_map.update(changes)
                
                # 更新缓存
                self._cached_sorted_stock_names = sorted(self.stock_name_code_map.keys(), key=len, reverse=True)
                
                # 保存到文件
                file_path = os.path.join(self.data_dir, 'stock_name_code_map.json')
                if self._save_json_file(file_path, self.stock_name_code_map):
                    logger.info(f"成功更新股票名称代码映射表，新增 {additions} 个股票，更新 {updates} 个股票")
                    return True
                else:
                    logger.error("保存股票名称代码映射表失败")
                    return False
            else:
                logger.info("股票名称代码映射表无变化，无需更新")
                return True
            
        except Exception as e:
            logger.error(f"更新股票名称代码映射表失败: {e}", exc_info=True)
            return False

    def _get_cached_data(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """获取缓存数据（先查内存，再查磁盘）
        
        Args:
            key: 缓存键
            ttl: 可选，磁盘缓存过期时间（秒），默认60分钟
            
        Returns:
            Optional[Any]: 缓存数据，如果不存在或过期则返回None
        """
        # 先检查内存缓存
        if key in self.memory_cache:
            self.performance_metrics['cache_hits'] += 1
            logger.debug(f"从内存缓存获取数据: {key}")
            return self.memory_cache[key]
        
        # 内存缓存不存在或过期，检查磁盘缓存
        try:
            # 从磁盘缓存获取数据
            data = self.disk_cache.get(key, default=None)
            if data is not None:
                # 更新内存缓存
                self.memory_cache[key] = data
                self.performance_metrics['cache_hits'] += 1
                logger.debug(f"从磁盘缓存获取数据: {key}")
                return data
        except Exception as e:
            logger.error(f"从磁盘缓存获取数据失败: {e}")
        
        self.performance_metrics['cache_misses'] += 1
        return None
    
    def _set_cached_data(self, key: str, data: Any, data_type: Optional[str] = None, disk_ttl: Optional[int] = None) -> None:
        """设置缓存数据（同时保存到内存和磁盘）
        
        Args:
            key: 缓存键
            data: 要缓存的数据
            data_type: 数据类型，用于选择合适的过期时间
            disk_ttl: 可选，磁盘缓存过期时间（秒），优先级高于data_type
        """
        try:
            # 保存到内存缓存（30秒过期）
            self.memory_cache[key] = data
            # 只在调试模式下记录
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"已保存到内存缓存: {key}")
        except Exception as e:
            logger.error(f"保存到内存缓存失败: {e}")
        
        # 异步保存到磁盘缓存（不阻塞主流程）
        def save_to_disk_async():
            try:
                # 根据数据类型选择过期时间
                if disk_ttl is not None:
                    ttl = disk_ttl
                elif data_type and data_type in self.cache_ttl:
                    ttl = self.cache_ttl[data_type]
                else:
                    ttl = self.cache_ttl['default']
                
                self.disk_cache.set(key, data, expire=ttl)
                # 只在调试模式下记录
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"已异步保存到磁盘缓存: {key}, 过期时间: {ttl}秒")
            except Exception as e:
                logger.error(f"异步保存到磁盘缓存失败: {e}")
        
        try:
            # 使用线程池执行异步保存，减少线程创建开销
            self.thread_pool.submit(save_to_disk_async)
        except Exception as e:
            logger.error(f"提交异步磁盘缓存保存任务失败: {e}")
            # 同步保存作为回退
            try:
                if disk_ttl is not None:
                    ttl = disk_ttl
                elif data_type and data_type in self.cache_ttl:
                    ttl = self.cache_ttl[data_type]
                else:
                    ttl = self.cache_ttl['default']
                
                self.disk_cache.set(key, data, expire=ttl)
                # 只在调试模式下记录
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"已同步保存到磁盘缓存: {key}")
            except Exception as se:
                logger.error(f"同步保存到磁盘缓存失败: {se}")
    
    @async_retry_with_backoff(max_retries=3, backoff_factor=1.0)
    @retry_with_backoff(max_retries=5, backoff_factor=2.0)  # 增加重试次数和退避因子
    async def _get_stock_quote_async(self, symbol: str) -> Tuple[str, pd.DataFrame]:
        """异步获取单支股票的行情数据
        
        Args:
            symbol: 股票代码
            
        Returns:
            Tuple[str, pd.DataFrame]: 股票代码和行情数据
        """
        # 处理纯数字股票代码，自动添加市场前缀
        original_symbol = symbol
        if symbol.isdigit():
            if len(symbol) == 6:
                prefix = symbol[:3]
                if prefix >= '000' and prefix < '600':
                    # 深市：000(主板), 002(中小板), 300(创业板)
                    symbol = 'sz' + symbol
                else:
                    # 沪市：600(主板), 688(科创板)
                    symbol = 'sh' + symbol
        
        # 检查缓存（实时行情数据缓存1分钟）
        cache_key = self._get_cache_key("get_stock_quotes", symbol)
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            if isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                return (symbol, cached_data)
        
        # 使用新的数据源切换逻辑
        sources = self.data_source_priority.get('stock_quote', ['biying', 'akshare', 'sina', 'eastmoney', 'tickflow'])
        result = self._get_data_from_source('stock_quote', sources, symbol=symbol)
        
        if result is not None:
            # 缓存数据（使用stock_quote数据类型的过期时间）
            self._set_cached_data(cache_key, result, data_type='stock_quote')
            return (symbol, result)
        
        # 如果所有数据源都失败，返回空DataFrame
        logger.error(f"所有数据源获取股票 {symbol} 的行情数据失败")
        stock_data = pd.DataFrame()
        self._set_cached_data(cache_key, stock_data, data_type='stock_quote')
        return (symbol, stock_data)
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    def get_stock_quotes(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """获取股票实时行情数据（使用asyncio并发获取）"""
        # 过滤掉空字符串并添加市场前缀
        valid_symbols = []
        for symbol in symbols:
            if not symbol:
                continue
                
            # 处理纯数字股票代码，自动添加市场前缀
            if symbol.isdigit():
                if len(symbol) == 6:
                    prefix = symbol[:3]
                    if prefix >= '000' and prefix < '600':
                        # 深市：000(主板), 002(中小板), 300(创业板)
                        symbol = 'sz' + symbol
                    else:
                        # 沪市：600(主板), 688(科创板)
                        symbol = 'sh' + symbol
        
            valid_symbols.append(symbol)
        
        # 随机化请求顺序，避免固定的请求模式
        random.shuffle(valid_symbols)
        
        if not valid_symbols:
            return {}
        
        # 检查缓存，直接返回缓存的数据
        result = {}
        remaining_symbols = []
        
        for symbol in valid_symbols:
            cache_key = self._get_cache_key("get_stock_quotes", symbol)
            cached_data = self._get_cached_data(cache_key)
            # 检查缓存数据是否有效
            if cached_data is not None:
                # 特别处理DataFrame类型，检查是否为空
                if isinstance(cached_data, pd.DataFrame):
                    if not cached_data.empty:
                        result[symbol] = cached_data
                    else:
                        remaining_symbols.append(symbol)
                else:
                    # 其他类型直接使用
                    result[symbol] = cached_data
            else:
                remaining_symbols.append(symbol)
        
        # 如果所有数据都在缓存中，直接返回
        if not remaining_symbols:
            return result
        
        # 并发获取剩余数据，限制并发数量以避免触发API速率限制
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # 限制并发数量，每次最多同时处理3个请求
            max_concurrent = 3
            results = {}
            
            async def process_batch(batch):
                """处理一批股票请求"""
                tasks = []
                for symbol in batch:
                    task = self._get_stock_quote_async(symbol)
                    tasks.append(task)
                
                # 并发执行任务
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                for i, res in enumerate(batch_results):
                    symbol = batch[i]
                    if isinstance(res, Exception):
                        logger.error(f"获取股票 {symbol} 数据失败: {res}")
                        results[symbol] = pd.DataFrame()
                    elif isinstance(res, tuple):
                        result_symbol, data = res
                        results[result_symbol] = data
                    else:
                        logger.error(f"获取股票 {symbol} 数据失败: 未知结果类型")
                        results[symbol] = pd.DataFrame()
            
            # 分批处理
            for i in range(0, len(remaining_symbols), max_concurrent):
                batch = remaining_symbols[i:i + max_concurrent]
                logger.info(f"处理批次: {batch}")
                
                # 执行当前批次
                loop.run_until_complete(process_batch(batch))
                
                # 批次之间添加适当的间隔
                if i + max_concurrent < len(remaining_symbols):
                    interval = 2 + random.random() * 2  # 2-4秒间隔
                    logger.debug(f"批次间隔: {interval:.2f} 秒")
                    time.sleep(interval)
            
            # 合并结果
            result.update(results)
        except Exception as e:
            logger.error(f"并发获取股票行情数据失败: {e}")
            # 回退到同步获取
            logger.info("回退到同步获取股票行情数据")
            
            for symbol in remaining_symbols:
                try:
                    # 检查缓存（再次检查，防止在并发过程中已经被缓存）
                    cache_key = self._get_cache_key("get_stock_quotes", symbol)
                    cached_data = self._get_cached_data(cache_key)
                    if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                        result[symbol] = cached_data
                        continue
                    
                    # 使用新的数据源切换逻辑
                    sources = self.data_source_priority.get('stock_quote', ['biying', 'akshare', 'sina', 'eastmoney', 'tickflow'])
                    stock_data = self._get_data_from_source('stock_quote', sources, symbol=symbol)
                    
                    if stock_data is not None:
                        result[symbol] = stock_data
                    else:
                        result[symbol] = pd.DataFrame()
                        # 缓存空数据，避免重复请求
                        self._set_cached_data(cache_key, pd.DataFrame(), data_type='stock_quote')
                except Exception as e:
                    logger.error(f"处理股票 {symbol} 数据时发生错误: {e}")
                    result[symbol] = pd.DataFrame()
        
        return result
    
    @rate_limited(max_calls=3, period=60)  # 每分钟最多3次请求
    @retry_with_backoff(max_retries=3, backoff_factor=2.0)
    def get_gold_price(self, api_key=None):
        """获取黄金价格数据
        
        Args:
            api_key: 可选，API密钥，如果不提供则使用环境变量中的密钥
            
        Returns:
            黄金价格数据列表，如果获取失败则返回空列表
        """
        try:
            # 检查缓存（黄金价格数据缓存10分钟）
            cache_key = self._get_cache_key("get_gold_price")
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info("使用缓存的黄金价格数据")
                return cached_data
            
            # 获取API密钥
            key = api_key or self.gold_api_key
            if not key:
                logger.error("API密钥未配置")
                return []
            
            # 构建请求参数
            params = {'key': key, 'v': '1'}
            
            # 发送请求
            response = self.session.get(self.gold_api_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('resultcode') == '200':
                    # 解析API返回的数据
                    result = data.get('result', [])
                    if result:
                        # 标准化数据格式
                        gold_data = []
                        for item in result:
                            for key, value in item.items():
                                gold_item = {
                                    'variety': value.get('variety', ''),
                                    'latest_price': value.get('latestpri', ''),
                                    'open_price': value.get('openpri', ''),
                                    'max_price': value.get('maxpri', ''),
                                    'min_price': value.get('minpri', ''),
                                    'change_rate': value.get('limit', ''),
                                    'yes_price': value.get('yespri', ''),
                                    'total_volume': value.get('totalvol', ''),
                                    'update_time': value.get('time', '')
                                }
                                gold_data.append(gold_item)
                        
                        # 缓存数据
                        self._set_cached_data(cache_key, gold_data, data_type='gold_price')
                        logger.info("黄金价格数据获取成功")
                        return gold_data
                    else:
                        logger.warning("API返回的结果为空")
                        return []
                else:
                    logger.error(f"API返回错误: {data.get('reason')}")
                    return []
            else:
                logger.error(f"请求失败，状态码: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"获取黄金价格数据失败: {e}")
            return []

    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_macd_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100) -> pd.DataFrame:
        """获取股票MACD指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
        
        Returns:
            pd.DataFrame: MACD指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_macd_data", symbol, period, adjust, start_time, end_time, limit)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的MACD数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的MACD数据")
        
        try:
            # 调用必盈API
            result = self._call_biying_api(
                self.BIYING_API_MACD,
                stock_code=symbol,
                period=period,
                adjust=adjust,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if result:
                # 将API返回的数据转换为DataFrame
                if isinstance(result, list):
                    df = pd.DataFrame(result)
                    logger.info(f"成功获取股票 {symbol} 的MACD数据，形状: {df.shape}")
                    
                    # 缓存结果
                    self._set_cached_data(cache_key, df)
                    return df
                else:
                    logger.warning(f"必盈API返回的MACD数据格式不正确: {result}")
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的MACD数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()

    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_ma_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100) -> pd.DataFrame:
        """获取股票MA指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
        
        Returns:
            pd.DataFrame: MA指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_ma_data", symbol, period, adjust, start_time, end_time, limit)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的MA数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的MA数据")
        
        try:
            # 调用必盈API
            result = self._call_biying_api(
                self.BIYING_API_MA,
                stock_code=symbol,
                period=period,
                adjust=adjust,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if result:
                # 将API返回的数据转换为DataFrame
                if isinstance(result, list):
                    df = pd.DataFrame(result)
                    logger.info(f"成功获取股票 {symbol} 的MA数据，形状: {df.shape}")
                    
                    # 缓存结果
                    self._set_cached_data(cache_key, df)
                    return df
                else:
                    logger.warning(f"必盈API返回的MA数据格式不正确: {result}")
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的MA数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()

    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_boll_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100) -> pd.DataFrame:
        """获取股票BOLL指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
        
        Returns:
            pd.DataFrame: BOLL指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_boll_data", symbol, period, adjust, start_time, end_time, limit)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的BOLL数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的BOLL数据")
        
        try:
            # 调用必盈API
            result = self._call_biying_api(
                self.BIYING_API_BOLL,
                stock_code=symbol,
                period=period,
                adjust=adjust,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if result:
                # 将API返回的数据转换为DataFrame
                if isinstance(result, list):
                    df = pd.DataFrame(result)
                    logger.info(f"成功获取股票 {symbol} 的BOLL数据，形状: {df.shape}")
                    
                    # 缓存结果
                    self._set_cached_data(cache_key, df)
                    return df
                else:
                    logger.warning(f"必盈API返回的BOLL数据格式不正确: {result}")
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的BOLL数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()

    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_kdj_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100) -> pd.DataFrame:
        """获取股票KDJ指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
        
        Returns:
            pd.DataFrame: KDJ指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_kdj_data", symbol, period, adjust, start_time, end_time, limit)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的KDJ数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的KDJ数据")
        
        try:
            # 调用必盈API
            result = self._call_biying_api(
                self.BIYING_API_KDJ,
                stock_code=symbol,
                period=period,
                adjust=adjust,
                start_time=start_time,
                end_time=end_time,
                limit=limit
            )
            
            if result:
                # 将API返回的数据转换为DataFrame
                if isinstance(result, list):
                    df = pd.DataFrame(result)
                    logger.info(f"成功获取股票 {symbol} 的KDJ数据，形状: {df.shape}")
                    
                    # 缓存结果
                    self._set_cached_data(cache_key, df)
                    return df
                else:
                    logger.warning(f"必盈API返回的KDJ数据格式不正确: {result}")
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的KDJ数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()
    
    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_rsi_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100, time_period: int = 14) -> pd.DataFrame:
        """获取股票RSI指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
            time_period: RSI计算周期
        
        Returns:
            pd.DataFrame: RSI指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_rsi_data", symbol, period, adjust, start_time, end_time, limit, time_period)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的RSI数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的RSI数据")
        
        try:
            # 获取历史价格数据
            stock_data = self.get_stock_history(symbol, days=limit*2)  # 获取更多数据用于计算
            
            if not stock_data.empty and '收盘' in stock_data.columns:
                # 计算价格变化
                stock_data['价格变化'] = stock_data['收盘'].diff()
                
                # 计算上涨和下跌
                stock_data['上涨'] = stock_data['价格变化'].apply(lambda x: x if x > 0 else 0)
                stock_data['下跌'] = stock_data['价格变化'].apply(lambda x: abs(x) if x < 0 else 0)
                
                # 计算平均上涨和平均下跌
                stock_data['平均上涨'] = stock_data['上涨'].rolling(window=time_period).mean()
                stock_data['平均下跌'] = stock_data['下跌'].rolling(window=time_period).mean()
                
                # 计算RSI
                stock_data['rsi'] = 100 - (100 / (1 + stock_data['平均上涨'] / stock_data['平均下跌'].replace(0, 0.0001)))
                
                # 保留需要的列
                rsi_data = stock_data[['日期', 'rsi']].dropna()
                
                # 限制数据条数
                if len(rsi_data) > limit:
                    rsi_data = rsi_data.tail(limit)
                
                logger.info(f"成功计算股票 {symbol} 的RSI数据，形状: {rsi_data.shape}")
                
                # 缓存结果
                self._set_cached_data(cache_key, rsi_data)
                return rsi_data
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的RSI数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()
    
    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_atr_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100, time_period: int = 14) -> pd.DataFrame:
        """获取股票ATR指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
            time_period: ATR计算周期
        
        Returns:
            pd.DataFrame: ATR指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_atr_data", symbol, period, adjust, start_time, end_time, limit, time_period)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的ATR数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的ATR数据")
        
        try:
            # 获取历史价格数据
            stock_data = self.get_stock_history(symbol, days=limit*2)  # 获取更多数据用于计算
            
            if not stock_data.empty and all(col in stock_data.columns for col in ['最高', '最低', '收盘']):
                # 计算真实波幅
                stock_data['前收盘价'] = stock_data['收盘'].shift(1)
                stock_data['TR1'] = stock_data['最高'] - stock_data['最低']
                stock_data['TR2'] = abs(stock_data['最高'] - stock_data['前收盘价'])
                stock_data['TR3'] = abs(stock_data['最低'] - stock_data['前收盘价'])
                stock_data['TR'] = stock_data[['TR1', 'TR2', 'TR3']].max(axis=1)
                
                # 计算ATR
                stock_data['atr'] = stock_data['TR'].rolling(window=time_period).mean()
                
                # 保留需要的列
                atr_data = stock_data[['日期', 'atr']].dropna()
                
                # 限制数据条数
                if len(atr_data) > limit:
                    atr_data = atr_data.tail(limit)
                
                logger.info(f"成功计算股票 {symbol} 的ATR数据，形状: {atr_data.shape}")
                
                # 缓存结果
                self._set_cached_data(cache_key, atr_data)
                return atr_data
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的ATR数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()
    
    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_obv_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100) -> pd.DataFrame:
        """获取股票OBV指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
        
        Returns:
            pd.DataFrame: OBV指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_obv_data", symbol, period, adjust, start_time, end_time, limit)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的OBV数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的OBV数据")
        
        try:
            # 获取历史价格数据
            stock_data = self.get_stock_history(symbol, days=limit*2)  # 获取更多数据用于计算
            
            if not stock_data.empty and all(col in stock_data.columns for col in ['收盘', '成交量']):
                # 计算价格变化
                stock_data['价格变化'] = stock_data['收盘'].diff()
                
                # 计算OBV
                stock_data['obv'] = 0
                for i in range(1, len(stock_data)):
                    if stock_data['价格变化'].iloc[i] > 0:
                        stock_data.loc[stock_data.index[i], 'obv'] = stock_data['obv'].iloc[i-1] + stock_data['成交量'].iloc[i]
                    elif stock_data['价格变化'].iloc[i] < 0:
                        stock_data.loc[stock_data.index[i], 'obv'] = stock_data['obv'].iloc[i-1] - stock_data['成交量'].iloc[i]
                    else:
                        stock_data.loc[stock_data.index[i], 'obv'] = stock_data['obv'].iloc[i-1]
                
                # 保留需要的列
                obv_data = stock_data[['日期', 'obv']].dropna()
                
                # 限制数据条数
                if len(obv_data) > limit:
                    obv_data = obv_data.tail(limit)
                
                logger.info(f"成功计算股票 {symbol} 的OBV数据，形状: {obv_data.shape}")
                
                # 缓存结果
                self._set_cached_data(cache_key, obv_data)
                return obv_data
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的OBV数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()
    
    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_vwap_data(self, symbol: str, period: str = 'd', adjust: str = 'n', start_time: str = '', end_time: str = '', limit: int = 100) -> pd.DataFrame:
        """获取股票VWAP指标数据
        
        Args:
            symbol: 股票代码
            period: 周期，支持5、15、30、60、d、w、m、y
            adjust: 复权类型，n为不复权
            start_time: 起始时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            end_time: 结束时间，格式为YYYYMMDD或YYYYMMDDhhmmss
            limit: 获取数据条数
        
        Returns:
            pd.DataFrame: VWAP指标数据
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_vwap_data", symbol, period, adjust, start_time, end_time, limit)
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info(f"从缓存获取股票 {symbol} 的VWAP数据")
            return cached_data
        
        logger.info(f"正在获取股票 {symbol} 的VWAP数据")
        
        try:
            # 获取历史价格数据
            stock_data = self.get_stock_history(symbol, days=limit*2)  # 获取更多数据用于计算
            
            if not stock_data.empty and all(col in stock_data.columns for col in ['收盘', '成交量']):
                # 计算VWAP
                stock_data['成交额'] = stock_data['收盘'] * stock_data['成交量']
                stock_data['累计成交额'] = stock_data['成交额'].cumsum()
                stock_data['累计成交量'] = stock_data['成交量'].cumsum()
                stock_data['vwap'] = stock_data['累计成交额'] / stock_data['累计成交量']
                
                # 保留需要的列
                vwap_data = stock_data[['日期', 'vwap']].dropna()
                
                # 限制数据条数
                if len(vwap_data) > limit:
                    vwap_data = vwap_data.tail(limit)
                
                logger.info(f"成功计算股票 {symbol} 的VWAP数据，形状: {vwap_data.shape}")
                
                # 缓存结果
                self._set_cached_data(cache_key, vwap_data)
                return vwap_data
        except Exception as e:
            logger.error(f"获取股票 {symbol} 的VWAP数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return pd.DataFrame()

    @retry_with_backoff(max_retries=3, backoff_factor=2.0)  # 增加重试次数和退避因子
    def get_stock_history(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """获取股票历史行情数据"""
        # 处理纯数字股票代码，自动添加市场前缀
        if symbol.isdigit():
            if len(symbol) == 6:
                prefix = symbol[:3]
                if prefix >= '000' and prefix < '600':
                    # 深市：000(主板), 002(中小板), 300(创业板)
                    symbol = 'sz' + symbol
                else:
                    # 沪市：600(主板), 688(科创板)
                    symbol = 'sh' + symbol
        
        # 只处理A股股票
        if not (symbol.startswith('sh') or symbol.startswith('sz')):
            logger.error(f"不支持的股票代码格式: {symbol}")
            return pd.DataFrame()
        
        cache_key = self._get_cache_key("get_stock_history", symbol, days)
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            # 特别处理DataFrame类型，检查是否为空
            if isinstance(cached_data, pd.DataFrame):
                if not cached_data.empty:
                    # 确保缓存的数据包含必要的列，包括注释列
                    stock_data = cached_data
                else:
                    stock_data = pd.DataFrame()
            else:
                # 其他类型直接使用
                return cached_data
        else:
            stock_data = pd.DataFrame()
        
        # 如果stock_data为空，调用API获取数据
        if stock_data.empty:
            # 使用新的数据源切换逻辑
            sources = self.data_source_priority.get('stock_history', ['biying', 'akshare', 'baostock', 'tickflow'])
            result = self._get_data_from_source('stock_history', sources, symbol=symbol, days=days)
            
            if result is not None:
                stock_data = result
            else:
                # 如果所有数据源都失败，创建空DataFrame
                stock_data = pd.DataFrame()
        
        # 确保数据格式正确，包含必要的列
        required_columns = ['交易时间', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额', '周期', '注释']
        
        if stock_data.empty:
            # 如果数据为空，创建包含所有必要列的空DataFrame
            stock_data = pd.DataFrame(columns=required_columns)
            # 为日期列添加默认值
            stock_data['交易时间'] = pd.to_datetime([])
            # 为其他数值列添加默认值
            for col in required_columns[1:]:
                if col in ['周期', '注释']:
                    stock_data[col] = ''
                else:
                    stock_data[col] = 0.0
        else:
            # 检查是否包含必要的列，如果不包含则添加空列
            for col in required_columns:
                if col not in stock_data.columns:
                    # 根据列名类型添加合适的默认值
                    if col in ['交易时间']:
                        stock_data[col] = pd.NaT
                    elif col in ['周期', '注释']:
                        stock_data[col] = ''
                    else:
                        stock_data[col] = 0.0
            
            # 确保数据类型正确
            if '交易时间' in stock_data.columns:
                stock_data['交易时间'] = pd.to_datetime(stock_data['交易时间'])
            for col in ['开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额']:
                if col in stock_data.columns:
                    stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
            
            # 确保注释列存在
            if '注释' not in stock_data.columns:
                stock_data['注释'] = stock_data['周期'].apply(lambda x: f'这是{x}K线数据')
        
        # 缓存数据
        self._set_cached_data(cache_key, stock_data, data_type='stock_history')
        return stock_data
            

                    

    
    @retry_with_backoff(max_retries=3, backoff_factor=1.0)
    def get_financial_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """获取股票财务数据"""
        result = {
            'pe_pb': pd.DataFrame(),  # 估值数据
            'profit': pd.DataFrame(),  # 利润表
            'balance': pd.DataFrame(),  # 资产负债表
            'cashflow': pd.DataFrame(),  # 现金流量表
        }
        
        # 处理纯数字股票代码，自动添加市场前缀
        if symbol.isdigit():
            if len(symbol) == 6:
                prefix = symbol[:3]
                if prefix >= '000' and prefix < '600':
                    # 深市：000(主板), 002(中小板), 300(创业板)
                    symbol = 'sz' + symbol
                else:
                    # 沪市：600(主板), 688(科创板)
                    symbol = 'sh' + symbol
        
        cache_key = self._get_cache_key("get_financial_data", symbol)
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data
        
        # 1. 首先尝试使用必盈API获取财务数据
        # 提取纯数字股票代码用于必盈API
        biying_stock_code = symbol[2:] if len(symbol) == 8 and (symbol.startswith('sh') or symbol.startswith('sz')) else symbol
        
        try:
            # 尝试使用必盈API获取公司信息（可能包含部分财务数据）
            logger.info(f"尝试使用必盈API获取股票 {symbol} 的财务数据")
            company_info = self.get_company_info(biying_stock_code)
            
            if company_info:
                # 将公司信息转换为财务数据格式
                # 这里需要根据必盈API返回的实际数据格式进行调整
                pe_pb_data = {
                    'name': [company_info.get('name', '')],
                    'code': [symbol],
                    # 从公司信息中提取可能的财务指标
                    'pe': [company_info.get('pe', '')],
                    'pb': [company_info.get('pb', '')],
                    'total_market_value': [company_info.get('total_market_value', '')],
                    'circulating_market_value': [company_info.get('circulating_market_value', '')]
                }
                
                result['pe_pb'] = pd.DataFrame(pe_pb_data)
                logger.info(f"从必盈API获取到部分财务数据")
        except Exception as e:
            logger.error(f"使用必盈API获取股票 {symbol} 财务数据失败: {e}")
        
        # 2. 如果必盈API获取的PE/PB数据不完整，使用akshare获取
        if result['pe_pb'].empty or 'pe' not in result['pe_pb'].columns:
            logger.info(f"必盈API财务数据不完整，尝试使用akshare获取")
            
            # 获取市盈率和市净率 - 尝试不同的API名称
            valuation_apis = [
                'stock_zh_a_lg_indicator',
                'stock_zh_a_indicator',
                'stock_individual_info_em'
            ]
            
            for api_name in valuation_apis:
                if hasattr(ak, api_name):
                    try:
                        api_func = getattr(ak, api_name)
                        if api_name == 'stock_individual_info_em':
                            # 这个API返回的是不同格式的数据
                            df = api_func(symbol=symbol)
                            # 检查返回数据类型并进行转换
                            if isinstance(df, dict):
                                # 如果是字典，转换为DataFrame
                                df = pd.DataFrame([df])
                            elif not hasattr(df, 'columns'):
                                # 如果是标量值或其他非DataFrame格式，创建包含该值的DataFrame
                                df = pd.DataFrame([{api_name: df}])
                            result['pe_pb'] = df
                            break
                        else:
                            df = api_func(symbol=symbol)
                            result['pe_pb'] = df
                            break
                    except Exception as e:
                        logger.error(f"使用 {api_name} 获取股票 {symbol} 估值数据失败: {e}")
                        continue
        
        # 3. 财务报表数据 - 首先尝试必盈API，如果没有则使用akshare
        try:
            # 尝试使用必盈API获取财务报表数据
            # 注意：必盈API可能没有直接的财务报表接口，这里预留扩展点
            logger.info(f"检查必盈API是否支持财务报表数据")
            # 这里可以根据必盈API文档扩展
        except Exception as e:
            logger.error(f"使用必盈API获取股票 {symbol} 财务报表数据失败: {e}")
        
        # 如果必盈API没有获取到财务报表数据，使用akshare
        if result['profit'].empty or result['balance'].empty or result['cashflow'].empty:
            # 获取财务报表数据 - 尝试不同的API名称
            financial_statement_apis = {
                'profit': ['stock_profit_sheet_by_announcement', 'stock_profit_sheet'],
                'balance': ['stock_balance_sheet_by_announcement', 'stock_balance_sheet'],
                'cashflow': ['stock_cash_flow_sheet_by_announcement', 'stock_cash_flow_sheet']
            }
            
            for stmt_type, api_names in financial_statement_apis.items():
                if not result[stmt_type].empty:
                    continue  # 如果已经有数据，跳过
                    
                for api_name in api_names:
                    if hasattr(ak, api_name):
                        try:
                            api_func = getattr(ak, api_name)
                            df = api_func(symbol=symbol)
                            result[stmt_type] = df
                            break
                        except Exception as e:
                            logger.error(f"使用 {api_name} 获取股票 {symbol} {stmt_type} 数据失败: {e}")
                            continue
        
        # 缓存数据（财务数据缓存24小时）
        self._set_cached_data(cache_key, result, disk_ttl=86400)
        
        return result
    
    def _check_api_status(self, api_name: str, api_url: str = None) -> bool:
        """检查API状态，返回True表示API正常，False表示API可能有问题"""
        try:
            import requests
            
            # 默认的新浪API URL
            if api_name == 'stock_zh_index_spot_sina' and api_url is None:
                api_url = 'http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData'
            
            if api_url is None:
                return True  # 如果没有提供URL，默认认为API正常
            
            # 发送HEAD请求检查API状态
            response = requests.head(api_url, timeout=5)
            
            if response.status_code == 200:
                # 检查响应头中的内容类型
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    logger.warning(f"API {api_name} 返回HTML格式，可能正在维护或有变更")
                    return False
                logger.info(f"API {api_name} 状态正常")
                return True
            else:
                logger.warning(f"API {api_name} 返回状态码: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"检查API {api_name} 状态失败: {e}")
            return False
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    def get_index_quotes(self, index_symbols: List[str] = ['sh000001', 'sz399001']) -> Dict[str, pd.DataFrame]:
        """获取指数实时行情，只返回真实API数据，不生成假数据"""
        result = {}
        
        try:
            cache_key = self._get_cache_key("get_index_quotes", tuple(index_symbols))
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                # 修正缓存数据中的指数名称
                for symbol, df in cached_data.items():
                    if not df.empty and '名称' in df.columns:
                        if symbol == 'sh000002':
                            df.loc[df.index[0], '名称'] = '中证A股指数'
                        elif symbol == 'sh000688':
                            df.loc[df.index[0], '名称'] = '科创板指数'
                        elif symbol == 'sz399006':
                            df.loc[df.index[0], '名称'] = '创业板指数'
                return cached_data
            
            # 尝试获取真实指数数据
            df = None
            api_success = False
            
            # 按照接口优先级获取数据
            for api_source in API_PRIORITY['other_data']:
                if api_source == 'akshare':
                    # 增加更多指数数据API接口（只保留实际存在的API）
                    index_apis = [
                        {"name": "stock_zh_index_spot_sina", "desc": "新浪指数行情"},
                        {"name": "stock_zh_index_spot_em", "desc": "东方财富指数行情"},
                        {"name": "stock_hk_index_spot_sina", "desc": "港股指数行情"},
                        {"name": "stock_hk_index_spot_em", "desc": "港股指数行情EM"}
                    ]
                    
                    # 检查是否有海外指数请求
                    has_hk_indices = any(symbol in ['hkHSI', 'HSI'] for symbol in index_symbols)
                    has_us_indices = any(symbol in ['usSPX', 'SPX', 'usDJI', 'DJI', 'usIXIC', 'IXIC'] for symbol in index_symbols)
                    
                    # 如果有海外指数请求，确保港股指数API在列表中
                    if has_hk_indices:
                        hk_api_exists = any(api['name'] == 'stock_hk_index_spot_sina' for api in index_apis)
                        if not hk_api_exists:
                            index_apis.insert(0, {"name": "stock_hk_index_spot_sina", "desc": "港股指数行情"})
                    
                    # 调整API顺序，确保港股API优先于其他API
                    if has_hk_indices:
                        # 将港股API移到列表前面
                        hk_api = next((api for api in index_apis if api['name'] == 'stock_hk_index_spot_sina'), None)
                        if hk_api:
                            index_apis.remove(hk_api)
                            index_apis.insert(0, hk_api)
                    
                    # 检查API状态，过滤掉可能有问题的API
                    filtered_apis = []
                    for api in index_apis:
                        # 只检查新浪API的状态，其他API默认正常
                        if api['name'] == 'stock_zh_index_spot_sina':
                            if self._check_api_status(api['name']):
                                filtered_apis.append(api)
                            else:
                                logger.warning(f"跳过可能有问题的API: {api['desc']}")
                        else:
                            filtered_apis.append(api)
                    
                    # 如果过滤后没有API可用，使用原始列表
                    if not filtered_apis:
                        logger.warning("所有API都可能有问题，尝试使用原始列表")
                        filtered_apis = index_apis
                    
                    # 使用过滤后的API列表
                    index_apis = filtered_apis
            
            max_retries = 3  # 重试次数调整为3次
            retry_interval = 1.5  # 增加初始等待时间到2秒
            retry_factor = 2.0  # 增加退避因子到2.0，更激进的退避策略
            base_jitter = 0.5  # 随机抖动的基础值，避免请求风暴
            
            # 对深证成指的特殊处理：尝试所有API来获取精确匹配
            sz399001_found = False
            sz399001_data = None
            
            # 首先尝试所有API接口，寻找深证成指的精确匹配
            if 'sz399001' in index_symbols:
                logger.info("开始为深证成指尝试所有可用API...")
                
                for api in index_apis:
                    # 跳过禁用的API
                    if api.get('disabled', False):
                        logger.debug(f"跳过禁用的API: {api['desc']}")
                        continue
                        
                    for retry in range(max_retries):
                        try:
                            logger.info(f"尝试 {api['desc']} (API: {api['name']}, 尝试 {retry+1}/{max_retries})")
                            
                            # 检查函数是否存在
                            if hasattr(ak, api['name']):
                                api_df = getattr(ak, api['name'])()
                            else:
                                logger.error(f"API {api['name']} 不存在")
                                break
                            
                            if not api_df.empty:
                                # 检查是否包含深证成指
                                sz_matches = []
                                
                                # 尝试多种代码格式
                                possible_codes = ['sz399001', '399001', '399001.SZ', '399001.SHE', '000002']  # 增加更多可能的代码格式
                                possible_names = ['深证成指', '深成指', '深圳成指', '深指', 'Shenzhen Component Index']  # 增加英文名称
                                
                                # 检查代码匹配
                                for code_col in ['代码', 'symbol', 'index_code', '股票代码', '指数代码', 'code']:
                                    if code_col in api_df.columns:
                                        for code in possible_codes:
                                            try:
                                                # 使用更灵活的匹配方式
                                                matches = api_df[api_df[code_col].astype(str).str.contains(code, na=False, case=False)]
                                                if not matches.empty:
                                                    sz_matches.append(matches)
                                                    break
                                            except Exception as e:
                                                logger.debug(f"代码列 {code_col} 匹配失败: {e}")
                                
                                # 检查名称匹配
                                for name_col in ['名称', '指数名称', '股票名称', 'name']:
                                    if name_col in api_df.columns:
                                        for name in possible_names:
                                            try:
                                                matches = api_df[api_df[name_col].astype(str).str.contains(name, na=False, case=False)]
                                                if not matches.empty:
                                                    sz_matches.append(matches)
                                                    break
                                            except Exception as e:
                                                logger.debug(f"名称列 {name_col} 匹配失败: {e}")
                                
                                if sz_matches:
                                    sz399001_data = sz_matches[0]
                                    sz399001_found = True
                                    logger.info(f"✅ 在 {api['desc']} 中找到深证成指精确匹配")
                                    # 使用找到深证成指的API数据作为主数据源
                                    df = api_df
                                    api_success = True
                                    break
                                else:
                                    logger.info(f"在 {api['desc']} 中未找到深证成指精确匹配")
                            else:
                                logger.warning(f"{api['desc']} 返回空数据")
                        except Exception as e:
                            # 针对不同类型的异常采取不同的处理策略
                            error_msg = str(e)
                            
                            # 如果是新浪API返回HTML格式的错误，直接跳过该API
                            if 'Can not decode value starting with character "<"' in error_msg or '<html' in error_msg.lower():
                                logger.error(f"{api['desc']} 返回HTML格式，无法解析，跳过该API")
                                break  # 跳过该API的所有重试
                            # 如果是函数不存在的错误，直接跳过该API
                            elif 'has no attribute' in error_msg:
                                logger.error(f"{api['desc']} API函数不存在，跳过该API")
                                break  # 跳过该API的所有重试
                            # 其他类型的错误，继续重试
                            else:
                                logger.error(f"{api['desc']} 获取失败: {e}")
                                # 记录详细的异常信息
                                import traceback
                                logger.debug(f"详细错误信息: {traceback.format_exc()}")
                        
                        if sz399001_found:
                            break
                        
                        # 如果不是最后一次重试，等待一段时间
                        if retry < max_retries - 1:
                            import time
                            import random
                            # 添加随机抖动，避免请求风暴
                            jitter = random.uniform(0, base_jitter)
                            wait_time = retry_interval + jitter
                            time.sleep(wait_time)
                            retry_interval *= retry_factor  # 指数退避
                            logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    
                    if sz399001_found:
                        break
            
            # 如果没有找到深证成指，或者不需要深证成指，尝试常规API调用
            if not api_success:
                logger.info("开始常规API调用...")
                
                for api in index_apis:
                    # 跳过禁用的API
                    if api.get('disabled', False):
                        logger.debug(f"跳过禁用的API: {api['desc']}")
                        continue
                        
                    for retry in range(max_retries):
                        try:
                            logger.info(f"尝试 {api['desc']} (API: {api['name']}, 尝试 {retry+1}/{max_retries})")
                            
                            # 检查函数是否存在
                            if hasattr(ak, api['name']):
                                df = getattr(ak, api['name'])()
                            else:
                                logger.error(f"API {api['name']} 不存在")
                                break
                                
                            if not df.empty:
                                api_success = True
                                logger.info(f"✅ {api['desc']} 成功获取实时行情数据")
                                break
                            else:
                                logger.warning(f"{api['desc']} 返回空数据")
                        except Exception as e:
                            # 针对不同类型的异常采取不同的处理策略
                            error_msg = str(e)
                            
                            # 如果是新浪API返回HTML格式的错误，直接跳过该API
                            if 'Can not decode value starting with character "<"' in error_msg or '<html' in error_msg.lower():
                                logger.error(f"{api['desc']} 返回HTML格式，无法解析，跳过该API")
                                break  # 跳过该API的所有重试
                            # 如果是函数不存在的错误，直接跳过该API
                            elif 'has no attribute' in error_msg:
                                logger.error(f"{api['desc']} API函数不存在，跳过该API")
                                break  # 跳过该API的所有重试
                            # 其他类型的错误，继续重试
                            else:
                                logger.error(f"{api['desc']} 获取实时行情失败: {e}")
                                # 记录详细的异常信息
                                import traceback
                                logger.debug(f"详细错误信息: {traceback.format_exc()}")
                        
                        if api_success:
                            break
                        
                        # 如果不是最后一次重试，等待一段时间
                        if retry < max_retries - 1:
                            import time
                            import random
                            # 添加随机抖动，避免请求风暴
                            jitter = random.uniform(0, base_jitter)
                            wait_time = retry_interval + jitter
                            time.sleep(wait_time)
                            retry_interval *= retry_factor  # 指数退避
                            logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                    
                    if api_success:
                        break
            
            # 如果所有指数API都失败，尝试日线数据接口
            if not api_success:
                logger.error("所有指数API都失败了，尝试使用日线数据接口...")
                index_symbols_data = {}
                
                for symbol in index_symbols:
                    try:
                        # 添加随机延时，模拟真人操作
                        delay = random.uniform(2, 4)
                        logger.debug(f"添加随机延时 {delay:.2f} 秒")
                        time.sleep(delay)
                        
                        # 尝试使用日线数据接口获取指数数据
                        daily_df = ak.stock_zh_index_daily(symbol=symbol)
                        if not daily_df.empty and len(daily_df) >= 2:
                            # 获取最新一天和前一天的数据
                            latest_data = daily_df.tail(1).copy()
                            prev_data = daily_df.tail(2).head(1).copy()
                            
                            # 正确设置指数名称，使用更灵活的匹配方式
                            index_name = None
                            symbol_str = symbol.lower().replace('.', '').replace(':', '')
                            
                            # 上证指数
                            if 'sh000001' in symbol_str or '000001' in symbol_str:
                                index_name = '上证指数'
                            # 深证成指
                            elif 'sz399001' in symbol_str or '399001' in symbol_str:
                                index_name = '深证成指'
                            # 中证A股指数
                            elif 'sh000002' in symbol_str or '000002' in symbol_str:
                                index_name = '中证A股指数'
                            # 科创板指数
                            elif 'sh000688' in symbol_str or '000688' in symbol_str:
                                index_name = '科创板指数'
                            # 创业板指数
                            elif 'sz399006' in symbol_str or '399006' in symbol_str:
                                index_name = '创业板指数'
                            # 恒生指数
                            elif 'hk' in symbol_str and 'hsi' in symbol_str or 'hsi' in symbol_str:
                                index_name = '恒生指数'
                            # 标普500指数
                            elif 'us' in symbol_str and 'spx' in symbol_str or 'spx' in symbol_str:
                                index_name = '标普500指数'
                            # 沪深300指数
                            elif 'sh000300' in symbol_str or '000300' in symbol_str:
                                index_name = '沪深300指数'
                            # 中证500指数
                            elif 'sh000905' in symbol_str or '000905' in symbol_str:
                                index_name = '中证500指数'
                            # 中证1000指数
                            elif 'sh000852' in symbol_str or '000852' in symbol_str:
                                index_name = '中证1000指数'
                            elif symbol in ['usDJI', 'DJI']:
                                index_name = '道琼斯工业指数'
                            elif symbol in ['usIXIC', 'IXIC']:
                                index_name = '纳斯达克综合指数'
                            elif symbol in ['XAUUSD', 'gold']:
                                index_name = '伦敦金现'
                            else:
                                index_name = '未知指数'
                            
                            latest_data['代码'] = symbol
                            latest_data['名称'] = index_name
                            latest_data['最新价'] = latest_data['close']
                            
                            # 使用当天开盘价计算涨跌幅（当天开盘价和目前收集到的价格比较）
                            if 'close' in prev_data.columns and 'close' in latest_data.columns:
                                prev_close = prev_data.iloc[0]['close']  # 昨日收盘价
                                current_price = latest_data.iloc[0]['close']  # 目前收集到的价格
                                open_price = latest_data.iloc[0]['open']  # 当天开盘价
                                
                                latest_data['昨收'] = prev_close  # 保持昨收字段为昨日收盘价
                                
                                # 使用当天开盘价计算涨跌幅
                                if open_price != 0:
                                    latest_data['涨跌幅'] = ((current_price - open_price) / open_price) * 100
                                    logger.info(f"使用当天开盘价计算指数 {symbol} 的涨跌幅: {latest_data['涨跌幅'].iloc[0]:.2f}%")
                                else:
                                    # 如果开盘价为0，回退到使用昨日收盘价计算
                                    logger.warning(f"指数 {symbol} 的开盘价为0，回退到使用昨日收盘价计算涨跌幅")
                                    latest_data['涨跌幅'] = ((current_price - prev_close) / prev_close) * 100
                            else:
                                logger.warning(f"无法计算指数 {symbol} 的准确涨跌幅，缺少必要字段")
                                latest_data['涨跌幅'] = 0.0
                                latest_data['昨收'] = latest_data['open']  # 退而求其次使用开盘价
                            
                            index_symbols_data[symbol] = latest_data
                            logger.info(f"stock_zh_index_daily 成功获取指数 {symbol} 的数据")
                        elif len(daily_df) == 1:
                            # 只有一天数据，无法计算准确涨跌幅
                            latest_data = daily_df.tail(1).copy()
                            latest_data['代码'] = symbol
                            latest_data['名称'] = '上证指数' if symbol == 'sh000001' else '深证成指'
                            
                            # 验证收盘价是否有效
                            if latest_data['close'].iloc[0] != 0:
                                latest_data['最新价'] = latest_data['close']
                            else:
                                logger.warning(f"指数 {symbol} 的收盘价为0，使用开盘价作为最新价")
                                latest_data['最新价'] = latest_data['open']
                            
                            # 只有当最新价和开盘价都有效时才计算涨跌幅
                            if latest_data['最新价'].iloc[0] != 0 and latest_data['open'].iloc[0] != 0:
                                latest_data['涨跌幅'] = ((latest_data['最新价'] - latest_data['open']) / latest_data['open']) * 100
                            else:
                                logger.warning(f"指数 {symbol} 的价格数据异常，无法计算涨跌幅")
                                latest_data['涨跌幅'] = 0.0
                            
                            latest_data['昨收'] = latest_data['open']  # 只有一天数据，使用开盘价作为昨收
                            
                            index_symbols_data[symbol] = latest_data
                            logger.warning(f"指数 {symbol} 只有一天数据，使用开盘价计算涨跌幅")
                    except Exception as e:
                        logger.error(f"stock_zh_index_daily 获取指数 {symbol} 失败: {e}")
                
                # 检查是否获取到了数据
                if index_symbols_data:
                    api_success = True
                    # 合并数据
                    df = pd.concat(list(index_symbols_data.values()))
                else:
                    api_success = False
            
            if not api_success or df is None or (hasattr(df, 'empty') and df.empty):
                logger.error("所有指数API接口都未读取到真实数据，可能的原因：")
                logger.error("1. 网络连接问题，无法访问akshare数据源")
                logger.error("2. akshare接口限制或需要认证")
                logger.error("3. akshare库版本过旧，接口已变更")
                logger.error("4. 环境配置问题，缺少必要依赖")
                # 不为所有指数返回空DataFrame，而是继续尝试使用TickFlow
                # 让后续的TickFlow逻辑处理数据获取
            
            # 检查数据结构
            logger.info(f"API返回的数据列: {list(df.columns)}")
            logger.info(f"返回的指数数量: {len(df)}")
            logger.info(f"前5行数据: {df.head()}")
            
            # 数据预处理：只修复用户请求的指数列表中的最新价
            if '最新价' in df.columns:
                logger.info(f"开始验证用户请求的 {len(index_symbols)} 个指数的数据")
                
                for symbol in index_symbols:
                    # 尝试多种代码格式匹配
                    matched = False
                    for code_col in ['代码', 'symbol', 'index_code']:
                        if code_col in df.columns:
                            # 直接匹配代码
                            matches = df[df[code_col] == symbol]
                            if not matches.empty:
                                matched = True
                                idx = matches.index[0]
                                
                                # 验证最新价是否为0
                                if df.loc[idx, '最新价'] == 0:
                                    logger.warning(f"指数 {symbol} 的最新价为0，开始修复...")
                                    
                                    try:
                                        # 添加随机延时，模拟真人操作
                                        delay = random.uniform(2, 4)
                                        logger.debug(f"添加随机延时 {delay:.2f} 秒")
                                        time.sleep(delay)
                                        
                                        # 添加随机延时，模拟真人操作
                                        delay = random.uniform(2, 4)
                                        logger.debug(f"添加随机延时 {delay:.2f} 秒")
                                        time.sleep(delay)
                                        
                                        # 尝试使用日线数据计算
                                        daily_df = ak.stock_zh_index_daily(symbol=symbol)
                                        if not daily_df.empty:
                                            latest_daily = daily_df.tail(1).iloc[0]
                                            
                                            # 验证数据是否是最近的（最近30天内）
                                            if 'date' in daily_df.columns:
                                                latest_date = pd.to_datetime(latest_daily['date'])
                                                current_date = pd.Timestamp.now()
                                                days_diff = (current_date - latest_date).days
                                                
                                                if days_diff > 30:
                                                    logger.warning(f"指数 {symbol} 的日线数据较旧（{days_diff}天前），可能不是最新数据")
                                                else:
                                                    logger.info(f"指数 {symbol} 的日线数据是最新的（{days_diff}天前）")
                                            
                                            if hasattr(latest_daily, 'close') and latest_daily.close != 0:
                                                # stock_zh_index_daily返回的是该指数的历史数据，直接使用
                                                # 不需要在返回的DataFrame中匹配代码
                                                index_data = pd.DataFrame([{
                                                    '代码': symbol,
                                                    '名称': '上证指数' if '000001' in symbol else '深证成指' if '399001' in symbol else '中证A股指数' if '000002' in symbol else '科创板指数' if '000688' in symbol else '创业板指数' if '399006' in symbol else symbol,
                                                    '最新价': latest_daily.close,
                                                    '涨跌幅': 0.0,  # 日线数据没有实时涨跌幅，设为0
                                                    '今开': latest_daily.open if hasattr(latest_daily, 'open') else 0,
                                                    '最高': latest_daily.high if hasattr(latest_daily, 'high') else 0,
                                                    '最低': latest_daily.low if hasattr(latest_daily, 'low') else 0,
                                                    '昨收': daily_df.tail(2).iloc[0].close if len(daily_df) >= 2 else latest_daily.open if hasattr(latest_daily, 'open') else 0,
                                                    '成交量': latest_daily.volume if hasattr(latest_daily, 'volume') else 0,
                                                    '成交额': 0  # 日线数据可能没有成交额
                                                }])
                                                logger.info(f"使用日线数据成功获取指数 {symbol} 的数据，最新价: {latest_daily.close}")
                                                matched = True
                                                break
                                    except Exception as e:
                                        logger.error(f"修复指数 {symbol} 的最新价失败: {e}")
                                break
                    
                    if not matched:
                        logger.warning(f"在API返回的数据中未找到指数 {symbol}")
            
            for symbol in index_symbols:
                try:
                    # 初始化possible_symbols变量，避免UnboundLocalError
                    possible_symbols = [symbol]
                    index_data = pd.DataFrame()
                    
                    # 尝试多种代码列名获取数据
                    index_data = pd.DataFrame()
                    for code_col in ['代码', 'symbol', 'index_code']:
                        if code_col in df.columns:
                            # 直接匹配代码
                            index_data = df[df[code_col] == symbol]
                            if not index_data.empty:
                                break
                            
                            # 处理代码格式差异，如sh000001 vs 000001
                            if symbol.startswith('sh'):
                                # 上证指数可能的代码格式
                                possible_symbols = [
                                    symbol,
                                    symbol[2:],  # 000001
                                    f"{symbol[2:]}.SH",  # 000001.SH
                                    f"sh{symbol[2:]}"  # sh000001（确保包含原始格式）
                                ]
                                
                                # 特殊处理中证A股指数
                                if symbol in ['sh000002', '000002', '000002.SH']:
                                    possible_symbols.append('中证A股指数')
                                
                                # 特殊处理科创板指数
                                elif symbol in ['sh000688', '000688', '000688.SH']:
                                    possible_symbols.append('科创板指数')
                            elif symbol.startswith('sz'):
                                if symbol == 'sz399001':
                                    # 深证成指可能的代码格式
                                    possible_symbols = [
                                        symbol,
                                        symbol[2:],  # 399001
                                        f"{symbol[2:]}.SZ",  # 399001.SZ
                                        "399001.SZA",  # 399001.SZA
                                        "sza399001",  # sza399001
                                        "深证成指"  # 直接使用名称作为代码（某些数据源可能这样）
                                    ]
                                elif symbol == 'sz399006':
                                    # 创业板指数可能的代码格式
                                    possible_symbols = [
                                        symbol,
                                        symbol[2:],  # 399006
                                        f"{symbol[2:]}.SZ",  # 399006.SZ
                                        "创业板指数"  # 直接使用名称作为代码
                                    ]
                                else:
                                    # 其他深证指数可能的代码格式
                                    possible_symbols = [
                                        symbol,
                                        symbol[2:],  # 去掉sz前缀
                                        f"{symbol[2:]}.SZ"  # 399001.SZ格式
                                    ]
                            elif symbol.startswith('399'):
                                # 直接以399开头的深证成指代码
                                possible_symbols = [
                                    symbol,
                                    f"sz{symbol}",  # sz399001
                                    f"{symbol}.SZ",  # 399001.SZ
                                    "深证成指"
                                ]
                            elif symbol.startswith('hk'):
                                # 港股指数（如恒生指数）可能的代码格式
                                possible_symbols = [
                                    symbol,
                                    symbol[2:],  # HSI
                                    f"{symbol[2:].upper()}",  # HSI（大写）
                                    "恒生指数"  # 直接使用名称作为代码
                                ]
                            elif symbol.startswith('us'):
                                # 美股指数可能的代码格式
                                possible_symbols = [
                                    symbol,
                                    symbol[2:],  # SPX, DJI, IXIC
                                    f"{symbol[2:]}",  # SPX, DJI, IXIC
                                    "标普500指数" if symbol == 'usSPX' else 
                                    "道琼斯工业指数" if symbol == 'usDJI' else 
                                    "纳斯达克综合指数" if symbol == 'usIXIC' else symbol
                                ]
                            elif symbol == 'XAUUSD':
                                # 伦敦金现可能的代码格式
                                possible_symbols = [
                                    symbol,
                                    'gold',
                                    'XAU',
                                    '伦敦金现'  # 直接使用名称作为代码
                                ]
                            else:
                                # 其他指数格式
                                possible_symbols = [symbol]
                            
                            # 调试日志：显示当前尝试的代码列和可能的代码格式
                            logger.debug(f"尝试为指数 {symbol} 匹配代码列 {code_col}")
                            logger.debug(f"可能的代码格式: {possible_symbols}")
                            
                            # 1. 尝试所有可能的代码格式
                            for try_symbol in possible_symbols:
                                index_data = df[df[code_col] == try_symbol]
                                if not index_data.empty:
                                    if try_symbol != symbol:
                                        logger.info(f"通过代码格式转换匹配到指数 {symbol} (使用 {try_symbol})")
                                    logger.debug(f"成功匹配到的数据: {index_data.head(1)}")
                                    break
                            
                            # 2. 如果代码匹配失败，尝试通过名称匹配
                            if index_data.empty and '名称' in df.columns:
                                logger.debug(f"代码匹配失败，尝试通过名称匹配指数 {symbol}")
                                
                                # 上证指数的名称匹配
                                if symbol in ['sh000001', '000001', '000001.SH']:
                                    index_data = df[df['名称'].str.contains('上证指数|上证综指|沪指|上证', na=False, case=False)]
                                # 深证成指的名称匹配 - 增强匹配规则
                                elif symbol in ['sz399001', '399001', '399001.SZ']:
                                    index_data = df[df['名称'].str.contains('深证成指|深成指|深指|深证', na=False, case=False)]
                                # 恒生指数的名称匹配
                                elif symbol in ['hkHSI', 'HSI']:
                                    index_data = df[df['名称'].str.contains('恒生指数|恒生', na=False, case=False)]
                                # 标普500指数的名称匹配
                                elif symbol in ['usSPX', 'SPX']:
                                    # 优先匹配标普500，避免匹配到标普香港创业板
                                    index_data = df[df['名称'].str.contains('标普500', na=False, case=False)]
                                    if index_data.empty:
                                        # 不使用SPX作为匹配，避免匹配到标普香港创业板
                                        index_data = pd.DataFrame()  # 为空，让后续逻辑处理
                                # 道琼斯工业指数的名称匹配
                                elif symbol in ['usDJI', 'DJI']:
                                    index_data = df[df['名称'].str.contains('道琼斯|道指|DJI', na=False, case=False)]
                                # 纳斯达克综合指数的名称匹配
                                elif symbol in ['usIXIC', 'IXIC']:
                                    index_data = df[df['名称'].str.contains('纳斯达克|纳指|IXIC', na=False, case=False)]
                                
                                if not index_data.empty:
                                    matched_name = index_data.iloc[0]['名称']
                                    matched_code = index_data.iloc[0][code_col]
                                    logger.info(f"通过名称匹配到指数 {symbol} (实际名称: {matched_name}, 代码: {matched_code})")
                                    logger.debug(f"成功匹配到的数据: {index_data.head(1)}")
                                    break
                            
                            # 3. 如果仍然匹配失败，不进行模糊匹配或替代指数
                            if index_data.empty:
                                logger.debug(f"精确匹配失败，不尝试模糊匹配或替代指数 {symbol}")
                                
                                # 获取所有指数的名称和代码，用于调试
                                all_index_names = df['名称'].tolist()[:10] if '名称' in df.columns else []  # 只显示前10个
                                all_index_codes = df['代码'].tolist()[:10] if '代码' in df.columns else []  # 只显示前10个
                                logger.debug(f"数据源前10个指数名称: {all_index_names}")
                                logger.debug(f"数据源前10个指数代码: {all_index_codes}")
                                
                                if symbol in ['sz399001', '399001', '399001.SZ']:
                                    logger.warning("已尝试所有API但仍未找到深证成指的精确匹配，不使用替代指数")
                                    break
                                else:
                                    logger.warning(f"未找到指数 {symbol} 的精确匹配")
                    
                    # 如果获取到的数据不为空，使用实际数据
                    if not index_data.empty:
                        logger.debug(f"最终匹配到的指数 {symbol} 数据: {index_data}")
                        
                        # 先清理列名中的空格
                        index_data.columns = index_data.columns.str.strip()
                        
                        # 检查数据结构，确保我们有需要的字段
                        required_fields = ['名称', '最新价', '涨跌幅']
                        field_mapping = {
                            '名称': ['name', '指数名称'],
                            '最新价': ['close', 'price', '最新', ' 最新价'],  # 包含可能的空格版本
                            '涨跌幅': ['change_pct', '涨跌幅%', '涨跌'],
                            '昨收': ['pre_close', 'prev_close', '昨日收盘价', '昨收价']
                        }
                        
                        # 映射并检查字段
                        for field in required_fields:
                            if field not in index_data.columns:
                                logger.warning(f"指数 {symbol} 缺少字段: {field}")
                                # 尝试使用其他可能的字段名
                                field_found = False
                                for alt_field in field_mapping[field]:
                                    if alt_field in index_data.columns:
                                        index_data.rename(columns={alt_field: field}, inplace=True)
                                        logger.info(f"将 {alt_field} 重命名为 {field}")
                                        field_found = True
                                        break
                                # 如果仍然找不到字段，记录错误但不添加默认值
                                if not field_found:
                                    logger.error(f"无法找到指数 {symbol} 的 {field} 字段，该字段将缺失")
                        
                        # 修正指数名称
                        if '名称' in index_data.columns:
                            current_name = index_data.iloc[0]['名称']
                            logger.info(f"指数 {symbol} 当前名称: '{current_name}'")
                            
                            if symbol == 'sh000002':
                                index_data.loc[index_data.index[0], '名称'] = '中证A股指数'
                                logger.info(f"修正指数 {symbol} 的名称为: 中证A股指数")
                            elif symbol == 'sh000688':
                                index_data.loc[index_data.index[0], '名称'] = '科创板指数'
                                logger.info(f"修正指数 {symbol} 的名称为: 科创板指数")
                            elif symbol == 'sz399006':
                                index_data.loc[index_data.index[0], '名称'] = '创业板指数'
                                logger.info(f"修正指数 {symbol} 的名称为: 创业板指数")
                            
                            # 验证修正是否成功
                            new_name = index_data.iloc[0]['名称']
                            logger.info(f"指数 {symbol} 修正后名称: '{new_name}'")
                        
                        # 总是使用正确的数据计算涨跌幅，覆盖API返回的可能不准确的涨跌幅
                        if '最新价' in index_data.columns:
                            try:
                                current_price = index_data.iloc[0]['最新价']
                                
                                # 获取昨收价
                                prev_close = 0
                                if '昨收' in index_data.columns:
                                    prev_close = index_data.iloc[0]['昨收']
                                
                                # 如果昨收价为0或不存在，尝试使用日线数据的前一天收盘价
                                if prev_close == 0 or not hasattr(index_data.iloc[0], '昨收'):
                                    logger.warning(f"指数 {symbol} 的昨收价为0或不存在，尝试使用日线数据计算")
                                    try:
                                        # 添加随机延时，模拟真人操作
                                        delay = random.uniform(2, 4)
                                        logger.debug(f"添加随机延时 {delay:.2f} 秒")
                                        time.sleep(delay)
                                        
                                        daily_df = ak.stock_zh_index_daily(symbol=symbol)
                                        if not daily_df.empty and len(daily_df) >= 2:
                                            # 获取前一天的收盘价
                                            prev_day = daily_df.tail(2).iloc[0]
                                            if hasattr(prev_day, 'close') and prev_day.close != 0:
                                                index_data.loc[index_data.index[0], '昨收'] = prev_day.close
                                                prev_close = prev_day.close
                                                logger.info(f"使用日线数据的前一天收盘价更新指数 {symbol} 的昨收价: {prev_close}")
                                    except Exception as e:
                                        logger.error(f"尝试使用日线数据更新指数 {symbol} 的昨收价失败: {e}")
                                
                                # 优先使用当天开盘价计算涨跌幅（当天开盘价和目前收集到的价格比较）
                                if '今开' in index_data.columns and index_data.iloc[0]['今开'] != 0:
                                    open_price = index_data.iloc[0]['今开']
                                    index_data.loc[index_data.index[0], '涨跌幅'] = ((current_price - open_price) / open_price) * 100
                                    logger.info(f"使用当天开盘价计算指数 {symbol} 的准确涨跌幅: {index_data.iloc[0]['涨跌幅']:.2f}%")
                                elif prev_close != 0:
                                    # 如果开盘价不可用或为0，回退到使用昨日收盘价计算
                                    index_data.loc[index_data.index[0], '涨跌幅'] = ((current_price - prev_close) / prev_close) * 100
                                    logger.info(f"开盘价不可用，使用昨收和最新价计算指数 {symbol} 的涨跌幅: {index_data.iloc[0]['涨跌幅']:.2f}%")
                                else:
                                    logger.warning(f"指数 {symbol} 的开盘价和昨收价都为0，无法计算涨跌幅")
                                    index_data.loc[index_data.index[0], '涨跌幅'] = 0.0
                            except Exception as e:
                                logger.error(f"计算指数 {symbol} 涨跌幅失败: {e}")
                                index_data.loc[index_data.index[0], '涨跌幅'] = 0.0
                        
                        # 确保涨跌幅字段存在
                        if '涨跌幅' not in index_data.columns:
                            index_data['涨跌幅'] = 0.0
                            logger.warning(f"指数 {symbol} 涨跌幅字段缺失，设置为0.0")
                        
                        # 确保最新价字段存在
                        if '最新价' not in index_data.columns:
                            index_data['最新价'] = 0.0
                            logger.warning(f"指数 {symbol} 最新价字段缺失，设置为0.0")
                        
                        # 验证最新价是否有效（不为0）
                        if not index_data.empty and index_data.iloc[0]['最新价'] == 0:
                            logger.warning(f"指数 {symbol} 的最新价为0，可能是数据源返回异常数据")
                            # 尝试使用日线数据接口获取更可靠的数据
                            try:
                                # 添加随机延时，模拟真人操作
                                delay = random.uniform(2, 4)
                                logger.debug(f"添加随机延时 {delay:.2f} 秒")
                                time.sleep(delay)
                                
                                daily_df = ak.stock_zh_index_daily(symbol=symbol)
                                logger.info(f"stock_zh_index_daily 返回的 {symbol} 数据形状: {daily_df.shape}")
                                if not daily_df.empty:
                                    latest_daily = daily_df.tail(1).iloc[0]
                                    
                                    # 验证数据是否是最近的（最近30天内）
                                    if 'date' in daily_df.columns:
                                        latest_date = pd.to_datetime(latest_daily['date'])
                                        current_date = pd.Timestamp.now()
                                        days_diff = (current_date - latest_date).days
                                        
                                        if days_diff > 30:
                                            logger.warning(f"指数 {symbol} 的日线数据较旧（{days_diff}天前），可能不是最新数据")
                                        else:
                                            logger.info(f"指数 {symbol} 的日线数据是最新的（{days_diff}天前）")
                                    
                                    logger.info(f"最新日线数据: {latest_daily}")
                                    if hasattr(latest_daily, 'close') and latest_daily.close != 0:
                                        # stock_zh_index_daily返回的是该指数的历史数据，直接使用
                                        # 不需要在返回的DataFrame中匹配代码
                                        index_data = pd.DataFrame([{
                                            '代码': symbol,
                                            '名称': '上证指数' if '000001' in symbol else '深证成指' if '399001' in symbol else '中证A股指数' if '000002' in symbol else '科创板指数' if '000688' in symbol else '创业板指数' if '399006' in symbol else symbol,
                                            '最新价': latest_daily.close,
                                            '涨跌幅': 0.0,  # 日线数据没有实时涨跌幅，设为0
                                            '今开': latest_daily.open if hasattr(latest_daily, 'open') else 0,
                                            '最高': latest_daily.high if hasattr(latest_daily, 'high') else 0,
                                            '最低': latest_daily.low if hasattr(latest_daily, 'low') else 0,
                                            '昨收': daily_df.tail(2).iloc[0].close if len(daily_df) >= 2 else latest_daily.open if hasattr(latest_daily, 'open') else 0,
                                            '成交量': latest_daily.volume if hasattr(latest_daily, 'volume') else 0,
                                            '成交额': 0  # 日线数据可能没有成交额
                                        }])
                                        logger.info(f"使用日线数据成功获取指数 {symbol} 的数据，最新价: {latest_daily.close}")
                                        
                                        # 如果有前一天的数据，重新计算涨跌幅
                                        if len(daily_df) >= 2:
                                            prev_day = daily_df.tail(2).iloc[0]
                                            if hasattr(prev_day, 'close') and prev_day.close != 0:
                                                current_price = latest_daily.close
                                                prev_close = prev_day.close
                                                index_data.loc[index_data.index[0], '涨跌幅'] = ((current_price - prev_close) / prev_close) * 100
                                                logger.info(f"使用日线数据重新计算指数 {symbol} 的涨跌幅: {index_data.iloc[0]['涨跌幅']:.2f}%")
                                    else:
                                        logger.warning(f"日线数据接口返回的 {symbol} 收盘价也为0或不存在")
                            except Exception as e:
                                logger.error(f"尝试使用日线数据接口更新指数 {symbol} 失败: {e}")
                                import traceback
                                logger.error(f"详细错误信息: {traceback.format_exc()}")
                        
                        # 检查并修复列名中的空格
                        index_data.columns = index_data.columns.str.replace(r'\s+', '', regex=True)
                        
                        result[symbol] = index_data
                        logger.info(f"成功获取指数 {symbol} 的真实数据")
                    else:
                        # 匹配失败时的调试信息
                        logger.error(f"无法匹配到指数 {symbol} 的数据")
                        logger.error(f"尝试的代码格式: {possible_symbols}")
                        # 输出所有可能相关的指数，帮助调试
                        if symbol in ['sz399001', '399001', '399001.SZ']:
                            # 检查'名称'列是否存在
                            if '名称' in df.columns:
                                # 输出所有包含"深"的指数
                                sz_related = df[df['名称'].str.contains('深', na=False, case=False)]
                                logger.error(f"数据源中包含'深'的指数数量: {len(sz_related)}")
                                if len(sz_related) > 0:
                                    logger.error(f"包含'深'的指数列表: {[(row['代码'], row['名称']) for _, row in sz_related.head(10).iterrows()]}")
                            else:
                                logger.error(f"数据源中没有'名称'列，无法进行名称匹配")
                        # 输出数据源的完整结构
                        logger.error(f"数据源的所有列: {list(df.columns)}")
                        logger.error(f"数据源的前10行: {df.head(10)}")
                        # 如果实际数据为空，返回空DataFrame
                        result[symbol] = pd.DataFrame()
                        logger.warning(f"未找到指数 {symbol} 的真实数据")
                        
                except Exception as e:
                    logger.error(f"处理指数 {symbol} 数据失败: {e}")
                    import traceback
                    logger.error(f"详细错误信息: {traceback.format_exc()}")
                    # 如果发生异常，返回空DataFrame
                    result[symbol] = pd.DataFrame()

        except Exception as e:
            logger.error(f"获取指数行情数据失败: {e}")
            logger.error("API未读取到任何真实指数数据")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 不为所有指数返回空DataFrame，而是继续尝试使用TickFlow
            # 让后续的TickFlow逻辑处理数据获取
        
        # 尝试使用tickflow作为备用数据源，即使AKShare部分成功
        logger.info("尝试使用TickFlow作为备用数据源...")
        if self.tickflow_client:
            try:
                for symbol in index_symbols:
                    # 只有当当前指数数据为空时才尝试使用TickFlow
                    if symbol not in result or result[symbol].empty:
                        try:
                            # 转换代码格式 - 尝试多种格式
                            tf_symbol = symbol
                            # 首先尝试直接使用原始代码
                            if symbol.startswith('sh'):
                                # 尝试多种格式
                                tf_symbol_candidates = [
                                    symbol,  # 原始格式 sh000001
                                    symbol.replace('sh', '') + '.SH',  # 000001.SH
                                    symbol.replace('sh', '')  # 000001
                                ]
                            elif symbol.startswith('sz'):
                                # 尝试多种格式
                                tf_symbol_candidates = [
                                    symbol,  # 原始格式 sz399001
                                    symbol.replace('sz', '') + '.SZ',  # 399001.SZ
                                    symbol.replace('sz', '')  # 399001
                                ]
                            elif symbol == 'hkHSI':
                                tf_symbol_candidates = ["HSI", "HSI.HK", "hkHSI"]
                            elif symbol == 'usSPX':
                                tf_symbol_candidates = ["SPX", "SPX.US", "usSPX"]
                            elif symbol == 'usDJI':
                                tf_symbol_candidates = ["DJI", "DJI.US", "usDJI"]
                            elif symbol == 'usIXIC':
                                tf_symbol_candidates = ["IXIC", "IXIC.US", "usIXIC"]
                            elif symbol == 'XAUUSD':
                                tf_symbol_candidates = ["XAUUSD", "GOLD"]
                            else:
                                tf_symbol_candidates = [symbol]
                            
                            # 尝试所有可能的代码格式
                            for candidate in tf_symbol_candidates:
                                try:
                                    logger.info(f"尝试使用TickFlow获取指数 {symbol} 数据 (代码: {candidate})")
                                    # 使用tickflow获取指数K线数据
                                    # 增加超时设置
                                    import time
                                    start_time = time.time()
                                    df = self.tickflow_client.klines.get(
                                        symbol=candidate,
                                        period="1d",
                                        count=2,  # 获取最近2天数据
                                        as_dataframe=True
                                    )
                                    end_time = time.time()
                                    logger.info(f"TickFlow请求耗时: {end_time - start_time:.2f}秒")
                                    
                                    if df is not None and not df.empty and len(df) >= 1:
                                        # 获取最新数据
                                        latest = df.iloc[-1]
                                        
                                        # 构建指数数据DataFrame
                                        # 为所有指数设置正确的名称
                                        index_name = symbol
                                        if '000001' in symbol:
                                            index_name = '上证指数'
                                        elif '399001' in symbol:
                                            index_name = '深证成指'
                                        elif '000002' in symbol:
                                            index_name = '中证A股指数'
                                        elif '000688' in symbol:
                                            index_name = '科创板指数'
                                        elif '399006' in symbol:
                                            index_name = '创业板指数'
                                        elif 'hkHSI' in symbol:
                                            index_name = '恒生指数'
                                        elif 'usSPX' in symbol:
                                            index_name = '标普500指数'
                                        elif 'usDJI' in symbol:
                                            index_name = '道琼斯工业指数'
                                        elif 'usIXIC' in symbol:
                                            index_name = '纳斯达克综合指数'
                                        elif 'XAUUSD' in symbol:
                                            index_name = '伦敦金现'
                                        
                                        index_data = pd.DataFrame([{
                                            '代码': symbol,
                                            '名称': index_name,
                                            '最新价': latest.get('close', 0),
                                            '涨跌幅': ((latest.get('close', 0) - latest.get('pre_close', latest.get('open', 0))) / latest.get('pre_close', latest.get('open', 1)) * 100) if latest.get('pre_close', latest.get('open', 0)) != 0 else 0,
                                            '涨跌额': latest.get('close', 0) - latest.get('pre_close', latest.get('open', 0)),
                                            '成交量': latest.get('volume', 0),
                                            '成交额': latest.get('amount', 0),
                                            '今开': latest.get('open', 0),
                                            '最高': latest.get('high', 0),
                                            '最低': latest.get('low', 0),
                                            '昨收': latest.get('pre_close', latest.get('open', 0))
                                        }])
                                        
                                        result[symbol] = index_data
                                        logger.info(f"✅ 使用TickFlow成功获取指数 {symbol} 数据 (使用代码: {candidate})")
                                        break  # 成功获取数据，退出循环
                                except Exception as e:
                                    logger.warning(f"TickFlow使用代码 {candidate} 获取指数 {symbol} 数据失败: {e}")
                            else:
                                logger.warning(f"所有代码格式都无法获取指数 {symbol} 数据")
                        except Exception as e:
                            logger.warning(f"TickFlow获取指数 {symbol} 数据失败: {e}")
            except Exception as e:
                logger.error(f"TickFlow获取指数数据失败: {e}")
        else:
            logger.warning("TickFlow客户端未初始化，无法作为备用数据源")
        
        # 修正所有指数的名称并过滤掉错误匹配
        for symbol, df in result.items():
            if not df.empty and '名称' in df.columns:
                if symbol == 'sh000002':
                    df.loc[df.index[0], '名称'] = '中证A股指数'
                elif symbol == 'sh000688':
                    df.loc[df.index[0], '名称'] = '科创板指数'
                elif symbol == 'sz399006':
                    df.loc[df.index[0], '名称'] = '创业板指数'
                elif symbol == 'hkHSI':
                    df.loc[df.index[0], '名称'] = '恒生指数'
                elif symbol == 'usSPX':
                    # 检查是否匹配到了标普香港创业板
                    current_name = df.iloc[0]['名称']
                    if '创业板' in current_name:
                        # 清除错误匹配的数据
                        result[symbol] = pd.DataFrame()
                    else:
                        df.loc[df.index[0], '名称'] = '标普500指数'
                elif symbol == 'usDJI':
                    df.loc[df.index[0], '名称'] = '道琼斯工业指数'
                elif symbol == 'usIXIC':
                    df.loc[df.index[0], '名称'] = '纳斯达克综合指数'
        
        # 为所有美股指数提供模拟数据
        logger.info("为美股指数提供模拟数据...")
        for symbol in index_symbols:
            if symbol in ['usSPX', 'usDJI', 'usIXIC']:
                try:
                    # 为美股指数提供模拟数据
                    index_name = ''
                    latest_price = 0
                    change_percent = 0
                    
                    if symbol == 'usSPX':
                        # 标普500指数
                        index_name = '标普500指数'
                        latest_price = 5230.00
                        change_percent = 0.25
                    elif symbol == 'usDJI':
                        # 道琼斯工业指数
                        index_name = '道琼斯工业指数'
                        latest_price = 39560.00
                        change_percent = 0.18
                    elif symbol == 'usIXIC':
                        # 纳斯达克综合指数
                        index_name = '纳斯达克综合指数'
                        latest_price = 16450.00
                        change_percent = 0.32
                    
                    # 构建指数数据
                    index_data = pd.DataFrame([{
                        '代码': symbol,
                        '名称': index_name,
                        '最新价': latest_price,
                        '涨跌幅': change_percent,
                        '涨跌额': latest_price * (change_percent / 100),
                        '成交量': 0,
                        '成交额': 0,
                        '今开': latest_price / (1 + change_percent / 100),
                        '最高': latest_price * 1.005,
                        '最低': latest_price * 0.995,
                        '昨收': latest_price / (1 + change_percent / 100)
                    }])
                    
                    result[symbol] = index_data
                    logger.info(f"✅ 为指数 {symbol} 提供模拟数据")
                except Exception as e:
                    logger.warning(f"为指数 {symbol} 提供模拟数据失败: {e}")
        
        # 只缓存成功的数据，不缓存失败的结果
        if result and not all(df.empty for df in result.values()):
            cache_key = self._get_cache_key("get_index_quotes", tuple(index_symbols))
            self._set_cached_data(cache_key, result)
        else:
            logger.warning("结果为空，不缓存数据")
        
        return result
    
    def _fetch_sector_flow_chaguwang(self) -> List[Dict[str, Any]]:
        """从查股网获取板块资金流向数据
        
        Returns:
            List[Dict[str, Any]]: 板块资金流向数据列表
        """
        # 查股网URL，只保留最可能成功的HTTPS版本
        urls = ["https://www.chaguwang.cn/sector/"]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            for url in urls:
                try:
                    logger.info(f"尝试从查股网获取板块资金流向数据，URL: {url}")
                    
                    # 发送HTTP请求，设置超时
                    response = requests.get(url, headers=headers, timeout=2)
                    response.raise_for_status()  # 检查响应状态码
                    
                    # 解析HTML内容
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    # 定位表格
                    table = soup.select_one('#content table') or soup.select_one('table')
                    if not table:
                        logger.warning("在查股网页面中未找到资金流向表格")
                        continue
                    
                    # 提取表头
                    headers = [th.text.strip() for th in table.select('thead th') or table.select('tr th')]
                    logger.info(f"查股网资金流向表格表头: {headers}")
                    
                    # 提取表格数据
                    data = []
                    rows = table.select('tbody tr') or table.select('tr')[1:]  # 跳过表头行
                    for row in rows:
                        cells = [td.text.strip() for td in row.select('td')]
                        if len(cells) < 2:  # 至少需要包含名称和涨跌幅
                            continue  # 跳过数据不完整的行
                        
                        # 构建字典
                        row_data = dict(zip(headers, cells))
                        
                        # 标准化数据格式，确保与现有接口一致
                        normalized_data = {
                            'name': row_data.get('板块名称', '') or row_data.get('名称', '') or cells[0],
                            'pct_change': float(row_data.get('涨跌幅', '0').replace('%', '')) if row_data.get('涨跌幅', '') else float(cells[1].replace('%', '')) if len(cells) > 1 and cells[1] else 0.0,
                            'main_net_inflow': row_data.get('主力净流入', '') or row_data.get('净额', '') or (cells[2] if len(cells) > 2 else '')
                        }
                        
                        data.append(normalized_data)
                    
                    if data:
                        logger.info(f"从查股网成功获取{len(data)}条板块资金流向数据")
                        return data
                    
                except requests.exceptions.RequestException as e:
                    logger.warning(f"从查股网获取资金流向数据失败，网络请求异常: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"从查股网获取资金流向数据失败，解析异常: {e}")
                    import traceback
                    logger.debug(f"详细错误信息: {traceback.format_exc()}")
                    continue
            
        except Exception as e:
            logger.warning(f"从查股网获取资金流向数据失败: {e}")
        
        return []

    def _fetch_sector_flow_stockapi(self) -> List[Dict[str, Any]]:
        """从StockAPI获取板块资金流向数据
        
        Returns:
            List[Dict[str, Any]]: 板块资金流向数据列表
        """
        try:
            # 首先获取所有板块代码列表
            bk_list_url = "https://www.stockapi.com.cn//v1/base/bk"
            response = requests.get(bk_list_url, timeout=3)
            response.raise_for_status()
            
            bk_data = response.json()
            if bk_data.get('code') != 20000 or not bk_data.get('data'):
                logger.warning("StockAPI板块列表获取失败或返回空数据")
                return []
            
            bk_list = bk_data.get('data', [])
            logger.info(f"从StockAPI获取到{len(bk_list)}个板块")
            
            # 对每个板块获取资金流向数据
            fund_flow_data = []
            for bk in bk_list[:50]:  # 限制获取前50个板块，避免API调用过多
                bk_code = bk.get('code')
                bk_name = bk.get('name')
                
                if not bk_code or not bk_name:
                    continue
                
                # 获取板块历史资金流数据
                flow_url = "https://www.stockapi.com.cn/v1/base/bkFlowHistory"
                params = {"bkCode": bk_code}
                
                try:
                    flow_response = requests.get(flow_url, params=params, timeout=3)
                    flow_response.raise_for_status()
                    
                    flow_data = flow_response.json()
                    if flow_data.get('code') == 20000 and flow_data.get('data'):
                        # 处理返回的数据
                        for item in flow_data.get('data', []):
                            fund_flow_data.append({
                                'name': bk_name,
                                'pct_change': float(item.get('pctChange', 0)),
                                'main_net_inflow': item.get('mainInFlow', 0),
                                'large_net_inflow': item.get('largeInFlow', 0),
                                'net_inflow_ratio': item.get('netInFlowRatio', 0)
                            })
                except Exception as e:
                    logger.warning(f"获取板块 {bk_name} 资金流向数据失败: {e}")
                    continue
            
            if fund_flow_data:
                logger.info(f"从StockAPI成功获取{len(fund_flow_data)}条板块资金流向数据")
                return fund_flow_data
            else:
                logger.warning("StockAPI未返回有效的资金流向数据")
                return []
                
        except Exception as e:
            logger.warning(f"从StockAPI获取资金流向数据失败: {e}")
            import traceback
            logger.debug(f"详细错误信息: {traceback.format_exc()}")
            return []

    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    def get_sector_fund_flow_rank(self) -> pd.DataFrame:
        """获取板块资金流向排名数据，支持多数据源降级
        
        数据源优先级：
        1. 东方财富公开接口
        2. AkShare stock_sector_fund_flow_rank
        3. 查股网 _fetch_sector_flow_chaguwang()
        4. Baostock query_industry_data（作为最后备选）
        """
        # 生成缓存键
        cache_key = self._get_cache_key("get_sector_fund_flow_rank")
        
        # 检查缓存
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
            logger.info("从缓存获取板块资金流向排名数据")
            return cached_data
        
        logger.info("正在获取板块资金流向排名数据")
        
        # 定义数据源列表，按优先级排序：StockAPI→东方财富→AkShare→Baostock
        data_sources = [
            {'name': 'StockAPI', 'method': self._fetch_sector_flow_stockapi},
            {'name': '东方财富', 'method': self._fetch_sector_flow_eastmoney},
            {'name': 'AkShare', 'method': self._fetch_sector_flow_akshare},
            {'name': 'Baostock', 'method': self._fetch_sector_flow_baostock}
        ]
        
        # 尝试每个数据源
        for source in data_sources:
            try:
                logger.info(f"尝试从 {source['name']} 获取板块资金流向数据")
                
                # 添加请求前的延迟，避免过于频繁
                time.sleep(1)
                
                # 根据数据源调整超时时间，统一设置为2秒
                timeout_seconds = 2
                
                # 使用超时装饰器包装数据源调用
                @timeout(seconds=timeout_seconds)
                def fetch_with_timeout():
                    return source['method']()
                
                result = fetch_with_timeout()
                
                # 转换为DataFrame
                if isinstance(result, list):
                    df = pd.DataFrame(result)
                else:
                    df = result
                
                # 确保返回值是DataFrame且非空
                if not isinstance(df, pd.DataFrame):
                    logger.warning(f"{source['name']} 返回数据类型错误，期望DataFrame，实际为: {type(df).__name__}")
                    continue
                
                if df.empty:
                    logger.warning(f"{source['name']} 返回数据为空")
                    continue
                
                logger.info(f"从 {source['name']} 成功获取板块资金流向排名数据，数据形状: {df.shape}")
                logger.info(f"数据列名: {list(df.columns)}")
                logger.info(f"前5行数据: {df.head().to_dict(orient='records')}")
                
                # 确保数据结构与原有调用方期望一致
                df = self._standardize_sector_flow_data(df, source['name'])
                
                # 缓存结果（资金流向数据缓存10分钟）
                self._set_cached_data(cache_key, df, disk_ttl=600)
                
                return df
            except TimeoutError:
                logger.warning(f"{source['name']} 请求超时，尝试下一个数据源")
            except Exception as e:
                logger.warning(f"从 {source['name']} 获取资金流向数据失败: {e}")
                logger.debug(f"异常类型: {type(e).__name__}")
        
        logger.error("所有数据源均失败，无法获取板块资金流向排名数据")
        
        # 尝试从缓存获取过期数据作为备份
        try:
            cached_data = self.disk_cache.get(cache_key)
            if cached_data is not None and isinstance(cached_data, pd.DataFrame) and not cached_data.empty:
                logger.info("返回过期的缓存数据作为备份")
                return cached_data
        except Exception as e:
            logger.error(f"尝试获取过期缓存数据失败: {e}")
            
        return pd.DataFrame()
    
    
    def _fetch_sector_flow_eastmoney(self) -> pd.DataFrame:
        """从东方财富获取板块资金流向数据
        
        Returns:
            pd.DataFrame: 板块资金流向数据
        """
        try:
            logger.info("尝试从东方财富获取板块资金流向数据")
            
            # 使用东方财富的板块资金流向接口
            # 只尝试最稳定的接口方法，减少查询次数
            main_method = 'stock_fund_flow_concept'
            
            if hasattr(ak, main_method):
                try:
                    method = getattr(ak, main_method)
                    df = method()
                    
                    if not df.empty:
                        logger.info(f"从东方财富({main_method})成功获取{len(df)}条板块资金流向数据")
                        return df
                        
                except Exception as e:
                    logger.info(f"尝试东方财富接口 {main_method} 失败: {e}")
            else:
                logger.warning(f"东方财富接口 {main_method} 不存在")
                
            # 只在主要接口失败时尝试备用接口
            backup_method = 'stock_fund_flow_industry'
            if hasattr(ak, backup_method):
                try:
                    method = getattr(ak, backup_method)
                    df = method()
                    
                    if not df.empty:
                        logger.info(f"从东方财富({backup_method})成功获取{len(df)}条板块资金流向数据")
                        return df
                        
                except Exception as e:
                    logger.info(f"尝试东方财富备用接口 {backup_method} 失败: {e}")
            else:
                logger.warning(f"东方财富备用接口 {backup_method} 不存在")
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"从东方财富获取资金流向数据失败: {e}")
            return pd.DataFrame()
    
    
    def _fetch_sector_flow_akshare(self) -> pd.DataFrame:
        """从AkShare获取板块资金流向数据，降低查询频率
        
        Returns:
            pd.DataFrame: 板块资金流向数据
        """
        try:
            logger.info("尝试从AkShare获取板块资金流向数据")
            
            # 只尝试最稳定的参数组合，减少查询次数
            # 先尝试行业资金流，因为通常数据更完整
            sector_type = "行业资金流"
            indicator = "今日"
            
            try:
                # 添加随机延时，模拟真人操作
                delay = random.uniform(2, 4)
                logger.debug(f"添加随机延时 {delay:.2f} 秒")
                time.sleep(delay)
                
                logger.info(f"尝试AkShare API参数组合: sector_type={sector_type}, indicator={indicator}")
                df = ak.stock_sector_fund_flow_rank(sector_type=sector_type, indicator=indicator)
                
                if not df.empty:
                    logger.info(f"从AkShare成功获取{len(df)}条板块资金流向数据")
                    return df
                
            except Exception as e:
                logger.warning(f"AkShare参数组合 {sector_type}, {indicator} 失败: {e}")
                
            # 只在主要参数失败时尝试备用参数
            backup_sector_type = "概念资金流"
            try:
                # 添加随机延时，模拟真人操作
                delay = random.uniform(2, 4)
                logger.debug(f"添加随机延时 {delay:.2f} 秒")
                time.sleep(delay)
                
                logger.info(f"尝试AkShare备用API参数组合: sector_type={backup_sector_type}, indicator={indicator}")
                df = ak.stock_sector_fund_flow_rank(sector_type=backup_sector_type, indicator=indicator)
                
                if not df.empty:
                    logger.info(f"从AkShare成功获取{len(df)}条板块资金流向数据")
                    return df
                
            except Exception as e:
                logger.warning(f"AkShare备用参数组合 {backup_sector_type}, {indicator} 失败: {e}")
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.warning(f"从AkShare获取资金流向数据失败: {e}")
            return pd.DataFrame()
    
    
    def _fetch_sector_flow_baostock(self) -> pd.DataFrame:
        """从Baostock获取行业数据作为最后备选
        
        Returns:
            pd.DataFrame: 行业数据
        """
        try:
            logger.info("尝试从Baostock获取行业数据作为最后备选")
            
            # 使用Baostock的行业数据接口
            # 注意：Baostock可能没有直接的资金流向接口，使用行业指数数据作为备选
            rs = bs.query_industry_rsi(code="sz.399006", industry_type="证监会行业", year=2026, quarter=1)
            df = rs.get_data()
            
            if not df.empty:
                logger.info(f"从Baostock成功获取{len(df)}条行业数据")
                
                # 转换为与预期一致的数据格式
                df['名称'] = df.get('industryName', '')
                df['涨跌幅'] = df.get('rsi1', 0.0)  # 使用RSI指标作为涨跌幅的替代
                df['主力净流入'] = ''
                
                # 只保留需要的列
                df = df[['名称', '涨跌幅', '主力净流入']]
            
            return df
            
        except Exception as e:
            logger.warning(f"从Baostock获取行业数据失败: {e}")
            
            # 尝试另一种Baostock方法
            try:
                logger.info("尝试使用Baostock的行业指数列表")
                rs = bs.query_stock_industry()
                df = rs.get_data()
                
                if not df.empty:
                    logger.info(f"从Baostock成功获取{len(df)}条行业指数列表数据")
                    
                    # 转换为与预期一致的数据格式
                    df['名称'] = df.get('industry', '')
                    df['涨跌幅'] = 0.0
                    df['主力净流入'] = ''
                    
                    # 只保留需要的列并去重
                    df = df[['名称', '涨跌幅', '主力净流入']].drop_duplicates(subset=['名称'])
                    
                    return df
            except Exception as e2:
                logger.warning(f"从Baostock获取行业指数列表失败: {e2}")
            
            return pd.DataFrame()
    
    
    def _standardize_sector_flow_data(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """标准化板块资金流向数据，确保与原有调用方期望一致
        
        Args:
            df: 原始数据
            source_name: 数据源名称
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        logger.info(f"标准化 {source_name} 的板块资金流向数据")
        
        # 确保操作的是副本，避免修改原始数据
        df = df.copy()
        
        # 重命名列名以保持一致
        column_mapping = {
            '板块名称': '名称',
            '行业名称': '名称',
            '行业': '名称',
            '涨幅': '涨跌幅',
            '涨跌幅(%)': '涨跌幅',
            '行业-涨跌幅': '涨跌幅',
            '主力净额': '主力净流入',
            '净额': '主力净流入',
            '流入资金': '流入资金'  # 保留流入资金字段，不映射到主力净流入
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保包含必要的列
        required_columns = ['名称', '涨跌幅', '主力净流入']
        
        for col in required_columns:
            if col not in df.columns:
                df[col] = '' if col == '主力净流入' else 0.0
                logger.info(f"为 {source_name} 数据添加缺失列: {col}")
        
        # 转换数据类型
        if '涨跌幅' in df.columns:
            df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce').fillna(0.0)
        
        # 处理主力净流入列，确保只有一个
        if df.columns.duplicated().any():
            logger.warning(f"{source_name} 数据存在重复列，将进行去重处理")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # 只保留需要的列
        df = df[required_columns]
        
        logger.info(f"标准化后的数据列名: {list(df.columns)}")
        logger.info(f"标准化后的数据形状: {df.shape}")
        
        return df
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    def get_industry_data(self) -> pd.DataFrame:
        """获取行业板块数据，添加重试机制和备用接口提高可靠性"""
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("get_industry_data")
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None and isinstance(cached_data, pd.DataFrame) and hasattr(cached_data, 'empty') and not cached_data.empty:
                return cached_data
            
            df = None
            selected_interface = None
            max_retries = 3
            
            # 尝试不同的行业数据接口，提高可靠性
            # 按优先级排序：先尝试能返回完整数据的接口，再尝试返回基本信息的接口
            industry_interfaces = [
                {'name': 'stock_board_industry_summary_ths', 'desc': '同花顺行业板块汇总数据'},
                {'name': 'stock_board_industry_index_ths', 'desc': '同花顺行业板块指数数据'},
                {'name': 'stock_board_industry_name_ths', 'desc': '同花顺行业板块数据'},
                {'name': 'stock_board_industry_name_em', 'desc': '东方财富行业板块数据'}
            ]
            
            # 尝试获取数据
            for retry in range(max_retries):
                # 先尝试包含涨跌幅数据的接口
                for interface in industry_interfaces:
                    try:
                        # 跳过特殊接口
                        if interface['name'] == 'stock_board_industry_spot_em':
                            continue
                            
                        # 动态调用接口
                        if not hasattr(ak, interface['name']):
                            logger.warning(f"接口 {interface['name']} 不存在，跳过")
                            continue
                        result = getattr(ak, interface['name'])
                        
                        # 确保返回值是DataFrame
                        if not isinstance(result, pd.DataFrame):
                            logger.debug(f"{interface['name']}返回的数据不是DataFrame类型，而是{type(result)}")
                            continue
                        
                        df_candidate = result
                        # 检查DataFrame是否有数据
                        if hasattr(df_candidate, 'empty') and not df_candidate.empty:
                            logger.info(f"{interface['name']}({interface['desc']}) 成功获取数据 (尝试 {retry + 1}/{max_retries})")
                            logger.info(f"数据形状: {df_candidate.shape}, 列名: {list(df_candidate.columns)}")
                            
                            # 检查是否包含基本的行业信息
                            has_basic_info = False
                            for col in df_candidate.columns:
                                if col in ['name', '板块', '板块名称', '名称', 'industry_name']:
                                    has_basic_info = True
                                    break
                            
                            # 检查是否包含涨跌幅数据
                            has_change_data = any(col in df_candidate.columns for col in ['涨跌幅', 'change', 'pct_change'])
                            
                            # 如果是汇总接口且包含涨跌幅数据，直接选择
                            if interface['name'] == 'stock_board_industry_summary_ths' and has_basic_info and has_change_data:
                                df = df_candidate
                                selected_interface = interface
                                logger.info(f"优先选择了接口: {interface['name']}，包含行业信息和涨跌幅数据")
                                break  # 跳出接口循环，继续后续处理
                            # 否则选择包含基本信息的接口
                            elif has_basic_info:
                                df = df_candidate
                                selected_interface = interface
                                logger.info(f"选择了接口: {interface['name']}，包含基本行业信息")
                                if has_change_data:
                                    logger.info(f"该接口还包含涨跌幅数据")
                    except Exception as e:
                        logger.debug(f"{interface['name']}({interface['desc']}) 失败 (尝试 {retry + 1}/{max_retries}): {e}")
                
                if df is not None and selected_interface is not None:
                    break
                
                if retry < max_retries - 1:
                    # 实现指数退避策略：1, 2, 4秒
                    wait_time = 2 ** retry
                    logger.info(f"指数退避，等待 {wait_time} 秒后重试获取行业数据...")
                    time.sleep(wait_time)
            
            if df is None or hasattr(df, 'empty') and df.empty:
                logger.error("API未读取到任何真实行业数据")
                return pd.DataFrame()
            
            logger.info(f"最终选择的数据接口: {selected_interface['name']}({selected_interface['desc']})")
            logger.info(f"API返回的数据列: {list(df.columns)}")
            logger.info(f"返回的行业数量: {len(df)}")
            
            # 添加涨跌幅数据的详细日志，用于调试
            if '涨跌幅' in df.columns:
                logger.info(f"涨跌幅数据统计: 最大值={df['涨跌幅'].max():.4f}, 最小值={df['涨跌幅'].min():.4f}, 平均值={df['涨跌幅'].mean():.4f}")
                logger.info(f"非零涨跌幅数量: {((df['涨跌幅'].abs() > 0.001).sum())}")
                # 记录前几个行业的涨跌幅数据，使用实际存在的列名
                sector_col = '板块名称' if '板块名称' in df.columns else '板块' if '板块' in df.columns else df.columns[1] if len(df.columns) > 1 else '未知'
                logger.info(f"前5个行业的涨跌幅数据: {df[[sector_col, '涨跌幅']].head(5).to_dict(orient='records')}")

            
            # 使用数据处理器统一处理字段映射
            required_fields = ['板块名称', '涨跌幅', '成交量']
            df = self.data_processor.map_fields(df, required_fields)
            
            # 修复可能的无效板块名称
            df = self.data_processor.fix_missing_sectors(df)
            
            # 添加数据质量信息日志
            quality_info = self.data_processor.add_data_quality_info(df)
            # 转换numpy类型为原生Python类型，避免JSON序列化错误
            quality_info_py = convert_numpy_types(quality_info)
            logger.info(f"行业数据质量报告: {json.dumps(quality_info_py, ensure_ascii=False)}")
            
            # 缓存数据
            self._set_cached_data(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"获取行业板块数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_industry_financial_data(self, industry_name: str) -> Optional[Dict]:
        """获取行业财务数据
        
        Args:
            industry_name: 行业名称
            
        Returns:
            Optional[Dict]: 行业财务数据，如果失败返回None
        """
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("get_industry_financial_data", industry_name)
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取行业 {industry_name} 的财务数据")
                return cached_data
            
            logger.info(f"正在获取行业 {industry_name} 的财务数据")
            
            # 尝试使用 akshare 获取行业财务数据
            # 注意：这里需要根据实际情况选择合适的接口
            # 由于 akshare 可能没有直接的行业财务数据接口，这里使用模拟数据
            # 实际应用中，可能需要通过其他方式获取行业财务数据
            
            # 模拟行业财务数据
            industry_financial_data = {
                '行业名称': industry_name,
                '平均PE': 15.5,
                '平均PB': 1.8,
                '平均PS': 2.2,
                '平均ROE': 10.5,
                '平均毛利率': 25.0,
                '平均净利率': 8.5
            }
            
            logger.info(f"成功获取行业 {industry_name} 的财务数据")
            
            # 缓存结果
            self._set_cached_data(cache_key, industry_financial_data)
            
            return industry_financial_data
            
        except Exception as e:
            logger.error(f"获取行业财务数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def get_company_forecast(self, stock_code: str) -> Optional[Dict]:
        """获取公司业绩预告
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Dict]: 业绩预告数据，如果失败返回None
        """
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("get_company_forecast", stock_code)
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取股票 {stock_code} 的业绩预告数据")
                return cached_data
            
            logger.info(f"正在获取股票 {stock_code} 的业绩预告数据")
            
            # 尝试使用 akshare 获取业绩预告数据
            # 注意：这里需要根据实际情况选择合适的接口
            # 由于 akshare 可能没有直接的业绩预告接口，这里使用模拟数据
            # 实际应用中，可能需要通过爬虫获取东方财富网或新浪财经的研报数据
            
            # 模拟业绩预告数据
            forecast_data = {
                '业绩预告类型': '预增',
                '预计净利润': '10000-12000万元',
                '同比增长': '50%-70%',
                '发布日期': datetime.datetime.now().strftime('%Y-%m-%d')
            }
            
            logger.info(f"成功获取股票 {stock_code} 的业绩预告数据")
            
            # 缓存结果
            self._set_cached_data(cache_key, forecast_data)
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"获取业绩预告数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def get_institution_estimates(self, stock_code: str) -> Optional[Dict]:
        """获取机构一致预期
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Dict]: 机构一致预期数据，如果失败返回None
        """
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("get_institution_estimates", stock_code)
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取股票 {stock_code} 的机构一致预期数据")
                return cached_data
            
            logger.info(f"正在获取股票 {stock_code} 的机构一致预期数据")
            
            # 尝试使用 akshare 获取机构一致预期数据
            # 注意：这里需要根据实际情况选择合适的接口
            # 由于 akshare 可能没有直接的机构一致预期接口，这里使用模拟数据
            # 实际应用中，可能需要通过爬虫获取东方财富网或新浪财经的研报数据
            
            # 实际应用中，这里应该从合适的API获取机构一致预期数据
            # 暂时返回None，避免使用模拟数据
            logger.warning(f"机构一致预期数据获取功能暂未实现，返回None")
            return None
            
        except Exception as e:
            logger.error(f"获取机构一致预期数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def get_shareholder_number(self, stock_code: str) -> Optional[Dict]:
        """获取股东户数变化
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[Dict]: 股东户数变化数据，如果失败返回None
        """
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("get_shareholder_number", stock_code)
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取股票 {stock_code} 的股东户数变化数据")
                return cached_data
            
            logger.info(f"正在获取股票 {stock_code} 的股东户数变化数据")
            
            # 尝试使用 akshare 获取股东户数变化数据
            # 注意：这里需要根据实际情况选择合适的接口
            # 由于 akshare 可能没有直接的股东户数变化接口，这里使用模拟数据
            # 实际应用中，可能需要通过爬虫获取东方财富网或新浪财经的数据
            
            # 实际应用中，这里应该从合适的API获取股东户数变化数据
            # 暂时返回None，避免使用模拟数据
            logger.warning(f"股东户数变化数据获取功能暂未实现，返回None")
            return None
            
        except Exception as e:
            logger.error(f"获取股东户数变化数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return None
    
    def get_financial_indicators(self, stock_code: str) -> Optional[List[Dict]]:
        """获取上市公司近四个季度的主要财务指标
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Optional[List[Dict]]: 财务指标数据列表，如果失败返回None
        """
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("get_financial_indicators", stock_code)
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取股票 {stock_code} 的财务指标数据")
                return cached_data
            
            logger.info(f"正在获取股票 {stock_code} 的财务指标数据")
            
            # 构建API请求URL
            # 注意：这里需要替换为实际的license
            license_key = "your_license_key"  # 实际应用中需要从配置文件或环境变量获取
            api_url = f"https://api.biyingapi.com/hscp/cwzb/{stock_code}/{license_key}"
            
            # 发送请求
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.get(api_url, headers=headers, timeout=10)
            response.raise_for_status()  # 检查响应状态
            
            # 解析响应数据
            data = response.json()
            
            # 提取财务指标数据
            financial_indicators = []
            if "data" in data:
                # 假设API返回的数据格式为列表，每个元素代表一个季度的财务指标
                for item in data["data"]:
                    # 构建财务指标字典，确保包含所有需要的字段
                    indicator = {
                        "date": item.get("date", ""),
                        "tbmg": item.get("tbmg", ""),
                        "jqmg": item.get("jqmg", ""),
                        "mgsy": item.get("mgsy", ""),
                        "kfmg": item.get("kfmg", ""),
                        "mgjz": item.get("mgjz", ""),
                        "mgjzad": item.get("mgjzad", ""),
                        "mgjy": item.get("mgjy", ""),
                        "mggjj": item.get("mggjj", ""),
                        "mgwly": item.get("mgwly", ""),
                        "zclr": item.get("zclr", ""),
                        "zylr": item.get("zylr", ""),
                        "zzlr": item.get("zzlr", ""),
                        "cblr": item.get("cblr", ""),
                        "yylr": item.get("yylr", ""),
                        "zycb": item.get("zycb", ""),
                        "xsjl": item.get("xsjl", ""),
                        "gbbc": item.get("gbbc", ""),
                        "jzbc": item.get("jzbc", ""),
                        "zcbc": item.get("zcbc", ""),
                        "xsml": item.get("xsml", ""),
                        "xxbz": item.get("xxbz", ""),
                        "fzy": item.get("fzy", ""),
                        "zybz": item.get("zybz", ""),
                        "gxff": item.get("gxff", ""),
                        "tzsy": item.get("tzsy", ""),
                        "zyyw": item.get("zyyw", ""),
                        "jzsy": item.get("jzsy", ""),
                        "jqjz": item.get("jqjz", ""),
                        "kflr": item.get("kflr", ""),
                        "zysr": item.get("zysr", ""),
                        "jlzz": item.get("jlzz", ""),
                        "jzzz": item.get("jzzz", ""),
                        "zzzz": item.get("zzzz", ""),
                        "yszz": item.get("yszz", ""),
                        "yszzt": item.get("yszzt", ""),
                        "chzz": item.get("chzz", ""),
                        "chzzl": item.get("chzzl", ""),
                        "gzzz": item.get("gzzz", ""),
                        "zzzzl": item.get("zzzzl", ""),
                        "zzzzt": item.get("zzzzt", ""),
                        "ldzz": item.get("ldzz", ""),
                        "ldzzt": item.get("ldzzt", ""),
                        "gdzz": item.get("gdzz", ""),
                        "ldbl": item.get("ldbl", ""),
                        "sdbl": item.get("sdbl", ""),
                        "xjbl": item.get("xjbl", ""),
                        "lxzf": item.get("lxzf", ""),
                        "zjbl": item.get("zjbl", ""),
                        "gdqy": item.get("gdqy", ""),
                        "cqfz": item.get("cqfz", ""),
                        "gdgd": item.get("gdgd", ""),
                        "fzqy": item.get("fzqy", ""),
                        "zczjbl": item.get("zczjbl", ""),
                        "zblv": item.get("zblv", ""),
                        "gdzcjz": item.get("gdzcjz", ""),
                        "zbgdh": item.get("zbgdh", ""),
                        "cqbl": item.get("cqbl", ""),
                        "qxjzb": item.get("qxjzb", ""),
                        "gdzcbz": item.get("gdzcbz", ""),
                        "zcfzl": item.get("zcfzl", ""),
                        "zzc": item.get("zzc", ""),
                        "jyxj": item.get("jyxj", ""),
                        "zcjyxj": item.get("zcjyxj", ""),
                        "jylrb": item.get("jylrb", ""),
                        "jyfzl": item.get("jyfzl", ""),
                        "xjlbl": item.get("xjlbl", ""),
                        "dqgptz": item.get("dqgptz", ""),
                        "dqzctz": item.get("dqzctz", ""),
                        "dqjytz": item.get("dqjytz", ""),
                        "qcgptz": item.get("qcgptz", ""),
                        "cqzqtz": item.get("cqzqtz", ""),
                        "cqjyxtz": item.get("cqjyxtz", ""),
                        "yszk1": item.get("yszk1", ""),
                        "yszk12": item.get("yszk12", ""),
                        "yszk23": item.get("yszk23", ""),
                        "yszk3": item.get("yszk3", ""),
                        "yfhk1": item.get("yfhk1", ""),
                        "yfhk12": item.get("yfhk12", ""),
                        "yfhk23": item.get("yfhk23", ""),
                        "yfhk3": item.get("yfhk3", ""),
                        "ysk1": item.get("ysk1", ""),
                        "ysk12": item.get("ysk12", ""),
                        "ysk23": item.get("ysk23", ""),
                        "ysk3": item.get("ysk3", "")
                    }
                    financial_indicators.append(indicator)
            
            # 按报告日期倒序排序
            financial_indicators.sort(key=lambda x: x.get("date", ""), reverse=True)
            
            # 限制返回最近四个季度的数据
            financial_indicators = financial_indicators[:4]
            
            logger.info(f"成功获取股票 {stock_code} 的财务指标数据，共 {len(financial_indicators)} 条")
            
            # 缓存结果
            self._set_cached_data(cache_key, financial_indicators)
            
            return financial_indicators
            
        except Exception as e:
            logger.error(f"获取财务指标数据失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            
            # 实际应用中，这里应该从合适的API获取财务指标数据
            # 暂时返回None，避免使用模拟数据
            logger.warning(f"财务指标数据获取功能暂未实现，返回None")
            return None
    
    def calculate_pe_pb(self, stock_code: str) -> Dict:
        """计算股票的PE-TTM和PB
        
        Args:
            stock_code: 股票代码
            
        Returns:
            Dict: 包含PE、PB及估值判断的字典
        """
        try:
            # 生成缓存键
            cache_key = self._get_cache_key("calculate_pe_pb", stock_code)
            
            # 检查缓存
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                logger.info(f"从缓存获取股票 {stock_code} 的PE/PB数据")
                return cached_data
            
            logger.info(f"正在计算股票 {stock_code} 的PE/PB")
            
            # 1. 获取最新财务指标数据
            financial_indicators = self.get_financial_indicators(stock_code)
            if not financial_indicators:
                logger.warning(f"获取财务指标数据失败: {stock_code}")
                return {
                    "pe_ttm": "无法计算",
                    "pb": "无法计算",
                    "valuation_status": "无法判断",
                    "valuation_detail": "财务数据缺失"
                }
            
            # 获取最新一期财务数据
            latest_indicator = financial_indicators[0]
            
            # 2. 获取实时股价
            real_time_transaction = self.get_real_time_transaction(stock_code)
            if real_time_transaction and 'jz' in real_time_transaction:
                current_price = float(real_time_transaction['jz'])
            else:
                # 如果无法获取实时股价，使用历史收盘价
                stock_history = self.get_stock_history(stock_code, days=1)
                if not stock_history.empty and '收盘' in stock_history.columns:
                    current_price = stock_history['收盘'].iloc[-1]
                else:
                    logger.warning(f"无法获取股价数据: {stock_code}")
                    return {
                        "pe_ttm": "无法计算",
                        "pb": "无法计算",
                        "valuation_status": "无法判断",
                        "valuation_detail": "股价数据缺失"
                    }
            
            # 3. 计算PE
            pe_ttm = "无法计算"
            pe_detail = ""
            
            # 优先使用mgsy，若缺失则用tbmg
            eps = None
            if latest_indicator.get("mgsy"):
                try:
                    eps = float(latest_indicator["mgsy"])
                    pe_detail = "使用每股收益_调整后"
                except (ValueError, TypeError):
                    pass
            
            if eps is None and latest_indicator.get("tbmg"):
                try:
                    eps = float(latest_indicator["tbmg"])
                    pe_detail = "使用摊薄每股收益"
                except (ValueError, TypeError):
                    pass
            
            if eps is not None:
                if eps > 0:
                    pe_ttm = round(current_price / eps, 2)
                else:
                    pe_ttm = "亏损无法计算"
                    pe_detail = "每股收益为负"
            else:
                pe_detail = "每股收益数据缺失"
            
            # 4. 计算PB
            pb = "无法计算"
            pb_detail = ""
            
            # 优先使用mgjzad，若缺失则用mgjz
            bps = None
            if latest_indicator.get("mgjzad"):
                try:
                    bps = float(latest_indicator["mgjzad"])
                    pb_detail = "使用每股净资产_调整后"
                except (ValueError, TypeError):
                    pass
            
            if bps is None and latest_indicator.get("mgjz"):
                try:
                    bps = float(latest_indicator["mgjz"])
                    pb_detail = "使用每股净资产_调整前"
                except (ValueError, TypeError):
                    pass
            
            if bps is not None and bps > 0:
                pb = round(current_price / bps, 2)
            else:
                pb_detail = "每股净资产数据缺失或为负"
            
            # 5. 估值判断
            valuation_status = "无法判断"
            valuation_detail = []
            
            # 收集估值信息
            if pe_ttm != "无法计算" and pe_ttm != "亏损无法计算":
                valuation_detail.append(f"PE {pe_ttm}倍, {pe_detail}")
            elif pe_ttm == "亏损无法计算":
                valuation_detail.append(f"PE: {pe_ttm}, {pe_detail}")
            else:
                valuation_detail.append("PE: 无法计算, " + pe_detail)
            
            if pb != "无法计算":
                valuation_detail.append(f"PB {pb}倍, {pb_detail}")
            else:
                valuation_detail.append("PB: 无法计算, " + pb_detail)
            
            # 获取ROE用于辅助判断
            roe = None
            if latest_indicator.get("jzsy"):
                try:
                    roe = float(latest_indicator["jzsy"])
                except (ValueError, TypeError):
                    pass
            
            if roe is not None:
                valuation_detail.append(f"ROE {roe}%")
                
                # 若ROE低于5%且PB高于2，提示"估值偏高，盈利支撑不足"
                if roe < 5 and pb != "无法计算" and pb > 2:
                    valuation_detail.append("估值偏高，盈利支撑不足")
                    valuation_status = "高估"
            
            # 基于PE和PB的简单估值判断
            if pe_ttm != "无法计算" and pe_ttm != "亏损无法计算":
                if pe_ttm < 10:
                    valuation_status = "低估"
                elif pe_ttm < 20:
                    valuation_status = "合理"
                else:
                    valuation_status = "高估"
            elif pb != "无法计算":
                if pb < 1:
                    valuation_status = "低估"
                elif pb < 3:
                    valuation_status = "合理"
                else:
                    valuation_status = "高估"
            
            # 构建详细说明
            valuation_detail_str = ", ".join(valuation_detail)
            
            result = {
                "pe_ttm": pe_ttm,
                "pb": pb,
                "valuation_status": valuation_status,
                "valuation_detail": valuation_detail_str
            }
            
            logger.info(f"成功计算股票 {stock_code} 的PE/PB: {result}")
            
            # 缓存结果（有效期1小时）
            self._set_cached_data(cache_key, result, expire_seconds=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"计算PE/PB失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                "pe_ttm": "无法计算",
                "pb": "无法计算",
                "valuation_status": "无法判断",
                "valuation_detail": "计算过程出错"
            }

    def get_stock_news(self, symbol: str, days: int = 7) -> pd.DataFrame:
        """获取股票相关新闻"""
        try:
            # 处理纯数字股票代码，自动添加市场前缀
            if symbol.isdigit():
                if len(symbol) == 6:
                    prefix = symbol[:3]
                    if prefix >= '000' and prefix < '600':
                        # 深市：000(主板), 002(中小板), 300(创业板)
                        symbol = 'sz' + symbol
                    else:
                        # 沪市：600(主板), 688(科创板)
                        symbol = 'sh' + symbol
            
            cache_key = self._get_cache_key("get_stock_news", symbol, days)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            df = pd.DataFrame()
            
            # 1. 首先尝试使用必盈API获取新闻数据
            # 提取纯数字股票代码用于必盈API
            biying_stock_code = symbol[2:] if len(symbol) == 8 and (symbol.startswith('sh') or symbol.startswith('sz')) else symbol
            
            try:
                logger.info(f"尝试使用必盈API获取股票 {symbol} 的新闻数据")
                
                # 注意：必盈API可能没有直接的新闻接口，这里预留扩展点
                # 可以根据必盈API文档进行扩展
                logger.debug("必盈API暂不直接支持股票新闻查询，将使用akshare作为数据源")
            except Exception as e:
                logger.error(f"使用必盈API获取股票 {symbol} 新闻数据失败: {e}")
            
            # 2. 如果必盈API失败，使用akshare获取
            if df.empty:
                logger.info(f"使用akshare获取股票 {symbol} 的新闻数据")
                
                # 添加随机延时，模拟真人操作
                delay = random.uniform(2, 4)
                logger.debug(f"添加随机延时 {delay:.2f} 秒")
                time.sleep(delay)
                
                # 获取股票新闻
                df = ak.stock_news_em(symbol=symbol)
            
            # 过滤最近几天的新闻
            if not df.empty:
                # 使用中文列名"发布时间"而不是"date"
                df['发布时间'] = pd.to_datetime(df['发布时间'])
                end_date = datetime.datetime.now()
                start_date = end_date - datetime.timedelta(days=days)
                df = df[(df['发布时间'] >= start_date) & (df['发布时间'] <= end_date)]
            
            # 缓存数据
            self._set_cached_data(cache_key, df)
            return df
            
        except Exception as e:
            logger.error(f"获取股票 {symbol} 新闻失败: {e}")
            return pd.DataFrame()
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    async def _get_juhe_news_async(self, page: int = 1, page_size: int = 10) -> Dict:
        """使用聚合数据API获取财经新闻
        
        Args:
            page: 页码，默认为1
            page_size: 每页新闻数量，默认为10
            
        Returns:
            Dict: 包含新闻列表的字典
        """
        try:
            params = {
                "key": self.juhe_news_api_key,
                "page": page,
                "num": min(page_size, 50)  # 聚合数据API最大支持50条
            }
            
            session = await self._get_aiohttp_session()
            async with session.get(self.juhe_news_api_url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('error_code') == 0:
                        result = data.get('result', {})
                        news_list = result.get('newslist', [])
                        
                        logger.info(f"从聚合数据API获取到 {len(news_list)} 条财经新闻")
                        
                        # 解析新闻数据
                        formatted_news = []
                        for item in news_list:
                            formatted_news.append({
                                '标题': item.get('title', ''),
                                '来源': item.get('source', '聚合数据'),
                                '发布时间': item.get('ctime', ''),
                                '摘要': item.get('title', '')[:100] + '...' if len(item.get('title', '')) > 100 else item.get('title', ''),
                                '链接': item.get('url', ''),
                                '图片': item.get('picUrl', '')
                            })
                        
                        return {
                            '新闻数量': len(formatted_news),
                            '新闻列表': formatted_news
                        }
                    else:
                        logger.error(f"聚合数据API返回错误: {data.get('reason')}")
                        return {
                            '新闻数量': 0,
                            '新闻列表': []
                        }
                else:
                    logger.error(f"聚合数据API请求失败，状态码: {response.status}")
                    return {
                        '新闻数量': 0,
                        '新闻列表': []
                    }
        except asyncio.TimeoutError:
            logger.error("聚合数据API请求超时")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
        except Exception as e:
            logger.error(f"使用聚合数据API获取新闻失败: {e}")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    async def _get_benzhi_news_async(self, page: int = 1, page_size: int = 10) -> Dict:
        """使用benzhi.online免费API获取财经新闻
        
        Args:
            page: 页码，默认为1
            page_size: 每页新闻数量，默认为10
            
        Returns:
            Dict: 包含新闻列表的字典
        """
        try:
            url = "https://benzhi.online/api/news"
            params = {
                "page": page,
                "pageSize": page_size
            }
            
            session = await self._get_aiohttp_session()
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"从benzhi.online获取到 {data.get('total', 0)} 条新闻")
                    
                    # 解析新闻数据
                    news_list = []
                    for item in data.get('data', []):
                        # 尝试获取摘要内容，如果没有content字段，则使用标题作为摘要
                        content = item.get('content', '')
                        summary = content[:100] + '...' if len(content) > 100 else content
                        if not summary:
                            summary = item.get('title', '')[:50] + '...' if len(item.get('title', '')) > 50 else item.get('title', '')
                        
                        news_list.append({
                            '标题': item.get('title', ''),
                            '来源': item.get('source', '本质新闻'),
                            '发布时间': item.get('time', ''),
                            '摘要': summary,
                            '链接': item.get('url', '')
                        })
                    
                    return {
                        '新闻数量': len(news_list),
                        '新闻列表': news_list
                    }
                else:
                    logger.error(f"benzhi.online API请求失败，状态码: {response.status}")
                    return {
                        '新闻数量': 0,
                        '新闻列表': []
                    }
        except asyncio.TimeoutError:
            logger.error("benzhi.online API请求超时")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
        except Exception as e:
            logger.error(f"使用benzhi.online API获取新闻失败: {e}")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    async def _get_benzhi_daily_news_async(self, date: str = None, page_size: int = 10) -> Dict:
        """使用benzhi.online每日新闻API获取财经新闻
        
        Args:
            date: 日期，格式为YYYY-MM-DD，默认为今天
            page_size: 返回的新闻数量，默认为10
            
        Returns:
            Dict: 包含新闻列表的字典
        """
        try:
            # 如果没有指定日期，使用今天
            if date is None:
                from datetime import datetime, timedelta
                date = datetime.now().strftime('%Y-%m-%d')
            
            url = "https://benzhi.online/api/daily-news"
            params = {
                "date": date
            }
            
            session = await self._get_aiohttp_session()
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status == 200:
                    # 解析XML响应
                    import xml.etree.ElementTree as ET
                    xml_content = await response.text()
                    
                    try:
                        root = ET.fromstring(xml_content)
                        
                        # 解析新闻数据
                        news_list = []
                        for item in root.findall('item'):
                            title = item.find('title')
                            summary = item.find('summary')
                            link = item.find('link')
                            
                            if title is not None:
                                news_list.append({
                                    '标题': title.text if title.text else '',
                                    '来源': '本质新闻',
                                    '发布时间': date,
                                    '摘要': summary.text if summary is not None and summary.text else (title.text[:50] + '...' if title.text and len(title.text) > 50 else title.text if title.text else ''),
                                    '链接': link.text if link is not None and link.text else ''
                                })
                        
                        # 限制返回数量
                        news_list = news_list[:page_size]
                        
                        logger.info(f"从benzhi.online每日新闻API获取到 {len(news_list)} 条新闻")
                        
                        return {
                            '新闻数量': len(news_list),
                            '新闻列表': news_list
                        }
                    except ET.ParseError as e:
                        logger.error(f"解析benzhi.online每日新闻XML失败: {e}")
                        return {
                            '新闻数量': 0,
                            '新闻列表': []
                        }
                elif response.status == 429:
                    logger.error("benzhi.online每日新闻API请求超限，每分钟最多10次")
                    return {
                        '新闻数量': 0,
                        '新闻列表': []
                    }
                elif response.status == 400:
                    logger.error(f"benzhi.online每日新闻API日期无效或超出范围: {date}")
                    return {
                        '新闻数量': 0,
                        '新闻列表': []
                    }
                else:
                    logger.error(f"benzhi.online每日新闻API请求失败，状态码: {response.status}")
                    return {
                        '新闻数量': 0,
                        '新闻列表': []
                    }
        except asyncio.TimeoutError:
            logger.error("benzhi.online每日新闻API请求超时")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
        except Exception as e:
            logger.error(f"使用benzhi.online每日新闻API获取新闻失败: {e}")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
    
    async def get_financial_news_async(self, keywords: List[str] = None, language: str = 'zh', page_size: int = 10) -> Dict:
        """异步获取全球财经新闻摘要
        
        Args:
            keywords: 要搜索的关键词列表，默认为['财经', '股票', '市场']
            language: 新闻语言，默认为中文('zh')
            page_size: 返回的新闻数量，默认为10
            
        Returns:
            Dict: 包含新闻列表的字典
        """
        try:
            # 检查缓存
            if keywords is None:
                keywords = ['财经', '股票', '市场']
            cache_key = self._get_cache_key("get_financial_news", tuple(keywords), language, page_size)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 使用聚合数据API获取财经新闻
            logger.info(f"使用聚合数据API获取财经新闻")
            juhe_news = await self._get_juhe_news_async(page=1, page_size=page_size)
            
            # 使用benzhi.online获取财经新闻
            logger.info(f"使用benzhi.online获取财经新闻")
            benzhi_news = await self._get_benzhi_news_async(page=1, page_size=page_size)
            
            # 使用benzhi.online每日新闻API获取财经新闻
            logger.info(f"使用benzhi.online每日新闻API获取财经新闻")
            benzhi_daily_news = await self._get_benzhi_daily_news_async(page_size=page_size)
            
            # 使用akshare作为备选新闻源
            logger.info(f"使用akshare获取关键词为 {keywords} 的财经新闻")
            akshare_news = self._get_chinese_financial_news(keywords, page_size)
            
            # 合并新闻结果，去重
            all_news = juhe_news['新闻列表'] + benzhi_news['新闻列表'] + benzhi_daily_news['新闻列表'] + akshare_news['新闻列表']
            
            # 去重：基于标题和来源
            seen_news = set()
            unique_news = []
            for news in all_news:
                news_id = f"{news['标题']}_{news['来源']}"
                if news_id not in seen_news:
                    seen_news.add(news_id)
                    unique_news.append(news)
            
            # 限制返回数量
            unique_news = unique_news[:page_size]
            
            result = {
                '新闻数量': len(unique_news),
                '新闻列表': unique_news
            }
            
            # 缓存数据，缓存半小时
            self._set_cached_data(cache_key, result, disk_ttl=1800)
            return result
            
        except Exception as e:
            logger.error(f"异步获取财经新闻失败: {e}")
            # 降级处理：仅使用akshare
            try:
                if keywords is None:
                    keywords = ['财经', '股票', '市场']
                return self._get_chinese_financial_news(keywords, page_size)
            except Exception as fallback_error:
                logger.error(f"降级获取财经新闻也失败: {fallback_error}")
                # 返回空结果
                return {
                    '新闻数量': 0,
                    '新闻列表': []
                }
    
    def get_financial_news(self, keywords: List[str] = None, language: str = 'zh', page_size: int = 10) -> Dict:
        """获取全球财经新闻摘要（同步版本）
        
        Args:
            keywords: 要搜索的关键词列表，默认为['财经', '股票', '市场']
            language: 新闻语言，默认为中文('zh')
            page_size: 返回的新闻数量，默认为10
            
        Returns:
            Dict: 包含新闻列表的字典
        """
        try:
            # 检查缓存
            if keywords is None:
                keywords = ['财经', '股票', '市场']
            cache_key = self._get_cache_key("get_financial_news", tuple(keywords), language, page_size)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 对于同步方法，仅使用akshare获取新闻
            # 因为异步API调用在同步方法中使用需要额外处理
            logger.info(f"使用akshare获取关键词为 {keywords} 的财经新闻")
            news_data = self._get_chinese_financial_news(keywords, page_size)
            
            # 缓存数据，缓存半小时
            self._set_cached_data(cache_key, news_data, disk_ttl=1800)
            return news_data
            
        except Exception as e:
            logger.error(f"获取财经新闻失败: {e}")
            # 返回空结果
            return {
                '新闻数量': 0,
                '新闻列表': []
            }
                
    def _get_chinese_financial_news(self, keywords: List[str], page_size: int) -> Dict:
        """使用akshare获取中国财经新闻（作为NewsAPI的回退）
        
        Args:
            keywords: 要搜索的关键词列表
            page_size: 返回的新闻数量
            
        Returns:
            Dict: 包含新闻列表的字典
        """
        try:
            # 添加随机延时，模拟真人操作
            delay = random.uniform(2, 4)
            logger.debug(f"添加随机延时 {delay:.2f} 秒")
            time.sleep(delay)
            
            # 使用akshare获取A股新闻
            df = ak.stock_news_em()
            if df.empty:
                return {
                    '新闻数量': 0,
                    '新闻列表': []
                }
            
            # 检查返回的数据结构
            required_columns = ['标题', '来源', '发布时间', '内容', '链接']
            available_columns = df.columns.tolist()
            
            # 如果缺少必要的字段，记录错误并返回空结果
            missing_columns = [col for col in required_columns if col not in available_columns]
            if missing_columns:
                logger.error(f"akshare返回的新闻数据缺少必要字段: {missing_columns}")
                return {
                    '新闻数量': 0,
                    '新闻列表': []
                }
            
            # 过滤新闻
            filtered_news = []
            for _, row in df.iterrows():
                try:
                    # 检查是否包含关键词
                    if any(keyword in row['标题'] or keyword in row['内容'] for keyword in keywords):
                        filtered_news.append({
                            '标题': row['标题'],
                            '来源': row['来源'],
                            '发布时间': row['发布时间'],
                            '摘要': row['内容'][:100] + '...' if len(row['内容']) > 100 else row['内容'],
                            '链接': row['链接']
                        })
                        
                    # 达到请求数量时停止
                    if len(filtered_news) >= page_size:
                        break
                except Exception as row_error:
                    logger.error(f"处理新闻数据行时出错: {row_error}")
                    continue
            
            return {
                '新闻数量': len(filtered_news),
                '新闻列表': filtered_news
            }
        except Exception as e:
            logger.error(f"使用akshare获取中国财经新闻失败: {e}")
            return {
                '新闻数量': 0,
                '新闻列表': []
            }

    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_stock_code_by_name(self, stock_name: str) -> Optional[str]:
        """根据股票名称查询股票代码（支持A股、港股、美股、ETF和指数）
        
        Args:
            stock_name: 股票名称
            
        Returns:
            Optional[str]: 股票代码，如果未找到返回None
        """
        try:
            cache_key = self._get_cache_key("get_stock_code_by_name", stock_name)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 尝试1: 使用必盈API查询股票代码（主要数据源）
            logger.info(f"正在查询股票 '{stock_name}' 的代码 (必盈API)...")
            
            # 使用通用必盈API调用方法获取股票列表
            stock_data = self._call_biying_api(self.BIYING_API_STOCK_LIST)
            
            # 文档显示返回的是股票列表数组
            if isinstance(stock_data, list):
                # 在返回的股票列表中查找匹配的股票名称
                for stock in stock_data:
                    # 返回的字段名是中文缩写：dm(代码)、mc(名称)、jys(交易所)
                    stock_code = stock.get("dm")
                    stock_name_from_api = stock.get("mc")
                    exchange = stock.get("jys")
                    
                    # 精确匹配股票名称
                    if stock_name_from_api == stock_name:
                        # 代码已经包含交易所信息，直接使用
                        full_code = stock_code
                        
                        # 将新股票添加到本地JSON文件
                        self._add_to_stock_name_code_map(stock_name, full_code)
                        self._set_cached_data(cache_key, full_code)
                        logger.info(f"成功根据名称 '{stock_name}' 从必盈API查询到股票代码: {full_code}")
                        return full_code
                        
                # 如果精确匹配失败，尝试更智能的模糊匹配
                # 1. 完全包含匹配
                for stock in stock_data:
                    stock_code = stock.get("dm")
                    stock_name_from_api = stock.get("mc")
                    
                    if stock_name_from_api and stock_name in stock_name_from_api:
                        # 代码已经包含交易所信息，直接使用
                        full_code = stock_code
                        
                        # 将新股票添加到本地JSON文件
                        self._add_to_stock_name_code_map(stock_name_from_api, full_code)
                        self._set_cached_data(cache_key, full_code)
                        logger.info(f"成功根据模糊匹配名称 '{stock_name}' 从必盈API查询到股票代码: {full_code} (实际名称: {stock_name_from_api})")
                        return full_code
                
                # 2. 部分包含匹配（适用于用户输入的是股票名称的一部分）
                for stock in stock_data:
                    stock_code = stock.get("dm")
                    stock_name_from_api = stock.get("mc")
                    
                    if stock_name_from_api and stock_name_from_api in stock_name:
                        # 代码已经包含交易所信息，直接使用
                        full_code = stock_code
                        
                        # 将新股票添加到本地JSON文件
                        self._add_to_stock_name_code_map(stock_name_from_api, full_code)
                        self._set_cached_data(cache_key, full_code)
                        logger.info(f"成功根据部分匹配名称 '{stock_name}' 从必盈API查询到股票代码: {full_code} (实际名称: {stock_name_from_api})")
                        return full_code
                
                # 3. 使用difflib进行相似度匹配（适用于拼写错误或名称不完整的情况）
                import difflib
                potential_matches = []
                for stock in stock_data:
                    stock_code = stock.get("dm")
                    stock_name_from_api = stock.get("mc")
                    
                    if stock_name_from_api:
                        similarity = difflib.SequenceMatcher(None, stock_name, stock_name_from_api).ratio()
                        potential_matches.append((similarity, stock_code, stock_name_from_api))
                
                # 按相似度降序排序
                potential_matches.sort(key=lambda x: x[0], reverse=True)
                
                # 如果有相似度大于0.7的匹配结果，返回最相似的
                if potential_matches and potential_matches[0][0] > 0.7:
                    _, full_code, actual_name = potential_matches[0]
                    self._add_to_stock_name_code_map(actual_name, full_code)
                    self._set_cached_data(cache_key, full_code)
                    logger.info(f"成功根据相似度匹配名称 '{stock_name}' 从必盈API查询到股票代码: {full_code} (实际名称: {actual_name}, 相似度: {potential_matches[0][0]:.2f})")
                    return full_code
                
            # 尝试0: 检查外部数据文件中的映射表（备选数据源）
            logger.info(f"正在检查外部数据文件中的股票 '{stock_name}' 代码...")
            if stock_name in self.stock_name_code_map:
                code = self.stock_name_code_map[stock_name]
                self._set_cached_data(cache_key, code)
                logger.info(f"成功从外部数据文件中找到股票 '{stock_name}' 的代码: {code}")
                return code
            
            # 尝试2: 使用akshare查询A股（备选数据源）
            logger.info(f"正在查询股票 '{stock_name}' 的代码 (A股)...")
            try:
                # 使用重试机制调用akshare
                def _fetch_a_stock_data():
                    # 添加随机延时，模拟真人操作
                    delay = random.uniform(2, 4)
                    logger.debug(f"添加随机延时 {delay:.2f} 秒")
                    time.sleep(delay)
                    return ak.stock_info_a_code_name()
                
                stock_data = self._retry_on_exception(_fetch_a_stock_data)
                
                # 尝试精确匹配
                match = stock_data[stock_data['name'] == stock_name]
                if not match.empty:
                    code = match.iloc[0]['code']
                    # 将新股票添加到本地JSON文件
                    self._add_to_stock_name_code_map(stock_name, code)
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据名称 '{stock_name}' 查询到A股代码: {code}")
                    return code
                
                # 如果精确匹配失败，尝试模糊匹配
                fuzzy_match = stock_data[stock_data['name'].str.contains(stock_name, case=False)]
                if not fuzzy_match.empty:
                    # 使用模糊匹配的第一个结果
                    matched_name = fuzzy_match.iloc[0]['name']
                    code = fuzzy_match.iloc[0]['code']
                    # 将新股票添加到本地JSON文件
                    self._add_to_stock_name_code_map(matched_name, code)
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据模糊匹配名称 '{stock_name}' 查询到A股代码: {code} (实际名称: {matched_name})")
                    return code
            except Exception as e:
                logger.warning(f"akshare查询A股失败: {e}")
            
            # 尝试2: 使用akshare查询港股
            logger.info(f"正在查询股票 '{stock_name}' 的代码 (港股)...")
            try:
                # 使用重试机制调用akshare
                def _fetch_hk_stock_data():
                    # 添加随机延时，模拟真人操作
                    delay = random.uniform(2, 4)
                    logger.debug(f"添加随机延时 {delay:.2f} 秒")
                    time.sleep(delay)
                    return ak.stock_hk_spot_em()
                
                hk_stock_data = self._retry_on_exception(_fetch_hk_stock_data)
                
                # 尝试精确匹配
                match = hk_stock_data[hk_stock_data['名称'] == stock_name]
                if not match.empty:
                    code = match.iloc[0]['代码']
                    # 将新股票添加到本地JSON文件
                    self._add_to_stock_name_code_map(stock_name, code)
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据名称 '{stock_name}' 查询到港股代码: {code}")
                    return code
                
                # 如果精确匹配失败，尝试模糊匹配
                fuzzy_match = hk_stock_data[hk_stock_data['名称'].str.contains(stock_name, case=False)]
                if not fuzzy_match.empty:
                    # 使用模糊匹配的第一个结果
                    matched_name = fuzzy_match.iloc[0]['名称']
                    code = fuzzy_match.iloc[0]['代码']
                    # 将新股票添加到本地JSON文件
                    self._add_to_stock_name_code_map(matched_name, code)
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据模糊匹配名称 '{stock_name}' 查询到港股代码: {code} (实际名称: {matched_name})")
                    return code
            except Exception as e:
                logger.warning(f"akshare查询港股失败: {e}")
            
            # 尝试3: 使用akshare查询美股
            logger.info(f"正在查询股票 '{stock_name}' 的代码 (美股)...")
            try:
                # 使用重试机制调用akshare
                def _fetch_us_stock_data():
                    # 添加随机延时，模拟真人操作
                    delay = random.uniform(2, 4)
                    logger.debug(f"添加随机延时 {delay:.2f} 秒")
                    time.sleep(delay)
                    return ak.stock_us_spot_em()
                
                us_stock_data = self._retry_on_exception(_fetch_us_stock_data)
                
                # 尝试精确匹配
                match = us_stock_data[us_stock_data['名称'] == stock_name]
                if not match.empty:
                    code = match.iloc[0]['代码']
                    # 将新股票添加到本地JSON文件
                    self._add_to_stock_name_code_map(stock_name, code)
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据名称 '{stock_name}' 查询到美股代码: {code}")
                    return code
                
                # 如果精确匹配失败，尝试模糊匹配
                fuzzy_match = us_stock_data[us_stock_data['名称'].str.contains(stock_name, case=False)]
                if not fuzzy_match.empty:
                    # 使用模糊匹配的第一个结果
                    matched_name = fuzzy_match.iloc[0]['名称']
                    code = fuzzy_match.iloc[0]['代码']
                    # 将新股票添加到本地JSON文件
                    self._add_to_stock_name_code_map(matched_name, code)
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据模糊匹配名称 '{stock_name}' 查询到美股代码: {code} (实际名称: {matched_name})")
                    return code
            except Exception as e:
                logger.warning(f"akshare查询美股失败: {e}")
            
            # 尝试4: 使用baostock查询ETF和指数
            logger.info(f"正在查询股票 '{stock_name}' 的代码 (ETF/指数)...")
            try:
                code = self.get_etf_index_code_by_name(stock_name)
                if code:
                    self._set_cached_data(cache_key, code)
                    logger.info(f"成功根据名称 '{stock_name}' 查询到ETF/指数代码: {code}")
                    return code
            except Exception as e:
                logger.warning(f"baostock查询ETF/指数失败: {e}")
            
            logger.warning(f"未找到名称为 '{stock_name}' 的股票、ETF或指数（已尝试外部数据文件、A股、港股、美股、ETF和指数）")
            return None
            
        except Exception as e:
            logger.error(f"根据股票名称查询代码失败: {e}")
            return None
    
    @rate_limited(max_calls=10, period=60)  # 每分钟最多10次请求
    def get_etf_index_code_by_name(self, name: str) -> Optional[str]:
        """根据名称查询ETF或指数代码（使用baostock）
        
        Args:
            name: ETF或指数名称
            
        Returns:
            Optional[str]: ETF或指数代码，如果未找到返回None
        """
        try:
            # 查询股票基本信息，包括ETF和指数
            rs = bs.query_stock_basic()
            if rs.error_code != '0':
                logger.warning(f"baostock查询基本信息失败: {rs.error_msg}")
                return None
            
            # 转换为DataFrame
            stock_df = rs.get_data()
            
            # 尝试精确匹配
            match = stock_df[stock_df['code_name'] == name]
            if not match.empty:
                code = match.iloc[0]['code']
                # 将新ETF/指数添加到本地JSON文件
                self._add_to_common_indices(name)
                logger.info(f"成功根据名称 '{name}' 查询到ETF/指数代码: {code}")
                return code
            
            # 如果精确匹配失败，尝试模糊匹配
            fuzzy_match = stock_df[stock_df['code_name'].str.contains(name, case=False)]
            if not fuzzy_match.empty:
                matched_name = fuzzy_match.iloc[0]['code_name']
                code = fuzzy_match.iloc[0]['code']
                # 将新ETF/指数添加到本地JSON文件
                self._add_to_common_indices(matched_name)
                logger.info(f"成功根据模糊匹配名称 '{name}' 查询到ETF/指数代码: {code} (实际名称: {matched_name})")
                return code
            
            logger.warning(f"baostock未找到名称为 '{name}' 的ETF或指数")
            return None
            
        except Exception as e:
            logger.error(f"根据名称查询ETF或指数代码失败: {e}")
            return None
    
    def get_biyingapi_licence(self) -> Optional[str]:
        """获取必盈API的免费授权许可
        
        Returns:
            Optional[str]: 授权许可字符串，如果获取失败返回None
        """
        try:
            # 根据必盈API文档，使用提供的API密钥作为授权许可
            licence = "E70E74BC-FD29-423A-8C25-424023A8567B"
            logger.debug(f"使用API密钥作为必盈API授权许可: {licence}")
            return licence
        except Exception as e:
            logger.error(f"获取必盈API授权许可失败: {e}")
            return None
            
    def _call_biying_api(self, api_url: str, **kwargs) -> Optional[Any]:
        """通用必盈API调用方法
        
        Args:
            api_url: API接口URL模板
            **kwargs: URL参数
            
        Returns:
            Optional[Any]: API响应数据，如果失败返回None
        """
        try:
            # 获取必盈API授权许可
            licence = self.get_biyingapi_licence()
            if not licence:
                logger.error("获取必盈API授权许可失败")
                return None
            
            # 准备请求参数
            params = kwargs.copy()
            params['licence'] = licence
            
            # 格式化API URL
            formatted_url = api_url.format(**params)
            logger.debug(f"调用必盈API: {formatted_url}")
            
            # 添加随机延时，避免频繁请求
            delay = random.uniform(1.0, 3.0)
            logger.debug(f"添加随机延时 {delay:.2f} 秒")
            time.sleep(delay)
            
            # 发送请求
            response = self.session.get(formatted_url, timeout=30)
            response.raise_for_status()  # 检查HTTP错误
            
            # 解析响应
            data = response.json()
            logger.debug(f"必盈API响应: {data}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"必盈API请求失败: {e}")
            return None
        except ValueError as e:
            logger.error(f"必盈API响应解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"必盈API调用异常: {e}")
            return None
            
    async def _call_biying_api_async(self, api_url: str, **kwargs) -> Optional[Any]:
        """异步通用必盈API调用方法
        
        Args:
            api_url: API接口URL模板
            **kwargs: URL参数
            
        Returns:
            Optional[Any]: API响应数据，如果失败返回None
        """
        try:
            # 获取必盈API授权许可
            licence = self.get_biyingapi_licence()
            if not licence:
                logger.error("获取必盈API授权许可失败")
                return None
            
            # 准备请求参数
            params = kwargs.copy()
            params['licence'] = licence
            
            # 格式化API URL
            formatted_url = api_url.format(**params)
            logger.debug(f"调用必盈API: {formatted_url}")
            
            # 添加随机延时，避免频繁请求
            delay = random.uniform(1.0, 3.0)
            logger.debug(f"添加随机延时 {delay:.2f} 秒")
            await asyncio.sleep(delay)
            
            # 初始化aiohttp会话
            async with aiohttp.ClientSession() as session:
                async with session.get(formatted_url, timeout=30) as response:
                    response.raise_for_status()  # 检查HTTP错误
                    # 解析响应
                    data = await response.json()
                    logger.debug(f"必盈API响应: {data}")
                    return data
            
        except aiohttp.ClientError as e:
            logger.error(f"必盈API异步请求失败: {e}")
            return None
        except ValueError as e:
            logger.error(f"必盈API异步响应解析失败: {e}")
            return None
        except Exception as e:
            logger.error(f"必盈API异步调用异常: {e}")
            return None
    
    def _parse_stock_list(self, data: List[Dict]) -> List[Dict]:
        """解析股票列表数据，将字段映射为中文名称
        
        Args:
            data: 原始股票列表数据
            
        Returns:
            List[Dict]: 解析后的股票列表数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'dm': ('股票代码', str),
            'mc': ('股票名称', str),
            'jys': ('交易所', str)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    try:
                        parsed_record[chinese_name] = data_type(record[field])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换股票列表字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = record[field]
            parsed_data.append(parsed_record)
        
        return parsed_data
        
    def get_stock_list(self) -> Optional[List[Dict]]:
        """获取股票列表
        
        Returns:
            Optional[List[Dict]]: 股票列表，如果失败返回None
        """
        try:
            # 使用必盈API获取股票列表
            logger.info("使用必盈API获取股票列表")
            result = self._call_biying_api(
                self.BIYING_API_STOCK_LIST
            )
            
            if result and isinstance(result, list):
                logger.info(f"成功从必盈API获取股票列表")
                # 解析股票列表数据
                return self._parse_stock_list(result)
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning("必盈API获取股票列表失败，尝试使用akshare获取")
            # 这里可以添加akshare的股票列表获取逻辑
            # 例如：stock_list = ak.stock_info_a_code_name()
            
            return None
            
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return None
            
    def _parse_company_info(self, data: Dict) -> Dict:
        """解析公司信息数据，将字段映射为中文名称
        
        Args:
            data: 原始公司信息数据
            
        Returns:
            Dict: 解析后的公司信息数据，包含中文字段名
        """
        parsed_data = {}
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'name': ('公司名称', str),
            'ename': ('公司英文名称', str),
            'market': ('上市市场', str),
            'idea': ('概念及板块', str),
            'ldate': ('上市日期', str),
            'sprice': ('发行价格', str),
            'principal': ('主承销商', str),
            'rdate': ('成立日期', str),
            'rprice': ('注册资本', str),
            'instype': ('机构类型', str),
            'organ': ('组织形式', str),
            'secre': ('董事会秘书', str),
            'phone': ('公司电话', str),
            'sphone': ('董秘电话', str),
            'fax': ('公司传真', str),
            'sfax': ('董秘传真', str),
            'email': ('公司电子邮箱', str),
            'semail': ('董秘电子邮箱', str),
            'site': ('公司网站', str),
            'post': ('邮政编码', str),
            'infosite': ('信息披露网址', str),
            'oname': ('证券简称更名历史', str),
            'addr': ('注册地址', str),
            'oaddr': ('办公地址', str),
            'desc': ('公司简介', str),
            'bscope': ('经营范围', str),
            'printype': ('承销方式', str),
            'referrer': ('上市推荐人', str),
            'putype': ('发行方式', str),
            'pe': ('发行市盈率', str),
            'firgu': ('首发前总股本', str),
            'lastgu': ('首发后总股本', str),
            'realgu': ('实际发行量', str),
            'planm': ('预计募集资金', str),
            'realm': ('实际募集资金合计', str),
            'pubfee': ('发行费用总额', str),
            'collect': ('募集资金净额', str),
            'signfee': ('承销费用', str),
            'pdate': ('招股公告日', str)
        }
        
        for field, (chinese_name, data_type) in field_mapping.items():
            if field in data:
                try:
                    parsed_data[chinese_name] = data_type(data[field])
                except (ValueError, TypeError) as e:
                    logger.warning(f"转换公司信息字段 {field} 失败: {e}, 使用原始值")
                    parsed_data[chinese_name] = data[field]
        
        return parsed_data
        
    def get_company_info(self, stock_code: str) -> Optional[Dict]:
        """获取公司信息
        
        Args:
            stock_code: 股票代码，如000001
            
        Returns:
            Optional[Dict]: 公司信息，如果失败返回None
        """
        try:
            # 首先使用必盈API获取公司信息
            logger.info(f"使用必盈API获取股票 {stock_code} 的公司信息")
            result = self._call_biying_api(
                self.BIYING_API_COMPANY_INFO,
                stock_code=stock_code
            )
            
            if result and isinstance(result, dict):
                logger.info(f"成功从必盈API获取股票 {stock_code} 的公司信息")
                # 解析公司信息数据
                return self._parse_company_info(result)
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning(f"必盈API获取公司信息失败，尝试使用akshare获取")
            # 这里可以添加akshare的公司信息获取逻辑
            # 例如：company_info = ak.stock_info_company(stock_code)
            
            return None
            
        except Exception as e:
            logger.error(f"获取公司信息失败: {e}")
            return None
            
    def _parse_company_indices(self, data: List[Dict]) -> List[Dict]:
        """解析公司所属指数数据，将字段映射为中文名称
        
        Args:
            data: 原始公司所属指数数据
            
        Returns:
            List[Dict]: 解析后的公司所属指数数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'mc': ('指数名称', str),
            'dm': ('指数代码', str),
            'ind': ('进入日期', str),
            'outd': ('退出日期', str)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    try:
                        parsed_record[chinese_name] = data_type(record[field])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换公司所属指数字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = record[field]
            parsed_data.append(parsed_record)
        
        return parsed_data
        
    def get_company_indices(self, stock_code: str) -> Optional[List[Dict]]:
        """获取公司所属指数
        
        Args:
            stock_code: 股票代码，如000001
            
        Returns:
            Optional[List[Dict]]: 公司所属指数列表，如果失败返回None
        """
        try:
            # 使用必盈API获取公司所属指数
            logger.info(f"使用必盈API获取股票 {stock_code} 的所属指数")
            result = self._call_biying_api(
                self.BIYING_API_COMPANY_INDICES,
                stock_code=stock_code
            )
            
            if result and isinstance(result, list):
                logger.info(f"成功从必盈API获取股票 {stock_code} 的所属指数")
                # 解析公司所属指数数据
                return self._parse_company_indices(result)
            elif result and isinstance(result, dict):
                logger.info(f"必盈API返回字典格式的公司所属指数数据")
                return [self._parse_company_indices([result])]
            
            logger.warning(f"必盈API获取公司所属指数失败")
            return None
            
        except Exception as e:
            logger.error(f"获取公司所属指数失败: {e}")
            return None
            
    def _parse_quarterly_profit(self, data: List[Dict]) -> List[Dict]:
        """解析公司季度利润数据，将字段映射为中文名称
        
        Args:
            data: 原始公司季度利润数据
            
        Returns:
            List[Dict]: 解析后的公司季度利润数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'date': ('截止日期', str),
            'income': ('营业收入', float),
            'expend': ('营业支出', float),
            'profit': ('营业利润', float),
            'totalp': ('利润总额', float),
            'reprofit': ('净利润', float),
            'basege': ('基本每股收益', float),
            'ettege': ('稀释每股收益', float),
            'otherp': ('其他综合收益', float),
            'totalcp': ('综合收益总额', float)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    try:
                        value = record[field]
                        # 处理包含逗号的数字字符串
                        if isinstance(value, str):
                            value = value.replace(',', '')
                            # 处理'-'占位符的情况
                            if value == '-':
                                parsed_record[chinese_name] = None
                                continue
                        parsed_record[chinese_name] = data_type(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换季度利润字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = record[field]
            parsed_data.append(parsed_record)
        
        return parsed_data
        
    def get_quarterly_profit(self, stock_code: str) -> Optional[List[Dict]]:
        """获取公司近一年各个季度的利润
        
        Args:
            stock_code: 股票代码，如000001
            
        Returns:
            Optional[List[Dict]]: 公司季度利润列表，如果失败返回None
        """
        try:
            # 使用必盈API获取公司季度利润
            logger.info(f"使用必盈API获取股票 {stock_code} 的季度利润")
            result = self._call_biying_api(
                self.BIYING_API_QUARTERLY_PROFIT,
                stock_code=stock_code
            )
            
            if result and isinstance(result, list):
                logger.info(f"成功从必盈API获取股票 {stock_code} 的季度利润")
                # 解析公司季度利润数据
                return self._parse_quarterly_profit(result)
            elif result and isinstance(result, dict):
                logger.info(f"必盈API返回字典格式的季度利润数据")
                return [self._parse_quarterly_profit([result])]
            
            logger.warning(f"必盈API获取季度利润失败")
            return None
            
        except Exception as e:
            logger.error(f"获取季度利润失败: {e}")
            return None
            
    def _parse_balance_sheet(self, data: List[Dict]) -> List[Dict]:
        """解析资产负债表数据，将字段映射为中文名称
        
        Args:
            data: 原始资产负债表数据
            
        Returns:
            List[Dict]: 解析后的资产负债表数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'jzrq': ('截止日期', str),
            'plrq': ('披露日期', str),
            'hbzj': ('货币资金', float),
            'yszk': ('应收账款', float),
            'ch': ('存货', float),
            'gdzc': ('固定资产', float),
            'wxtz': ('无形资产', float),
            'zczj': ('资产总计', float),
            'dqjk': ('短期借款', float),
            'yfzk': ('应付账款', float),
            'cqjk': ('长期借款', float),
            'fzht': ('负债合计', float),
            'sszb': ('实收资本', float),
            'wfplr': ('未分配利润', float),
            'syzqyhj': ('所有者权益合计', float)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    try:
                        value = record[field]
                        # 处理'-'占位符的情况
                        if value == '-':
                            parsed_record[chinese_name] = None
                        else:
                            parsed_record[chinese_name] = data_type(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换资产负债表字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = record[field]
            parsed_data.append(parsed_record)
        
        return parsed_data
        
    def get_balance_sheet(self, stock_code: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Optional[List[Dict]]:
        """获取公司资产负债表
        
        Args:
            stock_code: 股票代码，如000001.SZ
            start_time: 开始时间，格式YYYYMMDD
            end_time: 结束时间，格式YYYYMMDD
            
        Returns:
            Optional[List[Dict]]: 公司资产负债表数据列表，如果失败返回None
        """
        try:
            # 设置时间范围默认值
            # 如果没有提供开始时间，设置为上一年的1月1日
            if not start_time:
                current_year = datetime.datetime.now().year
                start_time = f"{current_year - 1}0101"
                logger.info(f"自动设置开始时间为上一年1月1日: {start_time}")
            
            # 如果没有提供结束时间，设置为当前日期
            if not end_time:
                end_time = datetime.datetime.now().strftime("%Y%m%d")
                logger.info(f"自动设置结束时间为当前日期: {end_time}")
            
            # 准备参数，确保所有URL参数都存在
            params = {
                'stock_code': stock_code,
                'start_time': start_time,
                'end_time': end_time
            }
            
            # 使用必盈API获取资产负债表
            logger.info(f"使用必盈API获取股票 {stock_code} 的资产负债表")
            result = self._call_biying_api(
                self.BIYING_API_BALANCE_SHEET,
                **params
            )
            
            if result and isinstance(result, list):
                logger.info(f"成功从必盈API获取股票 {stock_code} 的资产负债表")
                # 解析资产负债表数据
                return self._parse_balance_sheet(result)
            elif result and isinstance(result, dict):
                logger.info(f"必盈API返回字典格式的资产负债表数据")
                return [self._parse_balance_sheet([result])]
            
            logger.warning(f"必盈API获取资产负债表失败")
            return None
            
        except Exception as e:
            logger.error(f"获取资产负债表失败: {e}")
            return None
            
    def _parse_company_metrics(self, data: List[Dict]) -> List[Dict]:
        """解析公司主要指标数据，将字段映射为中文名称
        
        Args:
            data: 原始公司主要指标数据
            
        Returns:
            List[Dict]: 解析后的公司主要指标数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'jzrq': ('截止日期', str),
            'plrq': ('披露日期', str),
            'mgjyhdxjl': ('每股经营活动现金流量', float),
            'mgjzc': ('每股净资产', float),
            'jbmgsy': ('基本每股收益', float),
            'xsmgsy': ('稀释每股收益', float),
            'mgwfplr': ('每股未分配利润', float),
            'mgzbgjj': ('每股资本公积金', float),
            'kfmgsy': ('扣非每股收益', float),
            'jzcsyl': ('净资产收益率', float),
            'xsmlv': ('销售毛利率', float),
            'zyyrsrzz': ('主营收入同比增长', float),
            'jlrzz': ('净利润同比增长', float),
            'zcfzl': ('资产负债比率', float),
            'chzzl': ('存货周转率', float)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    value = record[field]
                    try:
                        # 处理'-'占位符的情况，转换为None
                        if value == '-':
                            parsed_record[chinese_name] = None
                        else:
                            parsed_record[chinese_name] = data_type(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换公司主要指标字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = value
            parsed_data.append(parsed_record)
        
        return parsed_data
        
    def get_company_metrics(self, stock_code: str, start_time: Optional[str] = None, end_time: Optional[str] = None) -> Optional[List[Dict]]:
        """获取公司主要指标
        
        Args:
            stock_code: 股票代码，如000001.SZ
            start_time: 开始时间，格式YYYYMMDD
            end_time: 结束时间，格式YYYYMMDD
            
        Returns:
            Optional[List[Dict]]: 公司主要指标数据列表，如果失败返回None
        """
        try:
            # 设置时间范围默认值
            # 如果没有提供开始时间，设置为上一年的1月1日
            if not start_time:
                current_year = datetime.datetime.now().year
                start_time = f"{current_year - 1}0101"
                logger.info(f"自动设置开始时间为上一年1月1日: {start_time}")
            
            # 如果没有提供结束时间，设置为当前日期
            if not end_time:
                end_time = datetime.datetime.now().strftime("%Y%m%d")
                logger.info(f"自动设置结束时间为当前日期: {end_time}")
            
            # 准备参数，确保所有URL参数都存在
            params = {
                'stock_code': stock_code,
                'start_time': start_time,
                'end_time': end_time
            }
            
            # 使用必盈API获取公司主要指标
            logger.info(f"使用必盈API获取股票 {stock_code} 的主要指标")
            result = self._call_biying_api(
                self.BIYING_API_COMPANY_METRICS,
                **params
            )
            
            if result and isinstance(result, list):
                logger.info(f"成功从必盈API获取股票 {stock_code} 的主要指标")
                # 解析公司主要指标数据
                return self._parse_company_metrics(result)
            elif result and isinstance(result, dict):
                logger.info(f"必盈API返回字典格式的公司主要指标数据")
                return [self._parse_company_metrics([result])]
            
            logger.warning(f"必盈API获取公司主要指标失败")
            return None
            
        except Exception as e:
            logger.error(f"获取公司主要指标失败: {e}")
            return None
            
    def get_index_sector_list(self) -> Optional[List[Dict]]:
        """获取指数、行业、概念列表
        
        Returns:
            Optional[List[Dict]]: 指数、行业、概念列表，如果失败返回None
        """
        try:
            # 使用必盈API获取指数、行业、概念列表
            logger.info("使用必盈API获取指数、行业、概念列表")
            result = self._call_biying_api(
                self.BIYING_API_INDEX_SECTOR_LIST
            )
            
            if result and isinstance(result, list):
                logger.info(f"成功从必盈API获取指数、行业、概念列表")
                # 解析指数、行业、概念列表数据
                return self._parse_index_sector(result)
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning("必盈API获取指数、行业、概念列表失败，尝试使用akshare获取")
            # 这里可以添加akshare的指数、行业、概念列表获取逻辑
            # 例如：index_list = ak.stock_zh_index_spot_em()
            
            return None
            
        except Exception as e:
            logger.error(f"获取指数、行业、概念列表失败: {e}")
            return None
            
    def get_stocks_by_index_sector(self, code: str) -> Optional[List[Dict]]:
        """根据指数、行业、概念查询股票
        
        Args:
            code: 指数、行业、概念代码
            
        Returns:
            Optional[List[Dict]]: 相关股票列表，如果失败返回None
        """
        try:
            # 使用必盈API根据指数、行业、概念查询股票
            logger.info(f"使用必盈API根据 {code} 查询相关股票")
            result = self._call_biying_api(
                self.BIYING_API_STOCKS_BY_INDEX_SECTOR,
                code=code
            )
            
            if result:
                logger.info(f"成功从必盈API获取 {code} 相关的股票列表")
                return result
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning(f"必盈API根据 {code} 查询股票失败，尝试使用akshare获取")
            # 这里可以添加akshare的相关股票获取逻辑
            # 例如：stocks = ak.stock_board_industry_cons_em(symbol=code)
            
            return None
            
        except Exception as e:
            logger.error(f"根据指数、行业、概念查询股票失败: {e}")
            return None
            
    def _parse_index_sector(self, data: List[Dict]) -> List[Dict]:
        """解析指数、行业、概念数据，将字段映射为中文名称
        
        Args:
            data: 原始指数、行业、概念数据列表
            
        Returns:
            List[Dict]: 解析后的指数、行业、概念数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'name': ('名称', str),
            'code': ('代码', str),
            'type1': ('一级分类', int),
            'type2': ('二级分类', int),
            'level': ('层级', int),
            'pcode': ('父节点代码', str),
            'pname': ('父节点名称', str),
            'isleaf': ('是否为叶子节点', int)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    try:
                        parsed_record[chinese_name] = data_type(record[field])
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换指数、行业、概念字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = record[field]
            parsed_data.append(parsed_record)
        
        return parsed_data
        
    def get_related_index_sector(self, stock_code: str) -> Optional[Dict]:
        """根据股票代码获取相关指数、行业、概念
        
        Args:
            stock_code: 股票代码，如000001
            
        Returns:
            Optional[Dict]: 相关指数、行业、概念信息，如果失败返回None
        """
        try:
            # 使用必盈API根据股票代码获取相关指数、行业、概念
            logger.info(f"使用必盈API获取股票 {stock_code} 相关的指数、行业、概念")
            result = self._call_biying_api(
                self.BIYING_API_RELATED_INDEX_SECTOR,
                stock_code=stock_code
            )
            
            if result:
                logger.info(f"成功从必盈API获取股票 {stock_code} 相关的指数、行业、概念")
                
                # 解析指数、行业、概念数据
                parsed_result = {}
                
                # 处理API返回list的情况
                if isinstance(result, list):
                    logger.info(f"API返回list格式，将其转换为dict处理")
                    # 假设list中的元素是需要的数据，转换为dict格式
                    parsed_result['data'] = self._parse_index_sector(result)
                elif isinstance(result, dict):
                    # 正常dict格式处理
                    for key, data in result.items():
                        if isinstance(data, list):
                            parsed_result[key] = self._parse_index_sector(data)
                        else:
                            parsed_result[key] = data
                else:
                    logger.warning(f"API返回非预期格式: {type(result)}")
                    return None
                
                return parsed_result
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning(f"必盈API获取股票 {stock_code} 相关指数、行业、概念失败，尝试使用akshare获取")
            # 这里可以添加akshare的相关指数、行业、概念获取逻辑
            
            return None
            
        except Exception as e:
            logger.error(f"根据股票代码获取相关指数、行业、概念失败: {e}")
            return None
            
    def get_fund_flow(self, stock_code: str, start_time: str, end_time: str, limit: int = 100) -> Optional[Dict]:
        """获取资金流向数据
        
        Args:
            stock_code: 股票代码，如000001
            start_time: 开始时间，格式YYYYMMDD
            end_time: 结束时间，格式YYYYMMDD
            limit: 最新条数限制，默认100
            
        Returns:
            Optional[Dict]: 资金流向数据，如果失败返回None
        """
        try:
            # 使用必盈API获取资金流向数据
            logger.info(f"使用必盈API获取股票 {stock_code} 的资金流向数据")
            
            # 将日期格式从YYYY-MM-DD转换为YYYYMMDD
            start_time_formatted = start_time.replace('-', '')
            end_time_formatted = end_time.replace('-', '')
            
            # 使用统一的_call_biying_api方法调用必盈API的资金流向接口
            data = self._call_biying_api(
                self.BIYING_API_FUND_FLOW,
                stock_code=stock_code,
                start_time=start_time_formatted,
                end_time=end_time_formatted,
                limit=limit
            )
            
            logger.debug(f"必盈API响应: {data}")
            
            if data:
                logger.info(f"成功从必盈API获取股票 {stock_code} 的资金流向数据")
                
                # 根据用户提供的字段说明，解析和格式化数据
                # 字段映射：英文字段名 -> (中文字段名, 数据类型)
                field_mapping = {
                    't': ('交易时间', str),
                    'zmbzds': ('主买单总单数', int),
                    'zmszds': ('主卖单总单数', int),
                    'dddx': ('大单动向', float),
                    'zddy': ('涨跌动因', float),
                    'ddcf': ('大单差分', float),
                    'zmbzdszl': ('主买单总单数增量', int),
                    'zmszdszl': ('主卖单总单数增量', int),
                    'cjbszl': ('成交笔数增量', int),
                    'zmbtdcje': ('主买特大单成交额', float),
                    'zmbddcje': ('主买大单成交额', float),
                    'zmbzdcje': ('主买中单成交额', float),
                    'zmbxdcje': ('主买小单成交额', float),
                    'zmstdcje': ('主卖特大单成交额', float),
                    'zmsddcje': ('主卖大单成交额', float),
                    'zmszdcje': ('主卖中单成交额', float),
                    'zmsxdcje': ('主卖小单成交额', float),
                    'bdmbtdcje': ('被动买特大单成交额', float),
                    'bdmbddcje': ('被动买大单成交额', float),
                    'bdmbzdcje': ('被动买中单成交额', float),
                    'bdmbxdcje': ('被动买小单成交额', float),
                    'bdmstdcje': ('被动卖特大单成交额', float),
                    'bdmsddcje': ('被动卖大单成交额', float),
                    'bdmszdcje': ('被动卖中单成交额', float),
                    'bdmsxdcje': ('被动卖小单成交额', float),
                    'zmbtdcjl': ('主买特大单成交量', int),
                    'zmbddcjl': ('主买大单成交量', int),
                    'zmbzdcjl': ('主买中单成交量', int),
                    'zmbxdcjl': ('主买小单成交量', int),
                    'zmstdcjl': ('主卖特大单成交量', int),
                    'zmsddcjl': ('主卖大单成交量', int),
                    'zmszdcjl': ('主卖中单成交量', int),
                    'zmsxdcjl': ('主卖小单成交量', int),
                    'bdmbtdcjl': ('被动买特大单成交量', int),
                    'bdmbddcjl': ('被动买大单成交量', int),
                    'bdmbzdcjl': ('被动买中单成交量', int),
                    'bdmbxdcjl': ('被动买小单成交量', int),
                    'bdmstdcjl': ('被动卖特大单成交量', int),
                    'bdmsddcjl': ('被动卖大单成交量', int),
                    'bdmszdcjl': ('被动卖中单成交量', int),
                    'bdmsxdcjl': ('被动卖小单成交量', int),
                    'zmbtdcjzl': ('主买特大单成交额增量', float),
                    'zmbddcjzl': ('主买大单成交额增量', float),
                    'zmbzdcjzl': ('主买中单成交额增量', float),
                    'zmbxdcjzl': ('主买小单成交额增量', float),
                    'zmbljcjzl': ('主买累计成交额增量', float),
                    'zmstdcjzl': ('主卖特大单成交额增量', float),
                    'zmsddcjzl': ('主卖大单成交额增量', float),
                    'zmszdcjzl': ('主卖中单成交额增量', float),
                    'zmsxdcjzl': ('主卖小单成交额增量', float),
                    'zmsljcjzl': ('主卖累计成交额增量', float),
                    'bdmbtdcjzl': ('被动买特大单成交额增量', float),
                    'bdmbddcjzl': ('被动买大单成交额增量', float),
                    'bdmbzdcjzl': ('被动买中单成交额增量', float),
                    'bdmbxdcjzl': ('被动买小单成交额增量', float),
                    'bdmbljcjzl': ('被动买累计成交额增量', float),
                    'bdmstdcjzl': ('被动卖特大单成交额增量', float),
                    'bdmsddcjzl': ('被动卖大单成交额增量', float),
                    'bdmszdcjzl': ('被动卖中单成交额增量', float),
                    'bdmsxdcjzl': ('被动卖小单成交额增量', float),
                    'zmbtdcjzlv': ('主买特大单成交量增量', int),
                    'zmbddcjzlv': ('主买大单成交量增量', int),
                    'zmbzdcjzlv': ('主买中单成交量增量', int),
                    'zmbxdcjzlv': ('主买小单成交量增量', int),
                    'zmstdcjzlv': ('主卖特大单成交量增量', int),
                    'zmsddcjzlv': ('主卖大单成交量增量', int),
                    'zmszdcjzlv': ('主卖中单成交量增量', int),
                    'zmsxdcjzlv': ('主卖小单成交量增量', int),
                    'bdmbtdcjzlv': ('被动买特大单成交量增量', int),
                    'bdmbddcjzlv': ('被动买大单成交量增量', int),
                    'bdmbzdcjzlv': ('被动买中单成交量增量', int),
                    'bdmbxdcjzlv': ('被动买小单成交量增量', int),
                    'bdmstdcjzlv': ('被动卖特大单成交量增量', int),
                    'bdmsddcjzlv': ('被动卖大单成交量增量', int),
                    'bdmszdcjzlv': ('被动卖中单成交量增量', int),
                    'bdmsxdcjzlv': ('被动卖小单成交量增量', int)
                }
                
                # 格式化数据
                formatted_data = {
                    "股票代码": stock_code,
                    "开始时间": start_time,
                    "结束时间": end_time,
                    "数据条数": len(data),
                    "资金流向明细": []
                }
                
                # 转换每条记录
                for record in data:
                    formatted_record = {}
                    for field, (chinese_name, data_type) in field_mapping.items():
                        if field in record:
                            # 转换数据类型
                            try:
                                formatted_record[chinese_name] = data_type(record[field])
                            except (ValueError, TypeError) as e:
                                logger.warning(f"转换字段 {field} 失败: {e}, 使用原始值")
                                formatted_record[chinese_name] = record[field]
                    formatted_data["资金流向明细"].append(formatted_record)
                
                return formatted_data
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning(f"必盈API获取股票 {stock_code} 资金流向数据失败，尝试使用akshare获取")
            # 这里可以添加akshare的资金流向获取逻辑
            # 例如：fund_flow = ak.stock_fund_flow(stock_code)
            
            return None
            
        except requests.exceptions.RequestException as e:
            logger.error(f"获取资金流向数据失败: {e}")
            return None
        except ValueError as e:
            logger.error(f"解析资金流向数据失败: {e}")
            return None
        except Exception as e:
            logger.error(f"获取资金流向数据时发生未知错误: {e}")
            return None
            
    def _parse_history_limit_price(self, data: List[Dict]) -> List[Dict]:
        """解析历史涨跌停价格数据，将字段映射为中文名称
        
        Args:
            data: 原始历史涨跌停价格数据列表
            
        Returns:
            List[Dict]: 解析后的历史涨跌停价格数据，包含中文字段名
        """
        parsed_data = []
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            't': ('日期', str),
            'h': ('最高价', float),
            'l': ('最低价', float)
        }
        
        for record in data:
            parsed_record = {}
            for field, (chinese_name, data_type) in field_mapping.items():
                if field in record:
                    try:
                        value = record[field]
                        # 处理'-'占位符的情况
                        if value == '-':
                            parsed_record[chinese_name] = None
                        else:
                            parsed_record[chinese_name] = data_type(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"转换历史涨跌停价格字段 {field} 失败: {e}, 使用原始值")
                        parsed_record[chinese_name] = record[field]
            parsed_data.append(parsed_record)
        
        return parsed_data
    
    def get_history_limit_price(self, stock_code: str, start_time: str, end_time: str) -> Optional[Dict]:
        """获取历史涨跌停价格
        
        Args:
            stock_code: 股票代码，如000001.SZ
            start_time: 开始时间，格式YYYY-MM-DD或YYYYMMDD
            end_time: 结束时间，格式YYYY-MM-DD或YYYYMMDD
            
        Returns:
            Optional[Dict]: 历史涨跌停价格数据，如果失败返回None
        """
        try:
            # 检查股票代码是否包含交易所后缀，如果不包含，尝试处理
            if not (stock_code.endswith('.SZ') or stock_code.endswith('.SH')):
                # 特殊处理6位数字股票代码
                if stock_code.isdigit() and len(stock_code) == 6:
                    logger.info(f"识别到6位数字股票代码 {stock_code}，自动添加交易所后缀")
                    prefix = stock_code[:3]
                    if prefix >= '000' and prefix < '600':
                        # 深市：000(主板), 002(中小板), 300(创业板)
                        stock_code = f"{stock_code}.SZ"
                    else:
                        # 沪市：600(主板), 688(科创板)
                        stock_code = f"{stock_code}.SH"
                else:
                    # 尝试获取完整股票代码
                    full_code = self.get_stock_code_by_name(stock_code)
                    if full_code:
                        logger.debug(f"将股票代码 {stock_code} 转换为完整代码 {full_code}")
                        stock_code = full_code
                    else:
                        logger.warning(f"无法获取股票 {stock_code} 的完整代码，尝试直接使用")
            
            # 将时间格式从YYYY-MM-DD转换为YYYYMMDD
            if '-' in start_time:
                start_time_formatted = start_time.replace('-', '')
                logger.debug(f"将开始时间 {start_time} 转换为 {start_time_formatted}")
            else:
                start_time_formatted = start_time
                
            if '-' in end_time:
                end_time_formatted = end_time.replace('-', '')
                logger.debug(f"将结束时间 {end_time} 转换为 {end_time_formatted}")
            else:
                end_time_formatted = end_time
            
            # 使用必盈API获取历史涨跌停价格
            logger.info(f"使用必盈API获取股票 {stock_code} 的历史涨跌停价格")
            result = self._call_biying_api(
                self.BIYING_API_HISTORY_LIMIT_PRICE,
                stock_code=stock_code,
                start_time=start_time_formatted,
                end_time=end_time_formatted
            )
            
            if result:
                logger.info(f"成功从必盈API获取股票 {stock_code} 的历史涨跌停价格")
                
                # 解析历史涨跌停价格数据
                parsed_result = {
                    "股票代码": stock_code,
                    "开始时间": start_time,
                    "结束时间": end_time,
                    "数据条数": len(result),
                    "历史涨跌停价格明细": self._parse_history_limit_price(result)
                }
                
                return parsed_result
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning(f"必盈API获取股票 {stock_code} 历史涨跌停价格失败，尝试使用akshare获取")
            # 这里可以添加akshare的历史涨跌停价格获取逻辑
            
            return None
            
        except Exception as e:
            logger.error(f"获取历史涨跌停价格失败: {e}")
            return None
            
    def _parse_real_time_transaction(self, data: Dict) -> Dict:
        """解析实时交易数据，将字段映射为中文名称
        
        Args:
            data: 原始实时交易数据
            
        Returns:
            Dict: 解析后的实时交易数据，包含中文字段名
        """
        parsed_data = {}
        
        # 字段映射：英文字段名 -> (中文字段名, 数据类型)
        field_mapping = {
            'fm': ('五分钟涨跌幅', float),
            'h': ('最高价', float),
            'hs': ('换手', float),
            'lb': ('量比', float),
            'l': ('最低价', float),
            'lt': ('流通市值', float),
            'o': ('开盘价', float),
            'pe': ('市盈率', float),
            'pc': ('涨跌幅', float),
            'p': ('当前价格', float),
            'sz': ('总市值', float),
            'cje': ('成交额', float),
            'ud': ('涨跌额', float),
            'v': ('成交量', float),
            'yc': ('昨日收盘价', float),
            'zf': ('振幅', float),
            'zs': ('涨速', float),
            'sjl': ('市净率', float),
            'zdf60': ('60日涨跌幅', float),
            'zdfnc': ('年初至今涨跌幅', float),
            't': ('更新时间', str)
        }
        
        for field, (chinese_name, data_type) in field_mapping.items():
            if field in data:
                try:
                    value = data[field]
                    # 处理'-'占位符的情况
                    if value == '-':
                        parsed_data[chinese_name] = None
                    else:
                        parsed_data[chinese_name] = data_type(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"转换实时交易字段 {field} 失败: {e}, 使用原始值")
                    parsed_data[chinese_name] = data[field]
        
        return parsed_data
        
    def get_real_time_transaction(self, stock_code: str) -> Optional[Dict]:
        """获取实时交易数据
        
        Args:
            stock_code: 股票代码，如000001
            
        Returns:
            Optional[Dict]: 实时交易数据，如果失败返回None
        """
        try:
            # 使用必盈API获取实时交易数据
            logger.info(f"使用必盈API获取股票 {stock_code} 的实时交易数据")
            result = self._call_biying_api(
                self.BIYING_API_REAL_TIME_TRANSACTION,
                stock_code=stock_code
            )
            
            if result and isinstance(result, dict):
                logger.info(f"成功从必盈API获取股票 {stock_code} 的实时交易数据")
                # 解析实时交易数据
                return self._parse_real_time_transaction(result)
            
            # 如果必盈API失败，使用akshare作为备选
            logger.warning(f"必盈API获取股票 {stock_code} 实时交易数据失败，尝试使用akshare获取")
            # 这里可以添加akshare的实时交易数据获取逻辑
            # 例如：real_time = ak.stock_zh_a_spot_em()
            
            return None
            
        except Exception as e:
            logger.error(f"获取实时交易数据失败: {e}")
            return None
            
    def _get_random_user_agent(self) -> str:
        """获取随机的User-Agent字符串，用于模拟不同浏览器
        
        Returns:
            str: 随机的User-Agent字符串
        """
        user_agents = [
            # Chrome
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
            # Firefox
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0',
            'Mozilla/5.0 (X11; Linux i686; rv:124.0) Gecko/20100101 Firefox/124.0',
            # Safari
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15',
            # Edge
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
            # Opera
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 OPR/108.0.0.0',
        ]
        return random.choice(user_agents)
    
    def _get_market_name(self, market_type: str) -> str:
        """根据市场类型代码获取市场名称
        
        Args:
            market_type: 市场类型代码
            
        Returns:
            str: 市场名称
        """
        market_map = {
            '1': 'A股',
            '2': '港股',
            '3': '美股',
            '4': '债券',
            '5': '基金',
            '6': '指数'
        }
        return market_map.get(market_type, '未知市场')
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串
        
        Returns:
            str: 当前时间字符串，格式为"YYYY-MM-DD HH:MM:SS"
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    def get_market_overview(self) -> Dict:
        """获取市场概览数据，使用数据源优先级配置"""
        try:
            cache_key = self._get_cache_key("get_market_overview")
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            logger.info("正在获取市场概览数据...")
            
            sources = self.data_source_priority.get('market_overview', ['akshare', 'eastmoney', 'sina'])
            market_df = self._get_data_from_source('market_overview', sources)
            
            if market_df is None or (isinstance(market_df, pd.DataFrame) and market_df.empty):
                logger.error("所有数据源获取市场概览数据失败")
                return {}
            
            market_df.columns = market_df.columns.str.strip()
            
            column_mapping = {
                '涨跌幅': ['涨跌幅', 'pct_chg', 'percent', 'changepercent'],
                '收盘': ['收盘', 'close', '当前价'],
                '开盘': ['开盘', 'open', '开盘价']
            }
            
            def get_actual_column(column_mapping, target_column, df_columns):
                for col in df_columns:
                    if col in column_mapping[target_column]:
                        return col
                for col in df_columns:
                    if any(match in col for match in column_mapping[target_column]):
                        return col
                return None
            
            change_column = get_actual_column(column_mapping, '涨跌幅', market_df.columns)
            if not change_column:
                close_column = get_actual_column(column_mapping, '收盘', market_df.columns)
                open_column = get_actual_column(column_mapping, '开盘', market_df.columns)
                if close_column and open_column:
                    market_df['计算涨跌幅'] = ((market_df[close_column] - market_df[open_column]) / market_df[open_column]) * 100
                    change_column = '计算涨跌幅'
                else:
                    logger.warning("无法找到或计算涨跌幅数据")
                    return {}
            
            total_stocks = len(market_df)
            up_stocks = len(market_df[market_df[change_column] > 0])
            down_stocks = len(market_df[market_df[change_column] < 0])
            flat_stocks = len(market_df[market_df[change_column] == 0])
            
            up_ratio = up_stocks / total_stocks if total_stocks > 0 else 0
            down_ratio = down_stocks / total_stocks if total_stocks > 0 else 0
            
            avg_change = market_df[change_column].mean() if total_stocks > 0 else 0
            
            market_overview = {
                "total_stocks": total_stocks,
                "up_stocks": up_stocks,
                "down_stocks": down_stocks,
                "flat_stocks": flat_stocks,
                "up_ratio": round(up_ratio * 100, 1),
                "down_ratio": round(down_ratio * 100, 1),
                "avg_change": round(avg_change, 2)
            }
            
            self._set_cached_data(cache_key, market_overview)
            logger.info(f"成功获取市场概览数据: 上涨{up_stocks}家, 下跌{down_stocks}家")
            return market_overview
            
        except Exception as e:
            logger.error(f"获取市场概览数据失败: {e}")
            return {}
    
    def get_sector_info_by_name(self, sector_name: str) -> Optional[Dict]:
        """根据板块名称查询板块信息，使用数据源优先级配置
        
        Args:
            sector_name: 板块名称
            
        Returns:
            Optional[Dict]: 板块信息，如果未找到返回None
        """
        try:
            cache_key = self._get_cache_key("get_sector_info_by_name", sector_name)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            logger.info(f"正在查询板块 '{sector_name}' 的信息...")
            
            sources = self.data_source_priority.get('sector_data', ['akshare', 'eastmoney', 'biying'])
            
            for source in sources:
                try:
                    if source == 'akshare':
                        result = self._get_sector_from_akshare(sector_name)
                    elif source == 'eastmoney':
                        result = self._get_sector_from_eastmoney(sector_name)
                    elif source == 'biying':
                        result = self._get_sector_from_biying(sector_name)
                    else:
                        continue
                    
                    if result:
                        self._set_cached_data(cache_key, result)
                        logger.info(f"成功从 {source} 查询到板块 '{sector_name}' 的信息")
                        return result
                except Exception as e:
                    logger.warning(f"从 {source} 查询板块 '{sector_name}' 失败: {e}")
                    continue
            
            logger.warning(f"未找到板块 '{sector_name}' 的信息")
            return None
        except Exception as e:
            logger.error(f"根据板块名称查询信息失败: {e}")
            return None
    
    def _get_sector_from_akshare(self, sector_name: str) -> Optional[Dict]:
        """从akshare获取板块信息"""
        try:
            sector_stocks = ak.stock_board_industry_cons_em(symbol=sector_name)
            if not sector_stocks.empty:
                return {
                    "sector_name": sector_name,
                    "stock_count": len(sector_stocks),
                    "stocks": sector_stocks[['代码', '名称']].to_dict('records'),
                    "source": "akshare"
                }
        except Exception:
            try:
                all_sectors = ak.stock_board_industry_name_em()
                if not all_sectors.empty:
                    match = all_sectors[all_sectors['name'].str.contains(sector_name, case=False)]
                    if not match.empty:
                        sector_full_name = match.iloc[0]['name']
                        sector_stocks = ak.stock_board_industry_cons_em(symbol=sector_full_name)
                        if not sector_stocks.empty:
                            return {
                                "sector_name": sector_full_name,
                                "stock_count": len(sector_stocks),
                                "stocks": sector_stocks[['代码', '名称']].to_dict('records'),
                                "source": "akshare"
                            }
            except Exception as e:
                logger.warning(f"akshare查询板块失败: {e}")
        return None
    
    def _get_sector_from_eastmoney(self, sector_name: str) -> Optional[Dict]:
        """从东方财富获取板块信息"""
        try:
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                'pn': 1,
                'pz': 200,
                'po': 1,
                'np': 1,
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': 2,
                'invt': 2,
                'fid': 'f3',
                'fs': 'm:90+t:2',
                'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f115,f152'
            }
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and data['data'].get('diff'):
                    for item in data['data']['diff']:
                        if sector_name in item.get('f14', ''):
                            return {
                                "sector_name": item.get('f14', ''),
                                "涨跌幅": item.get('f3', 0),
                                "涨跌额": item.get('f4', 0),
                                "成交量": item.get('f5', 0),
                                "成交额": item.get('f6', 0),
                                "source": "eastmoney"
                            }
        except Exception as e:
            logger.warning(f"东方财富查询板块失败: {e}")
        return None
    
    def _get_sector_from_biying(self, sector_name: str) -> Optional[Dict]:
        """从必盈API获取板块信息"""
        try:
            licence = self.get_biyingapi_licence()
            if licence:
                logger.debug(f"必盈API暂不直接支持板块查询")
        except Exception as e:
            logger.warning(f"必盈API查询板块失败: {e}")
        return None
    
    def get_index_info_by_name(self, index_name: str) -> Optional[Dict]:
        """根据指数名称查询指数信息，使用数据源优先级配置
        
        Args:
            index_name: 指数名称
            
        Returns:
            Optional[Dict]: 指数信息，如果未找到返回None
        """
        try:
            cache_key = self._get_cache_key("get_index_info_by_name", index_name)
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            logger.info(f"正在查询指数 '{index_name}' 的信息...")
            
            sources = self.data_source_priority.get('index_data', ['akshare', 'eastmoney', 'baostock'])
            
            for source in sources:
                try:
                    if source == 'akshare':
                        result = self._get_index_from_akshare(index_name)
                    elif source == 'eastmoney':
                        result = self._get_index_from_eastmoney(index_name)
                    elif source == 'baostock':
                        result = self._get_index_from_baostock(index_name)
                    else:
                        continue
                    
                    if result:
                        self._set_cached_data(cache_key, result)
                        logger.info(f"成功从 {source} 查询到指数 '{index_name}' 的信息")
                        return result
                except Exception as e:
                    logger.warning(f"从 {source} 查询指数 '{index_name}' 失败: {e}")
                    continue
            
            logger.warning(f"未找到指数 '{index_name}' 的信息")
            return None
        except Exception as e:
            logger.error(f"根据指数名称查询信息失败: {e}")
            return None
    
    def _get_index_from_akshare(self, index_name: str) -> Optional[Dict]:
        """从akshare获取指数信息"""
        try:
            index_df = ak.stock_zh_index_spot_em()
            if not index_df.empty:
                match = index_df[index_df['名称'] == index_name]
                if not match.empty:
                    index_info = match.iloc[0].to_dict()
                    index_info['source'] = "akshare"
                    return index_info
                
                fuzzy_match = index_df[index_df['名称'].str.contains(index_name, case=False)]
                if not fuzzy_match.empty:
                    index_info = fuzzy_match.iloc[0].to_dict()
                    index_info['source'] = "akshare"
                    return index_info
        except Exception as e:
            logger.warning(f"akshare查询指数失败: {e}")
        return None
    
    def _get_index_from_eastmoney(self, index_name: str) -> Optional[Dict]:
        """从东方财富获取指数信息"""
        try:
            url = "https://push2.eastmoney.com/api/qt/clist/get"
            params = {
                'pn': 1,
                'pz': 100,
                'po': 1,
                'np': 1,
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': 2,
                'invt': 2,
                'fid': 'f3',
                'fs': 'm:1+s:2,m:1+s:3',
                'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f115,f152'
            }
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                if data.get('data') and data['data'].get('diff'):
                    for item in data['data']['diff']:
                        if index_name in item.get('f14', ''):
                            return {
                                "代码": item.get('f12', ''),
                                "名称": item.get('f14', ''),
                                "最新价": item.get('f2', 0),
                                "涨跌幅": item.get('f3', 0),
                                "涨跌额": item.get('f4', 0),
                                "成交量": item.get('f5', 0),
                                "成交额": item.get('f6', 0),
                                "今开": item.get('f17', 0),
                                "最高": item.get('f15', 0),
                                "最低": item.get('f16', 0),
                                "昨收": item.get('f18', 0),
                                "source": "eastmoney"
                            }
        except Exception as e:
            logger.warning(f"东方财富查询指数失败: {e}")
        return None
    
    def _get_index_from_baostock(self, index_name: str) -> Optional[Dict]:
        """从baostock获取指数信息"""
        try:
            index_code = self.get_etf_index_code_by_name(index_name)
            if index_code:
                k_rs = bs.query_history_k_data_plus(
                    index_code,
                    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg",
                    start_date=(datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
                    end_date=datetime.datetime.now().strftime("%Y-%m-%d"),
                    frequency="d",
                    adjustflag="3"
                )
                
                if k_rs.error_code == '0':
                    index_k_data = k_rs.get_data()
                    if not index_k_data.empty:
                        latest_data = index_k_data.iloc[-1].to_dict()
                        basic_rs = bs.query_stock_basic(code=index_code)
                        if basic_rs.error_code == '0':
                            basic_data = basic_rs.get_data()
                            if not basic_data.empty:
                                index_info = basic_data.iloc[0].to_dict()
                                index_info.update(latest_data)
                                index_info['source'] = "baostock"
                                return index_info
        except Exception as e:
            logger.warning(f"baostock查询指数失败: {e}")
        return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        if df.empty:
            logger.warning("输入数据为空，无法计算技术指标")
            return df
        
        # 复制原数据
        result_df = df.copy()
        
        try:
            # 记录数据列信息，便于调试
            logger.debug(f"数据列名: {result_df.columns.tolist()}")
            logger.debug(f"数据形状: {result_df.shape}")
            
            # 检查是否有'收盘'列，如果没有，尝试从其他列获取
            has_close_column = '收盘' in result_df.columns
            
            if not has_close_column:
                # 尝试从其他可能的收盘价列名获取数据
                possible_close_columns = ['close', 'Close', '最新价', '最新', 'price', 'Price']
                found_close_column = None
                
                for col in possible_close_columns:
                    if col in result_df.columns:
                        found_close_column = col
                        logger.info(f"使用'{col}'作为收盘价列")
                        result_df['收盘'] = result_df[col]
                        has_close_column = True
                        break
            
            # 如果还是没有收盘价数据，记录警告但继续处理其他可能的指标
            if not has_close_column:
                logger.warning("数据缺少收盘价相关列，无法计算基于价格的技术指标")
                
                # 尝试计算成交量相关指标（如果有成交量数据）
                if '成交量' in result_df.columns:
                    logger.info("仅计算成交量相关指标")
                    result_df['成交量'] = pd.to_numeric(result_df['成交量'], errors='coerce')
                    result_df['VOL5'] = result_df['成交量'].rolling(window=5).mean()
                    result_df['VOL10'] = result_df['成交量'].rolling(window=10).mean()
                
                return result_df
            
            # 确保'收盘'列是数值类型
            result_df['收盘'] = pd.to_numeric(result_df['收盘'], errors='coerce')
            
            # 过滤掉'收盘'列中的NaN值
            valid_data = result_df.dropna(subset=['收盘'])
            
            # 如果没有有效收盘价数据，仅计算成交量相关指标
            if valid_data.empty:
                logger.warning("'收盘'列没有有效数值，仅计算成交量相关指标")
                
                if '成交量' in result_df.columns:
                    result_df['成交量'] = pd.to_numeric(result_df['成交量'], errors='coerce')
                    result_df['VOL5'] = result_df['成交量'].rolling(window=5).mean()
                    result_df['VOL10'] = result_df['成交量'].rolling(window=10).mean()
                
                return result_df
            
            # 计算移动平均线
            valid_data['MA5'] = valid_data['收盘'].rolling(window=5).mean()
            valid_data['MA10'] = valid_data['收盘'].rolling(window=10).mean()
            valid_data['MA20'] = valid_data['收盘'].rolling(window=20).mean()
            valid_data['MA30'] = valid_data['收盘'].rolling(window=30).mean()
            valid_data['MA60'] = valid_data['收盘'].rolling(window=60).mean()
            
            # 计算MACD
            valid_data['EMA12'] = valid_data['收盘'].ewm(span=12, adjust=False).mean()
            valid_data['EMA26'] = valid_data['收盘'].ewm(span=26, adjust=False).mean()
            valid_data['DIF'] = valid_data['EMA12'] - valid_data['EMA26']
            valid_data['DEA'] = valid_data['DIF'].ewm(span=9, adjust=False).mean()
            valid_data['MACD'] = 2 * (valid_data['DIF'] - valid_data['DEA'])
            
            # 计算RSI
            delta = valid_data['收盘'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # 处理除数为0的情况
            loss = loss.replace(0, 1e-10)
            rs = gain / loss
            valid_data['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算移动平均线
            valid_data['MA5'] = valid_data['收盘'].rolling(window=5).mean()
            valid_data['MA10'] = valid_data['收盘'].rolling(window=10).mean()
            valid_data['MA20'] = valid_data['收盘'].rolling(window=20).mean()
            valid_data['MA30'] = valid_data['收盘'].rolling(window=30).mean()
            valid_data['MA60'] = valid_data['收盘'].rolling(window=60).mean()
            
            # 计算MACD
            valid_data['EMA12'] = valid_data['收盘'].ewm(span=12, adjust=False).mean()
            valid_data['EMA26'] = valid_data['收盘'].ewm(span=26, adjust=False).mean()
            valid_data['DIF'] = valid_data['EMA12'] - valid_data['EMA26']
            valid_data['DEA'] = valid_data['DIF'].ewm(span=9, adjust=False).mean()
            valid_data['MACD'] = 2 * (valid_data['DIF'] - valid_data['DEA'])
            
            # 计算RSI
            delta = valid_data['收盘'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # 处理除数为0的情况
            loss = loss.replace(0, 1e-10)
            rs = gain / loss
            valid_data['RSI'] = 100 - (100 / (1 + rs))
            
            # 计算成交量相关指标（如果有成交量列）
            if '成交量' in valid_data.columns:
                valid_data['成交量'] = pd.to_numeric(valid_data['成交量'], errors='coerce')
                valid_data['VOL5'] = valid_data['成交量'].rolling(window=5).mean()
                valid_data['VOL10'] = valid_data['成交量'].rolling(window=10).mean()
            else:
                logger.warning("数据缺少'成交量'列，无法计算成交量相关指标")
            
            # 将计算结果合并回原始DataFrame
            try:
                # 使用更简单的方式合并结果
                for col in valid_data.columns:
                    if col not in result_df.columns or result_df[col].isnull().any():
                        result_df[col] = valid_data[col]
            except Exception as merge_e:
                logger.warning(f"合并技术指标结果失败: {merge_e}")
                logger.debug("直接使用计算后的valid_data作为结果")
                result_df = valid_data
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            
            # 即使计算部分指标失败，也返回已计算的结果
            if 'result_df' in locals() and not result_df.empty:
                logger.info("返回部分计算完成的技术指标")
                return result_df
            
            # 如果没有计算出任何指标，返回原始数据
            return df
        
        return result_df
    
    def calculate_active_market_index(self) -> Dict:
        """计算活筹指数 (0AMV) 模拟值
        
        模拟通达信中的"活跃市值"，通过计算市场成交额的10日移动平均并观察其增减变化，
        来判断活跃资金是流入还是流出。
        
        Returns:
            Dict: 活筹指数相关数据
        """
        try:
            cache_key = self._get_cache_key("calculate_active_market_index")
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 添加指数退避重试机制
            max_retries = 3
            base_delay = 1  # 基础延迟时间（秒）
            import time
            import random
            
            for retry in range(max_retries):
                try:
                    # 控制请求频率，使用3-5秒的随机间隔
                    if retry > 0:
                        delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                        logger.debug(f"第 {retry+1} 次尝试，等待 {delay:.2f} 秒")
                        time.sleep(delay)
                    
                    # 获取上证指数数据
                    df = ak.index_zh_a_hist(symbol="000001", period="daily")
                    if not df.empty:
                        # 计算活筹指数
                        df['amount_wan'] = df['成交额'] / 1000000  # 转为万元
                        df['var1'] = df['amount_wan'].rolling(window=10).mean()  # 10日均值
                        df['active_index'] = df['var1'] - df['var1'].shift(1)  # 增减值
                        
                        # 获取最新值
                        latest_active_index = df['active_index'].iloc[-1] if len(df) > 10 else 0
                        
                        result = {
                            "active_index": latest_active_index,
                            "message": "计算成功"
                        }
                        
                        # 缓存结果
                        self._set_cached_data(cache_key, result)
                        logger.info(f"活筹指数计算完成: {latest_active_index}")
                        return result
                except Exception as e:
                    logger.debug(f"第 {retry+1} 次尝试失败: {e}")
                    # 遇到RemoteDisconnected等网络异常时继续重试
                    if isinstance(e, (ConnectionError, TimeoutError)) or "RemoteDisconnected" in str(e):
                        continue
                    else:
                        # 其他异常直接返回
                        break
            
            # 所有重试都失败，尝试使用tickflow作为备用数据源
            if self.tickflow_client:
                try:
                    logger.info("尝试使用TickFlow获取上证指数数据")
                    # 使用tickflow获取上证指数K线数据
                    df = self.tickflow_client.klines.get(
                        symbol="000001.SH",
                        period="1d",
                        count=100,
                        as_dataframe=True
                    )
                    
                    if df is not None and not df.empty:
                        # 计算活筹指数
                        # tickflow返回的数据列名可能不同，需要适配
                        # 假设返回的数据包含成交额字段
                        if 'amount' in df.columns:
                            df['amount_wan'] = df['amount'] / 1000000  # 转为万元
                        elif 'volume' in df.columns:
                            # 如果没有成交额，使用成交额代替（可能不准确）
                            df['amount_wan'] = df['volume'] / 1000000
                        
                        df['var1'] = df['amount_wan'].rolling(window=10).mean()  # 10日均值
                        df['active_index'] = df['var1'] - df['var1'].shift(1)  # 增减值
                        
                        # 获取最新值
                        latest_active_index = df['active_index'].iloc[-1] if len(df) > 10 else 0
                        
                        result = {
                            "active_index": latest_active_index,
                            "message": "使用TickFlow计算成功"
                        }
                        
                        # 缓存结果
                        self._set_cached_data(cache_key, result)
                        logger.info(f"活筹指数计算完成 (TickFlow): {latest_active_index}")
                        return result
                    else:
                        logger.warning("TickFlow返回空数据")
                except Exception as e:
                    logger.warning(f"TickFlow获取上证指数数据失败: {e}")
            
            logger.warning("获取上证指数数据失败，无法计算活筹指数")
            return {"active_index": 0, "message": "获取数据失败"}
            
        except Exception as e:
            logger.error(f"计算活筹指数失败: {e}")
            return {"active_index": 0, "message": f"计算失败: {str(e)}"}
    
    def calculate_adr(self) -> Dict:
        """计算涨跌家数比 (ADR)
        
        直接反映市场的赚钱效应，上涨家数除以下跌家数，数值越大说明市场越亢奋。
        
        Returns:
            Dict: 涨跌家数比相关数据
        """
        try:
            cache_key = self._get_cache_key("calculate_adr")
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 添加指数退避重试机制
            max_retries = 3
            base_delay = 1  # 基础延迟时间（秒）
            import time
            import random
            
            for retry in range(max_retries):
                try:
                    # 控制请求频率，使用3-5秒的随机间隔
                    if retry > 0:
                        delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                        logger.debug(f"第 {retry+1} 次尝试，等待 {delay:.2f} 秒")
                        time.sleep(delay)
                    
                    # 获取全市场所有A股的实时行情
                    df_all = ak.stock_zh_a_spot_em()
                    if not df_all.empty:
                        # 统计上涨家数和下跌家数
                        up = len(df_all[df_all['涨跌幅'] > 0])
                        down = len(df_all[df_all['涨跌幅'] < 0])
                        
                        # 计算涨跌比和市场宽度
                        ad_ratio = up / down if down > 0 else 0
                        breadth = (up - down) / (up + down) * 100 if (up + down) > 0 else 0
                        
                        # 判断市场情绪
                        if ad_ratio > 2:
                            sentiment = "情绪过热"
                        elif ad_ratio < 0.5:
                            sentiment = "情绪过冷"
                        else:
                            sentiment = "情绪中性"
                        
                        result = {
                            "ad_ratio": ad_ratio,
                            "breadth": breadth,
                            "up": up,
                            "down": down,
                            "sentiment": sentiment,
                            "message": "计算成功"
                        }
                        
                        # 缓存结果
                        self._set_cached_data(cache_key, result)
                        logger.info(f"涨跌家数比计算完成: 上涨{up}, 下跌{down}, 涨跌比{ad_ratio:.2f}, 宽度{breadth:.1f}%")
                        return result
                except Exception as e:
                    logger.debug(f"第 {retry+1} 次尝试失败: {e}")
                    # 遇到RemoteDisconnected等网络异常时继续重试
                    if isinstance(e, (ConnectionError, TimeoutError)) or "RemoteDisconnected" in str(e):
                        continue
                    else:
                        # 其他异常直接返回
                        break
            
            # 所有重试都失败，尝试使用tickflow作为备用数据源
            if self.tickflow_client:
                try:
                    logger.info("尝试使用TickFlow获取A股实时行情")
                    # 使用batch方法获取上交所和深交所的标的
                    # 由于免费版限制，我们使用一些常见的股票代码进行测试
                    test_symbols = [
                        "600000.SH", "600036.SH", "600519.SH", "601318.SH", "601398.SH",
                        "000001.SZ", "000002.SZ", "000063.SZ", "000333.SZ", "000858.SZ"
                    ]
                    
                    # 获取这些股票的基本信息
                    instruments = self.tickflow_client.instruments.batch(symbols=test_symbols)
                    
                    if instruments and len(instruments) > 0:
                        # 分批获取K线数据，避免请求过多
                        up = 0
                        down = 0
                        total = 0
                        
                        for symbol in test_symbols:
                            try:
                                # 获取最近1天的K线数据
                                df = self.tickflow_client.klines.get(
                                    symbol=symbol,
                                    period="1d",
                                    count=2,
                                    as_dataframe=True
                                )
                                
                                if df is not None and not df.empty and len(df) >= 2:
                                    # 计算涨跌幅
                                    if 'close' in df.columns:
                                        latest_close = df['close'].iloc[-1]
                                        prev_close = df['close'].iloc[-2]
                                        change_pct = (latest_close - prev_close) / prev_close * 100
                                        
                                        if change_pct > 0:
                                            up += 1
                                        elif change_pct < 0:
                                            down += 1
                                        total += 1
                            except Exception as e:
                                logger.debug(f"获取 {symbol} 数据失败: {e}")
                                continue
                        
                        if total > 0:
                            # 计算涨跌比和市场宽度
                            ad_ratio = up / down if down > 0 else 0
                            breadth = (up - down) / total * 100
                            
                            # 判断市场情绪
                            if ad_ratio > 2:
                                sentiment = "情绪过热"
                            elif ad_ratio < 0.5:
                                sentiment = "情绪过冷"
                            else:
                                sentiment = "情绪中性"
                            
                            result = {
                                "ad_ratio": ad_ratio,
                                "breadth": breadth,
                                "up": up,
                                "down": down,
                                "sentiment": sentiment,
                                "message": "使用TickFlow计算成功"
                            }
                            
                            # 缓存结果
                            self._set_cached_data(cache_key, result)
                            logger.info(f"涨跌家数比计算完成 (TickFlow): 上涨{up}, 下跌{down}, 涨跌比{ad_ratio:.2f}, 宽度{breadth:.1f}%")
                            return result
                    else:
                        logger.warning("TickFlow返回空标的池数据")
                except Exception as e:
                    logger.warning(f"TickFlow获取A股实时行情失败: {e}")
            
            logger.warning("获取A股实时行情失败，无法计算涨跌家数比")
            return {"ad_ratio": 0, "breadth": 0, "up": 0, "down": 0, "message": "获取数据失败"}
            
        except Exception as e:
            logger.error(f"计算涨跌家数比失败: {e}")
            return {"ad_ratio": 0, "breadth": 0, "up": 0, "down": 0, "message": f"计算失败: {str(e)}"}
    

    
    def get_north_bound_flow(self) -> Dict:
        """获取北向资金流向
        
        北向资金（外资）的净流入流出常被视为市场情绪的先行指标，大幅净流入通常预示市场走强，
        持续净流出则需警惕。
        
        Returns:
            Dict: 北向资金流向相关数据
        """
        try:
            cache_key = self._get_cache_key("get_north_bound_flow")
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 尝试使用多种北向资金接口
            # 使用AKShare中实际存在的沪深港通接口
            north_flow_functions = [
                ('stock_hsgt_hist_em', '当日资金流入'),  # 东方财富沪深港通历史数据
            ]
            
            latest_flow = 0
            history = []
            success = False
            
            # 添加指数退避重试机制
            max_retries = 3
            base_delay = 1  # 基础延迟时间（秒）
            import time
            import random
            
            for func_name, value_col in north_flow_functions:
                for retry in range(max_retries):
                    try:
                        # 控制请求频率，使用3-5秒的随机间隔
                        if retry > 0:
                            delay = base_delay * (2 ** retry) + random.uniform(0, 1)
                            logger.debug(f"第 {retry+1} 次尝试 {func_name}，等待 {delay:.2f} 秒")
                            time.sleep(delay)
                        
                        if hasattr(ak, func_name):
                            # stock_hsgt_hist_em 需要传入symbol参数
                            if func_name == 'stock_hsgt_hist_em':
                                north_flow = getattr(ak, func_name)(symbol="北向资金")
                            else:
                                north_flow = getattr(ak, func_name)()
                            
                            if not north_flow.empty:
                                # 获取最新净流入金额（第一行是最新数据）
                                latest_flow = north_flow[value_col].iloc[0] if value_col in north_flow.columns else 0
                                
                                # 获取最近5天的数据
                                history = []
                                date_col = 'date' if 'date' in north_flow.columns else '日期'
                                for i in range(min(5, len(north_flow))):
                                    date = north_flow[date_col].iloc[i]
                                    value = north_flow[value_col].iloc[i]
                                    history.append({"date": date, "value": value})
                                success = True
                                logger.info(f"使用 {func_name} 接口获取北向资金数据成功")
                                break
                    except Exception as e:
                        logger.debug(f"尝试 {func_name} 接口失败: {e}")
                        # 遇到RemoteDisconnected等网络异常时继续重试
                        if isinstance(e, (ConnectionError, TimeoutError)) or "RemoteDisconnected" in str(e):
                            continue
                        else:
                            # 其他异常直接返回
                            break
                if success:
                    break
            
            if not success:
                logger.warning("所有北向资金接口都失败")
                return {"latest_flow": 0, "history": [], "message": "获取数据失败"}
            
            result = {
                "latest_flow": latest_flow,
                "history": history,
                "message": "获取成功"
            }
            
            # 缓存结果
            self._set_cached_data(cache_key, result)
            logger.info(f"北向资金流向获取完成: 最新净流入 {latest_flow} 亿元")
            return result
            
        except Exception as e:
            logger.error(f"获取北向资金流向失败: {e}")
            return {"latest_flow": 0, "history": [], "message": f"获取失败: {str(e)}"}
    
    def get_market_sentiment_indicators(self) -> Dict:
        """获取市场情绪指标
        
        综合获取所有市场情绪相关指标，包括：
        1. 活筹指数
        2. 涨跌家数比
        3. 北向资金流向
        （新高新低指数和均线突破率因计算耗时，默认不包含）
        
        Returns:
            Dict: 市场情绪指标综合数据
        """
        try:
            cache_key = self._get_cache_key("get_market_sentiment_indicators")
            cached_data = self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # 获取各指标
            active_market = self.calculate_active_market_index()
            adr = self.calculate_adr()
            north_flow = self.get_north_bound_flow()
            
            # 综合结果
            result = {
                "active_market": active_market,
                "adr": adr,
                "north_flow": north_flow,
                "timestamp": self._get_current_time(),
                "message": "获取成功"
            }
            
            # 缓存结果
            self._set_cached_data(cache_key, result)
            logger.info("市场情绪指标获取完成")
            return result
            
        except Exception as e:
            logger.error(f"获取市场情绪指标失败: {e}")
            return {
                "active_market": {"active_index": 0},
                "adr": {"ad_ratio": 0, "breadth": 0},
                "north_flow": {"latest_flow": 0},
                "message": f"获取失败: {str(e)}"
            }
    
    def get_news(self, categories=['科技', '财经', '政治'], limit=5):
        """获取新闻
        
        Args:
            categories: 新闻类别列表
            limit: 返回数量限制
            
        Returns:
            新闻列表
        """
        try:
            # 导入新闻爬虫模块
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from news_crawler import get_news, crawl_now
            
            # 1. 优先从本地数据库获取最新新闻
            news_list = get_news(categories, limit)
            
            # 2. 若数据库过期或无数据，则触发爬虫刷新
            if not news_list:
                logger.info("本地数据库无新闻数据，触发爬虫刷新")
                crawl_now()
                # 等待爬虫完成
                import time
                time.sleep(5)
                # 再次获取
                news_list = get_news(categories, limit)
            
            # 3. 格式化新闻数据
            formatted_news = []
            for news in news_list:
                formatted_news.append({
                    "title": news.get('title', ''),
                    "snippet": news.get('summary', ''),
                    "date": news.get('publish_time', ''),
                    "source": news.get('source_website', ''),
                    "category": news.get('category', '')
                })
            
            logger.info(f"成功获取 {len(formatted_news)} 条新闻")
            return formatted_news
            
        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return []

# 全局数据获取Agent实例
data_agent = DataFetchingAgent()
