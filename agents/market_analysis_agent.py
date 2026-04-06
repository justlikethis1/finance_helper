# -*- coding: utf-8 -*-
import asyncio
import time
import datetime
import os
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
import requests
import json
import http.client
import urllib.parse

from agents.data_agent import data_agent
from backend.model import model_manager
from .data_processor import DataProcessor

# 导入自定义日志配置
from backend.logging_config import get_logger

# 获取日志记录器
logger = get_logger("agents.market_analysis")

class MarketAnalysisAgent:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = 3600  # 缓存1小时
        self.search_api_key = os.getenv("TIANAPI_KEY", "")  # 从环境变量获取API密钥
        # 初始化数据处理器
        self.data_processor = DataProcessor()
        # 初始化磁盘缓存
        self.cache_dir = "cache/market_analysis"
        os.makedirs(self.cache_dir, exist_ok=True)
        # 创建静态数据目录
        self.static_data_dir = "static/market_data"
        os.makedirs(self.static_data_dir, exist_ok=True)
        # 加载磁盘缓存
        self._load_disk_cache()
        # 启动定时预计算和缓存清理任务
        self._start_scheduled_tasks()
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """生成缓存键"""
        return f"{func_name}:{args}:{kwargs}"
    
    def _get_cached_result(self, key: str) -> Optional[Dict]:
        """获取缓存结果"""
        if key in self.cache:
            cached_time, result = self.cache[key]
            if time.time() - cached_time < self.cache_expiry:
                return result
        return None
    
    def _load_disk_cache(self) -> None:
        """从磁盘加载缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, "market_analysis_cache.json")
            if os.path.exists(cache_file):
                with open(cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
                logger.info(f"成功从磁盘加载缓存，共 {len(self.cache)} 个条目")
        except Exception as e:
            logger.error(f"从磁盘加载缓存失败: {e}")
            self.cache = {}
    
    def _save_disk_cache(self) -> None:
        """将缓存保存到磁盘"""
        try:
            cache_file = os.path.join(self.cache_dir, "market_analysis_cache.json")
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"成功将缓存保存到磁盘，共 {len(self.cache)} 个条目")
        except Exception as e:
            logger.error(f"将缓存保存到磁盘失败: {e}")
    
    def _start_scheduled_tasks(self) -> None:
        """启动定时任务"""
        import threading
        import time
        
        # 启动定时预计算任务（每小时执行一次）
        def precompute_task():
            logger.info("启动市场分析定时预计算任务")
            while True:
                try:
                    # 获取事件循环
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    # 执行预计算
                    loop.run_until_complete(self._precompute_market_data())
                    loop.close()
                except Exception as e:
                    logger.error(f"定时预计算任务失败: {e}")
                
                # 每小时执行一次
                time.sleep(3600)
        
        # 启动缓存清理任务（每天执行一次）
        def cleanup_task():
            logger.info("启动缓存清理定时任务")
            while True:
                try:
                    self._cleanup_cache()
                except Exception as e:
                    logger.error(f"定时缓存清理任务失败: {e}")
                
                # 每天执行一次
                time.sleep(86400)
        
        # 启动定时任务线程
        precompute_thread = threading.Thread(target=precompute_task, daemon=True)
        precompute_thread.start()
        
        cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        cleanup_thread.start()
    
    async def _precompute_market_data(self) -> None:
        """预计算市场数据并保存为静态文件"""
        logger.info("开始预计算市场数据...")
        
        try:
            # 获取基础数据
            index_data = data_agent.get_index_quotes(['sh000001', 'sz399001'])
            market_overview = data_agent.get_market_overview()
            industry_data = data_agent.get_industry_data()
            
            # 获取板块资金流向排名数据
            try:
                sector_fund_flow = data_agent.get_sector_fund_flow_rank()
                logger.info(f"预计算时获取到的板块资金流向数据形状: {sector_fund_flow.shape}")
                logger.info(f"资金流向数据列名: {list(sector_fund_flow.columns) if not sector_fund_flow.empty else []}")
                if not sector_fund_flow.empty:
                    logger.info(f"资金流向数据前5行: {sector_fund_flow.head().to_dict(orient='records')}")
            except Exception as e:
                logger.error(f"预计算时获取板块资金流向排名数据失败: {e}")
                sector_fund_flow = pd.DataFrame()
            
            # 提取资金流向排名前5的板块
            top_fund_flow_sectors = []
            if sector_fund_flow is not None and not sector_fund_flow.empty:
                for _, row in sector_fund_flow.head(5).iterrows():
                    sector_name = row.get('板块名称', row.get('name', row.get('名称', '未知板块')))
                    top_fund_flow_sectors.append(sector_name)
                logger.info(f"资金净流入排名前5的板块: {top_fund_flow_sectors}")
            
            # 搜索当日财经新闻
            today_date = datetime.datetime.now().strftime("%Y-%m-%d")
            try:
                # 获取市场整体新闻，传递资金流向排名前5的板块作为关键词
                general_news = await self.search_financial_news(f"{today_date} 财经新闻 A股市场", num_results=10, keywords=top_fund_flow_sectors)
                
                # 获取主要板块的相关新闻
                sector_news = []
                # 合并主要板块和资金流向排名前5的板块
                main_sectors = list(set(["科技", "新能源", "医药", "金融", "消费"] + top_fund_flow_sectors))
                for sector in main_sectors:
                    sector_news += await self.search_financial_news(f"{today_date} {sector} 板块 新闻", num_results=3)
                
                # 合并新闻并去重
                all_news = general_news + sector_news
                seen_titles = set()
                unique_news = []
                for news_item in all_news:
                    if news_item["title"] not in seen_titles:
                        seen_titles.add(news_item["title"])
                        unique_news.append(news_item)
                
                news = unique_news
                logger.info(f"预计算时共获取到 {len(news)} 条财经新闻")
            except Exception as e:
                logger.error(f"预计算时搜索财经新闻失败: {e}")
                news = []
            
            # 生成分析数据
            market_summary = self._generate_detailed_market_summary(index_data, market_overview)
            strong_sectors = self._get_strong_sectors(industry_data)
            rotation_analysis = self._generate_rotation_analysis(industry_data)
            tomorrow_prediction = self._generate_tomorrow_prediction(index_data, market_overview, industry_data, news, sector_fund_flow)
            
            # 构建预计算数据
            precomputed_data = {
                "market_summary": market_summary,
                "strong_sectors": strong_sectors,
                "rotation_analysis": rotation_analysis,
                "tomorrow_prediction": tomorrow_prediction,
                "market_data": {
                    "market_overview": market_overview,
                    "index_data": index_data,
                    "industry_performance": self._get_industry_performance(industry_data)
                },
                "timestamp": time.time()
            }
            
            # 保存为静态文件
            static_file = os.path.join(self.static_data_dir, "precomputed_market_data.json")
            with open(static_file, "w", encoding="utf-8") as f:
                json.dump(precomputed_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.info(f"市场数据预计算完成，已保存到 {static_file}")
        except Exception as e:
            logger.error(f"预计算市场数据失败: {e}")
    
    def get_precomputed_market_data(self, only_timestamp: bool = False) -> Optional[Dict]:
        """获取预计算的市场数据
        
        Args:
            only_timestamp: 是否只返回时间戳信息
        
        Returns:
            包含市场数据的字典或仅包含时间戳的字典，或者None
        """
        static_file = os.path.join(self.static_data_dir, "precomputed_market_data.json")
        try:
            if os.path.exists(static_file):
                # 获取文件的最后修改时间
                file_mtime = os.path.getmtime(static_file)
                
                if only_timestamp:
                    return {"timestamp": file_mtime}
                
                with open(static_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 使用文件修改时间作为数据时间戳
                data["timestamp"] = file_mtime
                
                # 检查数据是否过期（超过1小时）
                if time.time() - file_mtime < 3600:
                    return data
                else:
                    logger.warning("预计算数据已过期")
        except Exception as e:
            logger.error(f"获取预计算市场数据失败: {e}")
        
        return None
    
    def check_market_data_update(self, last_timestamp: float) -> Dict:
        """检查市场数据是否有更新
        
        Args:
            last_timestamp: 客户端最后一次更新的时间戳
        
        Returns:
            包含更新状态的字典
        """
        result = {
            "has_update": False,
            "timestamp": 0,
            "message": "数据未更新"
        }
        
        try:
            # 检查预计算数据
            precomputed_data = self.get_precomputed_market_data(only_timestamp=True)
            if precomputed_data:
                current_timestamp = precomputed_data["timestamp"]
                result["timestamp"] = current_timestamp
                
                if current_timestamp > last_timestamp:
                    result["has_update"] = True
                    result["message"] = "数据已更新"
                    return result
            
            # 检查内存缓存
            cache_key = self._get_cache_key("analyze_market")
            if cache_key in self.cache:
                cached_time, _ = self.cache[cache_key]
                result["timestamp"] = cached_time
                
                if cached_time > last_timestamp:
                    result["has_update"] = True
                    result["message"] = "缓存数据已更新"
                    return result
                    
        except Exception as e:
            logger.error(f"检查市场数据更新失败: {e}")
            result["message"] = f"检查更新失败: {str(e)}"
        
        return result
    
    def _is_data_valid(self, data: Dict) -> bool:
        """检查市场数据是否有效
        
        Args:
            data: 要检查的市场数据
        
        Returns:
            如果数据有效返回True，否则返回False
        """
        if not data:
            return False
            
        # 检查核心字段是否存在且有效
        required_fields = ["market_summary", "strong_sectors", "timestamp"]
        for field in required_fields:
            if field not in data or not data[field]:
                return False
        
        # 检查时间戳是否有效
        timestamp = data.get("timestamp", 0)
        if timestamp <= 0 or time.time() - timestamp > 86400:  # 超过24小时的数据认为无效
            return False
        
        return True
    
    def _cleanup_cache(self) -> None:
        """清理过期缓存和临时文件"""
        logger.info("开始清理缓存...")
        
        # 清理内存缓存中的过期条目
        now = time.time()
        expired_keys = []
        for key, (cached_time, _) in self.cache.items():
            if now - cached_time > self.cache_expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            logger.info(f"清理了 {len(expired_keys)} 个过期内存缓存条目")
        
        # 清理磁盘缓存文件
        try:
            cache_file = os.path.join(self.cache_dir, "market_analysis_cache.json")
            if os.path.exists(cache_file):
                # 重新保存清理后的缓存
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
                logger.info("已更新磁盘缓存文件")
        except Exception as e:
            logger.error(f"清理磁盘缓存失败: {e}")
        
        # 清理旧的静态文件
        try:
            static_file = os.path.join(self.static_data_dir, "precomputed_market_data.json")
            if os.path.exists(static_file):
                # 检查文件是否过期（超过24小时）
                file_mtime = os.path.getmtime(static_file)
                if time.time() - file_mtime > 86400:
                    os.remove(static_file)
                    logger.info("已清理过期的静态数据文件")
        except Exception as e:
            logger.error(f"清理静态文件失败: {e}")
        
        logger.info("缓存清理完成")
    
    def _cache_result(self, key: str, result: Dict) -> None:
        """缓存结果"""
        self.cache[key] = (time.time(), result)
        # 异步保存到磁盘，避免阻塞
        import threading
        threading.Thread(target=self._save_disk_cache).start()
    
    async def search_financial_news(self, query: str, num_results: int = 20, max_retries: int = 3, retry_delay: float = 2.0, keywords: List[str] = None, return_status: bool = False) -> Union[List[Dict], Tuple[List[Dict], Dict]]:
        """搜索财经新闻，支持重试机制和详细错误处理
        
        Args:
            query: 搜索关键词
            num_results: 返回结果数量，默认20
            max_retries: 最大重试次数
            retry_delay: 重试延迟
            keywords: 额外的关键词列表，如公司名或板块名
            return_status: 是否返回API调用状态信息
            
        Returns:
            如果return_status为False，返回新闻列表，每个新闻包含标题、摘要和日期
            如果return_status为True，返回元组(新闻列表, 状态信息字典)
        """
        all_news = []
        
        # 处理单个查询的函数
        async def fetch_news_for_query():
            # 检查缓存
            cache_key = self._get_cache_key("search_financial_news", "general_news", num_results)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("使用缓存的新闻搜索结果")
                return cached_result, {"code": 200, "msg": "使用缓存数据"}
                
            try:
                # 优先使用新闻爬虫获取新闻数据
                from agents.data_agent import data_agent
                
                # 获取财经类新闻
                news_list = data_agent.get_news(categories=['财经'], limit=20)
                
                if news_list:
                    logger.info(f"从新闻爬虫获取财经新闻成功，共 {len(news_list)} 条")
                    
                    # 格式化新闻数据
                    formatted_news = []
                    for news in news_list:
                        formatted_news.append({
                            "title": news.get('title', ''),
                            "snippet": news.get('summary', ''),
                            "date": news.get('publish_time', ''),
                            "source": news.get('source_website', ''),
                            "category": news.get('category', '')
                        })
                    
                    # 缓存API结果
                    self._cache_result(cache_key, formatted_news)
                    return formatted_news, {"code": 200, "msg": "从新闻爬虫获取数据成功"}
                else:
                    logger.warning("新闻爬虫未返回数据，尝试使用TianAPI作为备用")
            except Exception as e:
                logger.error(f"新闻爬虫调用失败: {e}")
                logger.info("尝试使用TianAPI作为备用")
            
            if not self.search_api_key or self.search_api_key == "demo_key":
                # 模拟搜索结果
                result = []
                self._cache_result(cache_key, result)
                return result, {"code": 200, "msg": "使用模拟数据"}
            
            # 重试机制
            status_info = {"code": 0, "msg": "初始化"}
            for retry in range(max_retries):
                try:
                    # 使用天聚数行TianAPI财经新闻接口
                    # 设置num参数为20，禁用word参数
                    num = 20
                    params = urllib.parse.urlencode({
                        'key': self.search_api_key,
                        'num': str(num),
                        'form': '1'  # 兼容历史问题，建议传1
                    })
                    headers = {'Content-type': 'application/x-www-form-urlencoded'}
                    
                    # 使用http.client调用API，符合官方示例
                    conn = http.client.HTTPSConnection('apis.tianapi.com')  # 接口域名
                    conn.request('POST', '/caijing/index', params, headers)
                    tianapi = conn.getresponse()
                    result = tianapi.read()
                    data = result.decode('utf-8')
                    dict_data = json.loads(data)
                    conn.close()
                    
                    status_info = {"code": dict_data.get("code"), "msg": dict_data.get("msg", "未知错误")}
                    logger.info(f"TianAPI返回完整数据: {dict_data}")
                    if dict_data.get("code") == 200 and dict_data.get("result"):
                        # 尝试获取list字段，如果不存在则尝试获取newslist字段
                        results = dict_data.get("result", {}).get("list", dict_data.get("result", {}).get("newslist", []))
                        logger.info(f"TianAPI返回新闻数量: {len(results)}")
                        news_list = []
                        
                        for item in results:
                            news_item = {
                                "title": item.get("title", ""),
                                "snippet": item.get("description", ""),
                                "date": item.get("ctime", ""),
                                "source": "TianAPI",
                                "category": "财经"
                            }
                            news_list.append(news_item)
                        
                        # 缓存API结果
                        self._cache_result(cache_key, news_list)
                        logger.info("从TianAPI获取财经新闻成功")
                        return news_list, status_info
                    else:
                        logger.warning(f"TianAPI返回错误: 代码={dict_data.get('code')}, 消息={dict_data.get('msg', '未知错误')}")
                
                except http.client.HTTPException as e:
                    logger.warning(f"搜索财经新闻HTTP错误 (重试 {retry+1}/{max_retries}): {e}")
                    status_info = {"code": -1, "msg": f"HTTP错误: {str(e)}"}
                except json.JSONDecodeError as e:
                    logger.error(f"解析搜索结果JSON失败: {e}")
                    status_info = {"code": -2, "msg": f"JSON解析错误: {str(e)}"}
                    break  # JSON解析错误不需要重试
                except Exception as e:
                    logger.warning(f"搜索财经新闻失败 (重试 {retry+1}/{max_retries}): {e}")
                    status_info = {"code": -3, "msg": f"未知错误: {str(e)}"}
                
                # 指数退避重试
                if retry < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2 ** retry))
            
            return [], status_info
        
        # 只获取一次新闻，禁用word参数，获取20条
        news_list, status_info = await fetch_news_for_query()
        all_news.extend(news_list)
        
        # 去重并限制结果数量
        seen_titles = set()
        unique_news = []
        for news_item in all_news:
            if news_item["title"] not in seen_titles:
                seen_titles.add(news_item["title"])
                unique_news.append(news_item)
                if len(unique_news) >= num_results:
                    break
        
        # 根据return_status参数决定返回什么
        if return_status:
            return unique_news, [status_info]
        else:
            return unique_news
    
    async def analyze_market(self, check_validity: bool = True) -> Dict:
        """分析市场行情，包括大盘指数、板块表现、热点新闻等
        
        Args:
            check_validity: 是否检查数据有效性
        
        Returns:
            包含市场分析结果的字典
        """
        try:
            # 1. 首先尝试使用预计算的数据（最快）
            precomputed_data = self.get_precomputed_market_data()
            if precomputed_data:
                logger.info("使用预计算的市场分析结果")
                if check_validity and self._is_data_valid(precomputed_data):
                    return precomputed_data
                elif not check_validity:
                    return precomputed_data
                else:
                    logger.warning("预计算数据无效")
                
            # 2. 如果没有预计算数据，检查缓存
            cache_key = self._get_cache_key("analyze_market")
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("使用缓存的市场分析结果")
                if check_validity and self._is_data_valid(cached_result):
                    return cached_result
                elif not check_validity:
                    return cached_result
                else:
                    logger.warning("缓存数据无效")
            
            # 1. 获取大盘数据
            index_symbols = ['sh000001', 'sz399001']  # 上证指数和深证成指
            try:
                index_data = data_agent.get_index_quotes(index_symbols)
            except Exception as e:
                logger.error(f"获取大盘数据失败: {e}")
                index_data = {}
            
            # 2. 获取市场概览
            try:
                market_overview = data_agent.get_market_overview()
            except Exception as e:
                logger.error(f"获取市场概览失败: {e}")
                market_overview = {
                    "total_stocks": 5000,
                    "up_stocks": 2000,
                    "down_stocks": 2500,
                    "flat_stocks": 500,
                    "up_ratio": 40.0,
                    "down_ratio": 50.0
                }
            
            # 3. 获取行业板块数据
            try:
                # 清除行业数据的缓存，确保获取最新数据
                if hasattr(data_agent, 'cache'):
                    cache_key = data_agent._get_cache_key("get_industry_data")
                    if cache_key in data_agent.cache:
                        del data_agent.cache[cache_key]
                industry_data = data_agent.get_industry_data()
                logger.info(f"获取到的行业数据形状: {industry_data.shape}")
                if not industry_data.empty and '涨跌幅' in industry_data.columns:
                    logger.info(f"获取到的行业数据涨跌幅统计: 平均值={industry_data['涨跌幅'].mean()}, 最大值={industry_data['涨跌幅'].max()}, 最小值={industry_data['涨跌幅'].min()}")
                
            except Exception as e:
                logger.error(f"获取行业板块数据失败: {e}")
                industry_data = pd.DataFrame()
            
            # 4. 获取板块资金流向排名数据
            try:
                sector_fund_flow = data_agent.get_sector_fund_flow_rank()
                logger.info(f"获取到的板块资金流向数据形状: {sector_fund_flow.shape}")
                logger.info(f"资金流向数据列名: {list(sector_fund_flow.columns) if not sector_fund_flow.empty else []}")
                if not sector_fund_flow.empty:
                    logger.info(f"资金流向数据前5行: {sector_fund_flow.head().to_dict(orient='records')}")
            except Exception as e:
                logger.error(f"获取板块资金流向排名数据失败: {e}")
                sector_fund_flow = pd.DataFrame()
            
            # 提取资金流向排名前5的板块
            top_fund_flow_sectors = []
            if sector_fund_flow is not None and not sector_fund_flow.empty:
                for _, row in sector_fund_flow.head(5).iterrows():
                    sector_name = row.get('板块名称', row.get('name', row.get('名称', '未知板块')))
                    top_fund_flow_sectors.append(sector_name)
                logger.info(f"资金净流入排名前5的板块: {top_fund_flow_sectors}")
            
            # 4. 搜索当日财经新闻
            today_date = datetime.datetime.now().strftime("%Y-%m-%d")
            try:
                # 获取市场整体新闻，传递资金流向排名前5的板块作为关键词
                general_news = await self.search_financial_news(f"{today_date} 财经新闻 A股市场", num_results=10, keywords=top_fund_flow_sectors)
                
                # 获取主要板块的相关新闻
                sector_news = []
                # 合并主要板块和资金流向排名前5的板块
                main_sectors = list(set(["科技", "新能源", "医药", "金融", "消费"] + top_fund_flow_sectors))
                for sector in main_sectors:
                    sector_news += await self.search_financial_news(f"{today_date} {sector} 板块 新闻", num_results=3)
                
                # 合并新闻并去重
                all_news = general_news + sector_news
                seen_titles = set()
                unique_news = []
                for news_item in all_news:
                    if news_item["title"] not in seen_titles:
                        seen_titles.add(news_item["title"])
                        unique_news.append(news_item)
                
                news = unique_news
                logger.info(f"共获取到 {len(news)} 条财经新闻")
            except Exception as e:
                logger.error(f"搜索财经新闻失败: {e}")
                news = [
                    {"title": "市场震荡运行", "snippet": "今日A股市场震荡运行，市场情绪相对谨慎。", "date": "今天"},
                    {"title": "资金面保持平稳", "snippet": "市场资金面保持平稳，成交量与昨日基本持平。", "date": "今天"}
                ]
            
            # 5. 调用模型生成市场分析
            try:
                prompt = self._build_market_analysis_prompt(index_data, market_overview, industry_data, news, sector_fund_flow)
                market_analysis = await model_manager.async_generate_response(prompt)
                
                # 检查模型返回的是否是错误信息
                if market_analysis and ("模型未加载成功" in market_analysis or "无法生成响应" in market_analysis or "error" in market_analysis.lower()):
                    logger.warning("检测到模型返回的错误信息，将回退到本地生成分析结果")
                    # 直接抛出异常，进入本地生成逻辑
                    raise Exception("模型返回错误信息")
                
                # 6. 解析模型输出
                analysis_result = self._parse_analysis_result(market_analysis)
                
                # 检查解析结果是否为空
                if not analysis_result["market_summary"] and not analysis_result["strong_sectors"]:
                    logger.warning("模型解析结果为空，将回退到本地生成分析结果")
                    # 抛出异常，进入本地生成逻辑
                    raise Exception("模型解析结果为空")
                
                # 检查是否有单个章节缺失，触发对应章节的本地生成
                index_data = data_agent.get_index_quotes(['sh000001', 'sz399001']) if not 'index_data' in locals() else index_data
                market_overview = data_agent.get_market_overview() if not 'market_overview' in locals() else market_overview
                # 直接使用已经获取的最新行业数据，不重新从缓存获取
                industry_data = industry_data if 'industry_data' in locals() else data_agent.get_industry_data()
                
                if not analysis_result["rotation_analysis"]:
                    logger.warning("模型未生成板块轮动分析，将使用本地生成结果")
                    analysis_result["rotation_analysis"] = self._generate_rotation_analysis(industry_data)
                    logger.info(f"本地生成的板块轮动分析: {analysis_result['rotation_analysis'][:100]}...")
                
                if not analysis_result["tomorrow_prediction"]:
                    logger.warning("模型未生成明日板块预测，将使用本地生成结果")
                    analysis_result["tomorrow_prediction"] = self._generate_tomorrow_prediction(index_data, market_overview, industry_data, news, sector_fund_flow)
                    logger.info(f"本地生成的明日板块预测: {analysis_result['tomorrow_prediction'][:100]}...")
                
                if not analysis_result["market_summary"]:
                    logger.warning("模型未生成大盘综述，将使用本地生成结果")
                    analysis_result["market_summary"] = self._generate_detailed_market_summary(index_data, market_overview)
                    logger.info(f"本地生成的大盘综述: {analysis_result['market_summary'][:100]}...")
                
                if not analysis_result["strong_sectors"]:
                    logger.warning("模型未生成强势板块，将使用本地生成结果")
                    analysis_result["strong_sectors"] = self._get_strong_sectors(industry_data)
                    logger.info(f"本地生成的强势板块: {analysis_result['strong_sectors']}")
            except Exception as e:
                logger.error(f"模型分析失败或结果为空: {e}")
                
                # 基于真实数据或默认数据生成更详细的分析结果
                market_summary = self._generate_detailed_market_summary(index_data, market_overview)
                strong_sectors = self._get_strong_sectors(industry_data)
                rotation_analysis = self._generate_rotation_analysis(industry_data)
                tomorrow_prediction = self._generate_tomorrow_prediction(index_data, market_overview, industry_data, news, sector_fund_flow)
                
                logger.info(f"本地生成的市场综述: {market_summary[:100]}...")
                logger.info(f"本地生成的强势板块: {strong_sectors}")
                logger.info(f"本地生成的板块轮动分析: {rotation_analysis[:100]}...")
                logger.info(f"本地生成的明日板块预测: {tomorrow_prediction[:100]}...")
                
                analysis_result = {
                    "market_summary": market_summary,
                    "strong_sectors": strong_sectors,
                    "rotation_analysis": rotation_analysis,
                    "tomorrow_prediction": tomorrow_prediction,
                    "market_data": {
                        "market_overview": market_overview,
                        "index_data": index_data,
                        "industry_performance": self._get_industry_performance(industry_data)
                    }
                }
            
            # 7. 添加时间戳
            analysis_result["timestamp"] = time.time()
            
            # 8. 缓存结果
            self._cache_result(cache_key, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"市场分析失败: {e}")
            # 获取一些基本数据用于生成默认分析
            index_data = data_agent.get_index_quotes(['sh000001', 'sz399001'])
            market_overview = data_agent.get_market_overview()
            industry_data = data_agent.get_industry_data()
            
            return {
                "market_summary": self._generate_detailed_market_summary(index_data, market_overview),
                "strong_sectors": self._get_strong_sectors(industry_data),
                "rotation_analysis": self._generate_rotation_analysis(industry_data),
                "tomorrow_prediction": self._generate_tomorrow_prediction(index_data, market_overview, industry_data, []),
                "market_data": {
                    "market_overview": market_overview,
                    "index_data": index_data,
                    "industry_performance": self._get_industry_performance(industry_data)
                }
            }
    
    def _generate_detailed_market_summary(self, index_data: Dict, market_overview: Dict) -> str:
        """生成详细的市场概览"""
        # 提取指数数据
        sh_data = index_data.get('sh000001')
        sz_data = index_data.get('sz399001')
        
        # 不使用默认值，直接从数据中获取
        sh_price = None
        sh_change = None
        sz_price = None
        sz_change = None
        
        if sh_data is not None and not sh_data.empty:
            if '最新价' in sh_data.columns and '涨跌幅' in sh_data.columns:
                sh_price = float(sh_data.iloc[0]['最新价'])
                sh_change = float(sh_data.iloc[0]['涨跌幅'])
        
        if sz_data is not None and not sz_data.empty:
            if '最新价' in sz_data.columns and '涨跌幅' in sz_data.columns:
                sz_price = float(sz_data.iloc[0]['最新价'])
                sz_change = float(sz_data.iloc[0]['涨跌幅'])
        
        # 获取市场概览数据，为None的情况提供默认值
        up_stocks = market_overview.get('up_stocks', 0)
        down_stocks = market_overview.get('down_stocks', 0)
        flat_stocks = market_overview.get('flat_stocks', 0)
        up_ratio = market_overview.get('up_ratio', 0)
        down_ratio = market_overview.get('down_ratio', 0)
        avg_change = market_overview.get('avg_change', 0)
        
        # 改进市场表现判断逻辑
        # 处理可能为None的情况
        if sh_change is None and sz_change is None:
            market_performance = "数据不足，无法判断"
        elif sh_change is None:
            # 仅使用深证成指数据
            if sz_change > 1.0:
                market_performance = "强势上涨"
            elif sz_change > 0.5:
                market_performance = "小幅上涨"
            elif abs(sz_change) < 0.5:
                market_performance = "震荡整理"
            elif sz_change < -0.5:
                market_performance = "小幅下跌"
            else:
                market_performance = "弱势下行"
        elif sz_change is None:
            # 仅使用上证指数数据
            if sh_change > 1.0:
                market_performance = "强势上涨"
            elif sh_change > 0.5:
                market_performance = "小幅上涨"
            elif abs(sh_change) < 0.5:
                market_performance = "震荡整理"
            elif sh_change < -0.5:
                market_performance = "小幅下跌"
            else:
                market_performance = "弱势下行"
        else:
            # 同时使用两个指数数据
            if sh_change > 1.0 and sz_change > 1.0:
                market_performance = "强势上涨"
            elif sh_change > 0.5 and sz_change > 0.5:
                market_performance = "小幅上涨"
            elif abs(sh_change) < 0.5 and abs(sz_change) < 0.5:
                market_performance = "震荡整理"
            elif sh_change < -0.5 and sz_change < -0.5:
                market_performance = "小幅下跌"
            else:
                market_performance = "弱势下行"
        
        # 改进市场活跃度判断逻辑
        total_stocks = up_stocks + down_stocks + flat_stocks
        if total_stocks > 0:
            # 计算涨跌家数差值比例
            diff_ratio = abs(up_stocks - down_stocks) / total_stocks
            if diff_ratio > 0.3:
                activity = "活跃，市场情绪分化明显"
            elif diff_ratio > 0.15:
                activity = "较活跃，市场方向较为明确"
            else:
                activity = "相对平稳，市场情绪谨慎"
        else:
            activity = "相对平稳"
        
        # 构建市场概览文本，只包含有数据的部分
        summary_parts = []
        
        if sh_price is not None and sh_change is not None:
            summary_parts.append(f"上证指数报收于{sh_price:.2f}点，{'上涨' if sh_change >= 0 else '下跌'}{abs(sh_change):.2f}%")
        
        if sz_price is not None and sz_change is not None:
            summary_parts.append(f"深证成指报收于{sz_price:.2f}点，{'上涨' if sz_change >= 0 else '下跌'}{abs(sz_change):.2f}%")
        
        if up_stocks is not None and down_stocks is not None and flat_stocks is not None:
            summary_parts.append(f"两市共有{up_stocks}只股票上涨，{down_stocks}只股票下跌，{flat_stocks}只股票平盘")
        
        if up_ratio is not None and down_ratio is not None:
            summary_parts.append(f"上涨家数占比{up_ratio}%，下跌家数占比{down_ratio}%")
        
        if avg_change is not None:
            summary_parts.append(f"市场整体平均涨跌幅为{avg_change:.2f}%")
        
        if market_performance:
            summary_parts.insert(0, f"今日市场整体表现{market_performance}")
        
        if activity:
            summary_parts.append(f"市场活跃度{activity}")
        
        summary = "。".join(summary_parts) + "。" if summary_parts else ""
        
        return summary
    
    def _get_strong_sectors(self, industry_data: pd.DataFrame) -> List[str]:
        """获取强势板块"""
        if industry_data.empty:
            logger.warning("行业数据为空，返回空板块列表")
            return []
        
        try:
            # 修复可能的无效板块名称
            processed_data = self.data_processor.fix_missing_sectors(industry_data)
            
            # 获取前5个涨跌幅最高的板块
            strong_sectors = self.data_processor.get_top_items(
                df=processed_data, 
                field='板块名称', 
                sort_field='涨跌幅', 
                ascending=False, 
                top_n=5
            )
            
            if not strong_sectors:
                logger.warning("无法从数据中提取强势板块，使用预设板块列表")
                # 如果没有获取到任何板块，使用预设列表的前5个
                strong_sectors = self.data_processor.preset_sectors[:5]
            
            logger.info(f"获取到的强势板块: {strong_sectors}")
            return strong_sectors
        except Exception as e:
            logger.error(f"获取强势板块失败: {e}")
            # 发生异常时返回预设板块列表
            return self.data_processor.preset_sectors[:5]
    
    def _generate_rotation_analysis(self, industry_data: pd.DataFrame) -> str:
        """生成板块轮动分析"""
        if industry_data.empty:
            return "暂无板块轮动分析数据"
        
        try:
            # 使用数据处理器修复可能的无效板块名称
            processed_data = self.data_processor.fix_missing_sectors(industry_data)
            
            # 确保涨跌幅是数值类型
            try:
                processed_data['涨跌幅'] = pd.to_numeric(processed_data['涨跌幅'], errors='coerce').fillna(0.0)
            except Exception as e:
                logger.error(f"转换涨跌幅为数值类型失败: {e}")
                return "暂无板块轮动分析数据"
            
            # 计算各板块的涨跌幅分布
            positive_industries = processed_data[processed_data['涨跌幅'] > 0]
            negative_industries = processed_data[processed_data['涨跌幅'] < 0]
            flat_industries = processed_data[processed_data['涨跌幅'] == 0]
            
            # 计算涨跌比例
            total_industries = len(processed_data)
            up_ratio = len(positive_industries) / total_industries * 100 if total_industries > 0 else 0
            down_ratio = len(negative_industries) / total_industries * 100 if total_industries > 0 else 0
            
            # 分析强势和弱势板块
            strong_sectors = self._get_strong_sectors(processed_data)
            weak_sectors = []
            
            # 获取弱势板块
            try:
                sorted_industries = processed_data.sort_values(by='涨跌幅', ascending=True)
                weak_sectors = sorted_industries['板块名称'].head(3).tolist()
            except Exception as e:
                logger.error(f"获取弱势板块失败: {e}")
                weak_sectors = ['未知板块']
            
            if not strong_sectors:
                strong_sectors = self.data_processor.preset_sectors[:3]
            
            if not weak_sectors:
                weak_sectors = ['未知板块']
            
            # 分析轮动强度
            rotation_intensity = '明显' if up_ratio > 70 or down_ratio > 70 else '较强' if up_ratio > 60 or down_ratio > 60 else '平稳'
            
            # 分析风格偏好
            growth_sectors = ['科技', '新能源', '医药', '创新', '半导体', '人工智能', '5G']
            value_sectors = ['金融', '地产', '银行', '保险', '消费', '基建', '电力']
            growth_count = sum(1 for sector in strong_sectors if any(growth in sector for growth in growth_sectors))
            value_count = sum(1 for sector in strong_sectors if any(value in sector for value in value_sectors))
            
            style_preference = '成长' if growth_count > value_count else '价值' if value_count > growth_count else '均衡'
            
            # 分析资金流向（基于涨跌幅推断）
            top_3_strong = strong_sectors[:3]
            
            # 分析板块联动性
            avg_change = processed_data['涨跌幅'].mean()
            max_change = processed_data['涨跌幅'].max()
            min_change = processed_data['涨跌幅'].min()
            
            # 构建分析内容 - 新的专业模板
            analysis_parts = []
            
            # 1. 板块整体表现概览
            analysis_parts.append(f"板块整体表现：今日市场共{total_industries}个板块，其中{len(positive_industries)}个上涨({up_ratio:.1f}%)，{len(negative_industries)}个下跌({down_ratio:.1f}%)，{len(flat_industries)}个平盘")
            
            # 2. 强势板块分析
            if top_3_strong:
                analysis_parts.append(f"强势板块：领涨板块为{top_3_strong[0]}，涨幅达{max_change:.2f}%，其他强势板块包括{', '.join(top_3_strong[1:])}")
            
            # 3. 弱势板块分析
            if weak_sectors:
                analysis_parts.append(f"弱势板块：领跌板块为{weak_sectors[0]}，跌幅达{min_change:.2f}%，其他弱势板块包括{', '.join(weak_sectors[1:])}")
            
            # 4. 资金流向分析
            if top_3_strong:
                analysis_parts.append(f"资金流向：市场资金主要流向{top_3_strong[0]}和{top_3_strong[1]}板块，显示出明显的资金聚集效应")
            
            # 5. 风格偏好分析
            analysis_parts.append(f"风格偏好：当前市场风格偏好{style_preference}，{style_preference}类板块表现活跃")
            
            # 6. 轮动趋势判断
            if up_ratio > 70:
                analysis_parts.append("轮动趋势：整体市场呈现普涨格局，热点板块快速轮动，市场情绪积极")
            elif down_ratio > 70:
                analysis_parts.append("轮动趋势：市场整体回调，大部分板块走势较弱，资金避险情绪上升")
            else:
                analysis_parts.append("轮动趋势：市场分化明显，结构性机会与风险并存，板块轮动节奏{rotation_intensity}")
            
            # 7. 板块联动性分析
            try:
                avg_change_numeric = float(avg_change)
                analysis_parts.append(f"板块联动性：板块间联动性{'较强' if abs(avg_change_numeric) > 1 else '一般'}，{abs(avg_change_numeric) > 1 and '市场一致性较高' or '市场分歧较大'}")
            except (ValueError, TypeError):
                analysis_parts.append("板块联动性：板块间联动性一般，市场分歧较大")
            
            # 8. 投资建议
            if up_ratio > 60:
                analysis_parts.append("投资建议：建议关注强势板块的持续性机会，适当布局相关产业链")
            elif down_ratio > 60:
                analysis_parts.append("投资建议：建议控制仓位，关注防御性板块，等待市场企稳")
            else:
                analysis_parts.append("投资建议：建议采取均衡配置策略，关注政策利好的行业板块")
            
            analysis = "。".join(analysis_parts) + "。"
            
            return analysis
        except Exception as e:
            logger.error(f"生成板块轮动分析失败: {e}")
            # 返回默认分析
            return "板块轮动分析：市场整体呈现震荡格局，板块间分化明显，建议关注政策利好的行业板块，保持理性投资心态。"
    
    def _analyze_sector_volume_flow(self, industry_data: pd.DataFrame) -> List[str]:
        """分析板块成交量与资金流向，返回资金净流入排名前五的板块"""
        if industry_data.empty:
            return []
        
        try:
            # 确保数据包含必要字段
            industry_data['涨跌幅'] = pd.to_numeric(industry_data.get('涨跌幅', 0.0), errors='coerce').fillna(0.0)
            
            # 检查是否有成交量或成交额字段
            volume_columns = ['成交量', '成交额', '量比', 'amount']
            volume_col = None
            for col in volume_columns:
                if col in industry_data.columns:
                    volume_col = col
                    break
            
            if volume_col:
                # 计算板块的资金强度（涨跌幅 * 成交量/成交额）
                industry_data['资金强度'] = industry_data['涨跌幅'] * pd.to_numeric(industry_data[volume_col], errors='coerce').fillna(0.0)
                
                # 按资金强度降序排列，取前5个板块
                top_flow_sectors = industry_data.sort_values(by='资金强度', ascending=False).head(5)
                
                # 提取板块名称
                sector_names = []
                for _, row in top_flow_sectors.iterrows():
                    sector_names.append(row.get('板块名称', row.get('name', row.get('名称', '未知板块'))))
                
                return sector_names
            else:
                # 如果没有成交量数据，返回涨跌幅前5的板块
                strong_sectors = self._get_strong_sectors(industry_data)
                return strong_sectors[:5]
        except Exception as e:
            logger.error(f"分析板块资金流向失败: {e}")
            return []
    
    def _analyze_sector_rotation(self, industry_data: pd.DataFrame, days: int = 3) -> Dict[str, float]:
        """分析近期板块轮动节奏，返回板块持续性得分"""
        # 由于数据限制，这里我们使用当日数据结合历史强度来模拟轮动分析
        # 在实际应用中，应该使用过去几天的板块数据进行分析
        if industry_data.empty:
            return {}
        
        try:
            industry_data['涨跌幅'] = pd.to_numeric(industry_data.get('涨跌幅', 0.0), errors='coerce').fillna(0.0)
            
            # 计算板块强度得分（基于涨跌幅和成交量）
            volume_columns = ['成交量', '成交额', '量比', 'amount']
            volume_col = None
            for col in volume_columns:
                if col in industry_data.columns:
                    volume_col = col
                    break
            
            if volume_col:
                industry_data['强度得分'] = industry_data['涨跌幅'] * (pd.to_numeric(industry_data[volume_col], errors='coerce').fillna(0.0) / 100000000)  # 标准化成交量
            else:
                industry_data['强度得分'] = industry_data['涨跌幅']
            
            # 提取板块名称和强度得分
            rotation_scores = {}
            for _, row in industry_data.iterrows():
                sector_name = row.get('板块名称', row.get('name', row.get('名称', '未知板块')))
                rotation_scores[sector_name] = row['强度得分']
            
            # 返回排序后的前20个板块
            sorted_scores = dict(sorted(rotation_scores.items(), key=lambda x: x[1], reverse=True)[:20])
            return sorted_scores
        except Exception as e:
            logger.error(f"分析板块轮动节奏失败: {e}")
            return {}
    
    def _analyze_market_sentiment(self, market_overview: Dict, industry_data: pd.DataFrame) -> Dict[str, float]:
        """分析市场情绪，包括涨跌家数比、涨停数量等"""
        sentiment = {}
        
        try:
            # 计算涨跌家数比
            total_stocks = market_overview.get('total_stocks', 1)
            up_stocks = market_overview.get('up_stocks', 0)
            down_stocks = market_overview.get('down_stocks', 1)  # 避免除零
            
            sentiment['涨跌家数比'] = up_stocks / down_stocks if down_stocks > 0 else up_stocks
            sentiment['上涨比例'] = (up_stocks / total_stocks) * 100 if total_stocks > 0 else 0
            
            # 计算涨停数量（模拟，假设涨跌幅>9.8%为涨停）
            if not industry_data.empty:
                industry_data['涨跌幅'] = pd.to_numeric(industry_data.get('涨跌幅', 0.0), errors='coerce').fillna(0.0)
                sentiment['涨停板块数量'] = len(industry_data[industry_data['涨跌幅'] > 9.8])
            else:
                sentiment['涨停板块数量'] = 0
            
        except Exception as e:
            logger.error(f"分析市场情绪失败: {e}")
        
        return sentiment
    
    def _analyze_financial_news(self, news: List[Dict]) -> Dict[str, List[str]]:
        """分析财经新闻，提取政策利好和行业消息"""
        news_analysis = {
            '政策利好板块': [],
            '行业消息板块': []
        }
        
        if not news:
            return news_analysis
        
        # 定义政策和行业关键词
        policy_keywords = ['政策', '央行', '证监会', '国务院', '财政部', '发改委', '补贴', '减税', '降息', '降准']
        
        # 定义主要行业关键词
        industry_keywords = {
            '科技': ['科技', '半导体', '芯片', '人工智能', 'AI', '5G', '算力', '大数据'],
            '新能源': ['新能源', '光伏', '太阳能', '风电', '锂电池', '电动车', '充电桩'],
            '医药': ['医药', '创新药', '疫苗', '医疗', '生物', '医疗器械'],
            '消费': ['消费', '零售', '食品', '饮料', '白酒', '家电', '旅游'],
            '金融': ['金融', '银行', '保险', '券商', '证券', '基金'],
            '地产': ['地产', '房地产', '基建', '建筑', '建材'],
            '资源': ['资源', '煤炭', '石油', '天然气', '有色', '金属', '黄金']
        }
        
        try:
            for news_item in news:
                content = f"{news_item.get('title', '')} {news_item.get('snippet', '')}"
                
                # 检查政策利好
                has_policy = any(keyword in content for keyword in policy_keywords)
                
                # 检查行业消息
                for sector, keywords in industry_keywords.items():
                    if any(keyword in content for keyword in keywords):
                        if has_policy:
                            news_analysis['政策利好板块'].append(sector)
                        else:
                            news_analysis['行业消息板块'].append(sector)
            
            # 去重
            news_analysis['政策利好板块'] = list(set(news_analysis['政策利好板块']))
            news_analysis['行业消息板块'] = list(set(news_analysis['行业消息板块']))
            
        except Exception as e:
            logger.error(f"分析财经新闻失败: {e}")
        
        return news_analysis
    
    def _generate_tomorrow_prediction(self, index_data: Dict, market_overview: Dict, industry_data: pd.DataFrame, news: List[Dict] = None, sector_fund_flow: pd.DataFrame = None) -> str:
        """基于多因素综合分析生成明日板块预测"""
        try:
            # 1. 分析大盘涨跌及成交量变化
            sh_data = index_data.get('sh000001')
            sz_data = index_data.get('sz399001')
            
            sh_change = 0.0
            sz_change = 0.0
            
            if sh_data is not None and not sh_data.empty:
                if '涨跌幅' in sh_data.columns:
                    sh_change = float(sh_data.iloc[0]['涨跌幅'])
            
            if sz_data is not None and not sz_data.empty:
                if '涨跌幅' in sz_data.columns:
                    sz_change = float(sz_data.iloc[0]['涨跌幅'])
            
            # 2. 分析板块成交量与资金流向
            top_flow_sectors = self._analyze_sector_volume_flow(industry_data)
            
            # 3. 分析板块资金流向排名
            fund_flow_sectors = []
            if sector_fund_flow is not None and not sector_fund_flow.empty:
                # 获取资金净流入排名前5的板块
                for _, row in sector_fund_flow.head(5).iterrows():
                    sector_name = row.get('板块名称', row.get('name', row.get('名称', '未知板块')))
                    fund_flow_sectors.append(sector_name)
                logger.info(f"资金净流入排名前5的板块: {fund_flow_sectors}")
            
            # 4. 分析近期板块轮动节奏
            rotation_scores = self._analyze_sector_rotation(industry_data)
            
            # 5. 分析市场情绪
            market_sentiment = self._analyze_market_sentiment(market_overview, industry_data)
            
            # 6. 分析财经新闻
            news_analysis = self._analyze_financial_news(news or [])
            
            # 7. 北向资金板块配置（已移除）
            north_bound_sectors = []
            
            # 8. 计算板块相对强弱（RSI）
            sector_rsi = self._calculate_sector_rsi(industry_data)
            
            # 9. 分析涨停板/炸板率（已禁用）
            sector_boom_rate = {}
            
            # 10. 分析隔夜外盘映射（已禁用）
            overseas_mapping = []
            
            # 11. 识别市场状态
            market_state = self._identify_market_state(sh_change, sz_change, market_sentiment)
            
            # 12. 综合所有因素，生成预测
            predicted_sectors = []
            sector_confidence = {}
            sector_drivers = {}
            sector_risks = {}
            
            # 策略1: 多因子综合评分
            # 构建潜在板块集合，只使用非空的数据源
            potential_sectors = set()
            
            if top_flow_sectors:
                potential_sectors.update(top_flow_sectors[:5])
            if fund_flow_sectors:
                potential_sectors.update(fund_flow_sectors[:5])
            if news_analysis:
                potential_sectors.update(news_analysis.get('政策利好板块', []))
                potential_sectors.update(news_analysis.get('行业消息板块', []))
            if overseas_mapping:
                potential_sectors.update(overseas_mapping[:3])
            
            if potential_sectors:
                # 动态调整权重
                weights = self._get_dynamic_weights(market_state)
                
                # 计算每个板块的综合得分
                sector_scores = {}
                for sector in potential_sectors:
                    score = 0
                    drivers = []
                    risks = []
                    
                    # 资金流向因子
                    if fund_flow_sectors and sector in fund_flow_sectors[:5]:
                        score += weights['fund_flow']
                        drivers.append("主力资金净流入")
                    if top_flow_sectors and sector in top_flow_sectors[:5]:
                        score += weights['volume_flow']
                        drivers.append("成交量放大")
                    
                    # 新闻催化因子
                    if news_analysis:
                        if sector in news_analysis.get('政策利好板块', []):
                            score += weights['policy_news']
                            drivers.append("政策利好")
                        if sector in news_analysis.get('行业消息板块', []):
                            score += weights['industry_news']
                            drivers.append("行业消息")
                    
                    # 轮动因子
                    if rotation_scores and sector in rotation_scores:
                        max_rotation = max(rotation_scores.values())
                        if max_rotation > 0:
                            score += weights['rotation'] * (rotation_scores[sector] / max_rotation)
                            drivers.append("板块轮动走强")
                    
                    # 相对强弱因子
                    if sector_rsi and sector in sector_rsi:
                        if sector_rsi[sector] > 50:
                            score += weights['rsi']
                            drivers.append("相对强弱良好")
                        elif sector_rsi[sector] > 70:
                            risks.append("RSI已处高位，注意回调")
                    
                    # 涨停板因子
                    if sector_boom_rate and sector in sector_boom_rate:
                        if sector_boom_rate[sector].get('boom_rate', 0) > 0.1:
                            score += weights['boom_rate']
                            drivers.append("板块内涨停股增多")
                    
                    # 外盘映射因子
                    if overseas_mapping and sector in overseas_mapping:
                        score += weights['overseas_mapping']
                        drivers.append("外盘映射利好")
                    
                    # 一致性检验
                    factor_count = 0
                    if fund_flow_sectors and sector in fund_flow_sectors[:5]:
                        factor_count += 1
                    if news_analysis:
                        if sector in news_analysis.get('政策利好板块', []) or sector in news_analysis.get('行业消息板块', []):
                            factor_count += 1
                    if rotation_scores and sector in rotation_scores and rotation_scores[sector] > 0:
                        factor_count += 1
                    
                    if factor_count >= 2:
                        score *= 1.2  # 一致性加成
                        drivers.append("多因素共振")
                    
                    # 风险扣除
                    if market_state == '下跌':
                        score *= 0.8  # 下跌市道降低评分
                    
                    # 量价背离检查
                    if top_flow_sectors and sector in top_flow_sectors[:5] and fund_flow_sectors and sector not in fund_flow_sectors[:10]:
                        score *= 0.7  # 量价背离，降低评分
                        risks.append("量价背离")
                    
                    sector_scores[sector] = score
                    sector_drivers[sector] = drivers
                    sector_risks[sector] = risks
                
                # 按评分排序
                sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
                
                # 计算置信度
                if sector_scores:
                    max_score = max(sector_scores.values())
                    for sector, score in sorted_sectors[:3]:
                        confidence = min(100, int((score / max_score) * 100))
                        sector_confidence[sector] = confidence
                        predicted_sectors.append(sector)
            
            # 策略2: 如果没有多因子支持的板块，使用轮动得分高的板块
            if not predicted_sectors and rotation_scores:
                predicted_sectors = list(rotation_scores.keys())[:3]
                for sector in predicted_sectors:
                    sector_confidence[sector] = 50
                    sector_drivers[sector] = ["近期板块轮动持续性强"]
                    sector_risks[sector] = []
            
            # 策略3: 如果没有轮动数据，使用强势板块
            if not predicted_sectors:
                strong_sectors = self._get_strong_sectors(industry_data)
                if strong_sectors:
                    predicted_sectors = strong_sectors[:3]
                    for sector in predicted_sectors:
                        sector_confidence[sector] = 40
                        sector_drivers[sector] = ["今日市场表现强势"]
                        sector_risks[sector] = []
                else:
                    # 策略4: 如果强势板块也为空，返回明确的无数据信息
                    return "明日板块预测：因数据缺失，无法进行板块预测，建议观望"

            
            # 评估市场系统性风险
            system_risk = self._assess_system_risk(market_sentiment, market_state)
            
            # 格式化输出
            if predicted_sectors:
                # 确保板块名称不重复
                predicted_sectors = list(dict.fromkeys(predicted_sectors))[:3]
                
                # 构建预测结果
                prediction = "明日板块预测\n"
                for i, sector in enumerate(predicted_sectors, 1):
                    confidence = sector_confidence.get(sector, 50)
                    drivers = sector_drivers.get(sector, ["综合市场因素"])
                    risks = sector_risks.get(sector, [])
                    
                    prediction += f"{i}. {sector}（置信度{confidence}%）\n"
                    prediction += f"   - 驱动：{'; '.join(drivers)}\n"
                    if risks:
                        prediction += f"   - 风险：{'; '.join(risks)}\n"
                
                # 添加市场系统性风险提示
                if system_risk:
                    prediction += f"\n市场风险提示：{system_risk}\n"
                
                return prediction
            else:
                # 最终备用策略：返回明确的无数据信息
                return "明日板块预测：因数据缺失，无法进行板块预测，建议观望"
        except Exception as e:
            logger.error(f"生成明日板块预测失败: {e}")
            import traceback
            traceback.print_exc()
            # 异常情况下返回明确的无数据信息
            return "明日板块预测：因数据缺失，无法进行板块预测，建议观望"
    

    def _calculate_sector_rsi(self, industry_data: pd.DataFrame) -> Dict[str, float]:
        """计算板块相对强弱（RSI）"""
        sector_rsi = {}
        
        try:
            # 检查缓存
            cache_key = self._get_cache_key("_calculate_sector_rsi")
            cached_rsi = self._get_cached_result(cache_key)
            if cached_rsi:
                logger.info("使用缓存的RSI数据")
                return cached_rsi
            
            # 尝试使用StockAPI获取板块RSI数据
            import requests
            from datetime import datetime, timedelta
            
            # 1. 先获取板块代码列表
            bk_list_url = "https://www.stockapi.com.cn/v1/base/bk"
            try:
                bk_response = requests.get(bk_list_url, timeout=10)
                bk_response.raise_for_status()
                bk_data = bk_response.json()
                
                # 检查返回数据
                if bk_data.get('code') == 20000 and bk_data.get('data'):
                    bk_list = bk_data['data']
                    logger.info(f"从StockAPI获取到 {len(bk_list)} 个板块代码")
                else:
                    logger.warning(f"StockAPI板块列表返回错误: {bk_data.get('msg', '未知错误')}")
                    # 如果获取板块列表失败，返回空字典
                    logger.info("StockAPI板块列表获取失败，返回空RSI数据")
                    return sector_rsi
            except Exception as e:
                logger.warning(f"从StockAPI获取板块列表失败: {e}")
                # 如果获取板块列表失败，返回空字典
                logger.info("StockAPI板块列表获取失败，返回空RSI数据")
                return sector_rsi
            
            # 2. 为每个板块获取RSI数据
            # 计算日期范围（最近1天）
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = end_date
            
            # 限制获取RSI的板块数量，避免API调用过多
            max_sectors = min(10, len(bk_list))
            processed_sectors = 0
            
            for bk in bk_list[:max_sectors]:
                try:
                    bk_code = bk.get('code', '')
                    bk_name = bk.get('name', '')
                    
                    if not bk_code:
                        continue
                    
                    # 调用日线RSI接口
                    rsi_url = "https://www.stockapi.com.cn/v1/quota/rsi2"
                    params = {
                        "code": bk_code,
                        "cycle1": 6,
                        "cycle2": 12,
                        "cycle3": 24,
                        "startDate": start_date,
                        "endDate": end_date,
                        "calculationCycle": 100  # 100-日
                    }
                    
                    rsi_response = requests.get(rsi_url, params=params, timeout=10)
                    rsi_response.raise_for_status()
                    rsi_data = rsi_response.json()
                    
                    # 检查返回数据
                    if rsi_data.get('code') == 20000 and rsi_data.get('data'):
                        # 获取最新的RSI值
                        rsi_values = rsi_data['data']
                        if rsi_values:
                            latest_rsi_data = rsi_values[-1] if isinstance(rsi_values, list) else rsi_values
                            if isinstance(latest_rsi_data, dict):
                                # 使用12日RSI值
                                rsi_value = latest_rsi_data.get('rsi2', 50)
                                if isinstance(rsi_value, (int, float)):
                                    sector_rsi[bk_name] = float(rsi_value)
                                    processed_sectors += 1
                                    logger.debug(f"板块 {bk_name} ({bk_code}) RSI: {rsi_value}")
                    else:
                        logger.debug(f"板块 {bk_name} ({bk_code}) RSI数据为空")
                        
                except Exception as e:
                    logger.debug(f"获取板块 {bk.get('name', '')} RSI失败: {e}")
                    continue
            
            logger.info(f"成功获取 {processed_sectors} 个板块的RSI数据")
            
            # 如果没有获取到任何RSI数据，返回空字典
            if not sector_rsi:
                logger.warning("StockAPI未返回任何RSI数据，返回空RSI数据")
                return sector_rsi
            
            # 缓存RSI数据（缓存24小时）
            if sector_rsi:
                self._cache_result(cache_key, sector_rsi)
                logger.info("已缓存RSI数据")
                    
        except Exception as e:
            logger.error(f"计算板块RSI失败: {e}")
            # 如果出错，返回空字典
            logger.info("计算RSI时发生异常，返回空RSI数据")
        
        return sector_rsi
    
    def _calculate_rsi_from_industry_data(self, industry_data: pd.DataFrame) -> Dict[str, float]:
        """基于行业数据计算板块RSI（备用方法）"""
        sector_rsi = {}
        
        if not industry_data.empty:
            # 基于涨跌幅计算简单的RSI值
            for _, row in industry_data.head(10).iterrows():
                sector_name = row.get('板块名称', row.get('name', row.get('名称', '未知板块')))
                change = row.get('涨跌幅', 0)
                
                # 简单的RSI计算：基于涨跌幅映射到0-100范围
                # 涨幅越大，RSI越高；跌幅越大，RSI越低
                if change > 0:
                    rsi = 50 + min(50, change * 5)  # 50-100
                elif change < 0:
                    rsi = 50 + max(-50, change * 5)  # 0-50
                else:
                    rsi = 50
                
                sector_rsi[sector_name] = round(rsi, 2)
        
        return sector_rsi
    
    def _analyze_sector_boom_rate(self) -> Dict[str, Dict]:
        """分析板块涨停板/炸板率（使用AKShare获取）"""
        try:
            # 尝试使用AKShare获取板块涨停板数据
            import akshare as ak
            # 获取行业板块数据
            industry_data = ak.stock_board_industry_name_ths()
            if not industry_data.empty:
                # 计算板块涨停率（模拟实现，实际应该统计板块内涨停股票数）
                boom_rate_data = {}
                for _, row in industry_data.head(10).iterrows():
                    sector_name = row.get('板块名称', row.get('name', '未知板块'))
                    # 基于涨跌幅模拟涨停率
                    change = float(row.get('涨跌幅', 0))
                    boom_rate = min(1.0, max(0.0, (change - 5) / 5)) if change > 5 else 0
                    boom_rate_data[sector_name] = {
                        "boom_rate": boom_rate,
                        "bust_rate": 0.05  # 固定炸板率
                    }
                logger.info("从AKShare成功获取板块涨停板数据")
                return boom_rate_data
            else:
                logger.warning("AKShare未返回有效的行业板块数据")
                return {}
        except Exception as e:
            logger.warning(f"从AKShare获取板块涨停板数据失败: {e}")
            return {}
    
    def _analyze_overseas_mapping(self) -> List[str]:
        """分析隔夜外盘映射（已禁用）"""
        # 已根据要求禁用此功能
        logger.info("隔夜外盘映射分析功能已禁用")
        return []
    
    def _identify_market_state(self, sh_change: float, sz_change: float, market_sentiment: Dict) -> str:
        """识别市场状态"""
        avg_change = (sh_change + sz_change) / 2
        up_ratio = market_sentiment.get('上涨比例', 50)
        
        if avg_change > 1 and up_ratio > 70:
            return '上升'
        elif avg_change < -1 and up_ratio < 30:
            return '下跌'
        else:
            return '震荡'
    
    def _get_dynamic_weights(self, market_state: str) -> Dict:
        """根据市场状态动态调整权重"""
        base_weights = {
            'fund_flow': 0.3,
            'volume_flow': 0.15,
            'north_bound': 0.15,
            'policy_news': 0.2,
            'industry_news': 0.1,
            'rotation': 0.1,
            'rsi': 0.05,
            'boom_rate': 0.03,
            'overseas_mapping': 0.02
        }
        
        # 根据市场状态调整权重
        if market_state == '上升':
            base_weights['fund_flow'] = 0.4
            base_weights['rotation'] = 0.15
            base_weights['boom_rate'] = 0.05
        elif market_state == '下跌':
            base_weights['policy_news'] = 0.25
            base_weights['north_bound'] = 0.2
            base_weights['rsi'] = 0.1
        
        return base_weights
    
    def _assess_system_risk(self, market_sentiment: Dict, market_state: str) -> str:
        """评估市场系统性风险"""
        up_stocks = market_sentiment.get('上涨家数', 0)
        down_stocks = market_sentiment.get('下跌家数', 1)
        up_ratio = market_sentiment.get('上涨比例', 50)
        limit_up_count = market_sentiment.get('涨停板块数量', 0)
        
        if down_stocks > 0:
            up_down_ratio = up_stocks / down_stocks
            if up_down_ratio < 0.3 and limit_up_count < 30:
                return "市场情绪低迷，预测可靠性降低"
        
        if market_state == '下跌' and up_ratio < 20:
            return "市场处于下跌趋势，建议谨慎操作"
        
        return ""
    
    def _extract_news_keywords(self, news: List[Dict]) -> List[str]:
        """从新闻中提取关键词"""
        keywords = []
        for news_item in news:
            title = news_item.get('title', '')
            snippet = news_item.get('snippet', '')
            content = title + ' ' + snippet
            
            # 简单提取金融相关关键词
            financial_keywords = ['政策', '央行', '利率', '汇率', '外资', '资金', '流动性', '通胀', '经济数据', '财报', '监管']
            for keyword in financial_keywords:
                if keyword in content:
                    keywords.append(keyword)
        
        return list(set(keywords))
    
    def _get_risk_factors(self, news_keywords: List[str]) -> str:
        """获取风险因素"""
        if not news_keywords:
            return "外部市场波动"
        
        return '、'.join(news_keywords)
    
    def _get_industry_performance(self, industry_data: pd.DataFrame) -> Dict:
        """获取行业表现数据，包括强势板块、弱势板块和整体统计"""
        # 初始化返回结构
        result = {
            "strong_sectors": [],  # 强势板块（前10）
            "weak_sectors": [],    # 弱势板块（后5）
            "sector_statistics": {},  # 行业整体统计信息
            "total_sectors": 0      # 总行业数量
        }
        
        if industry_data.empty:
            return result
        
        try:
            # 确保有需要的字段
            # 不直接设置为0.0，而是使用数据处理器来映射正确的涨跌幅字段
            required_fields = ['板块名称', '涨跌幅']
            processed_data = self.data_processor.map_fields(industry_data, required_fields)
            
            # 修复可能的无效板块名称
            processed_data = self.data_processor.fix_missing_sectors(processed_data)
            
            if '涨跌幅' not in processed_data.columns:
                logger.error("处理后的行业数据仍然没有涨跌幅字段")
                return result
                
            # 确保'涨跌幅'字段是数值类型
            try:
                # 首先尝试使用pd.to_numeric进行安全转换
                processed_data['涨跌幅'] = pd.to_numeric(processed_data['涨跌幅'], errors='coerce')
                
                # 如果转换后仍有NaN值，尝试处理带百分号的字符串
                if processed_data['涨跌幅'].isnull().any():
                    # 转换为字符串并去除百分号
                    temp_col = processed_data['涨跌幅'].astype(str).str.replace('%', '')
                    # 再次转换为数值
                    temp_col = pd.to_numeric(temp_col, errors='coerce')
                    # 填充转换成功的值
                    processed_data['涨跌幅'] = processed_data['涨跌幅'].combine_first(temp_col)
                
                # 填充剩余的NaN值为0.0
                processed_data['涨跌幅'] = processed_data['涨跌幅'].fillna(0.0)
            except Exception as e:
                logger.error(f"转换'涨跌幅'字段为数值类型失败: {e}")
                # 最终安全回退
                processed_data['涨跌幅'] = pd.to_numeric(processed_data['涨跌幅'], errors='coerce').fillna(0.0)
            
            # 确保processed_data有成交量字段
            if '成交量' not in processed_data.columns:
                processed_data['成交量'] = 0
            
            # 计算行业整体统计信息
            result['total_sectors'] = len(processed_data)
            result['sector_statistics'] = {
                "average_change": round(processed_data['涨跌幅'].mean(), 2),
                "positive_sectors": len(processed_data[processed_data['涨跌幅'] > 0]),
                "negative_sectors": len(processed_data[processed_data['涨跌幅'] < 0]),
                "flat_sectors": len(processed_data[processed_data['涨跌幅'] == 0]),
                "max_change": round(processed_data['涨跌幅'].max(), 2),
                "min_change": round(processed_data['涨跌幅'].min(), 2)
            }
            
            # 按涨跌幅排序，获取前10个最强势的板块
            sorted_industries = processed_data.sort_values(by='涨跌幅', ascending=False)
            
            # 添加强势板块
            for rank, (_, row) in enumerate(sorted_industries.head(10).iterrows(), 1):
                sector_info = {
                    "rank": rank,
                    "name": row.get('板块名称', '未知板块'),
                    "change": float(row.get('涨跌幅', 0.00)),
                    "volume": int(row.get('成交量', 0)),
                    "is_positive": True
                }
                result['strong_sectors'].append(sector_info)
            
            # 添加弱势板块（后5个）
            for rank, (_, row) in enumerate(sorted_industries.tail(5).iterrows(), 1):
                sector_info = {
                    "rank": len(sorted_industries) - rank + 1,  # 倒序排名
                    "name": row.get('板块名称', '未知板块'),
                    "change": float(row.get('涨跌幅', 0.00)),
                    "volume": int(row.get('成交量', 0)),
                    "is_positive": False
                }
                result['weak_sectors'].append(sector_info)
            
            # 如果数据不足，确保返回结构完整
            if not result['strong_sectors']:
                result['strong_sectors'] = []
            if not result['weak_sectors']:
                result['weak_sectors'] = []
            
            return result
            
        except Exception as e:
            logger.error(f"获取行业表现数据失败: {e}")
            return result
    
    def _build_market_analysis_prompt(self, index_data: Dict, market_overview: Dict, industry_data: pd.DataFrame, news: List[Dict], sector_fund_flow: pd.DataFrame = None) -> str:
        """构建市场分析提示词"""
        # 构建指数数据描述
        index_desc = ""
        for symbol, df in index_data.items():
            if not df.empty:
                name = df.iloc[0]['名称']
                price = df.iloc[0]['最新价']
                change_pct = df.iloc[0]['涨跌幅']
                index_desc += f"{name}({symbol}): {price}点，{change_pct:.2f}%\n"
        
        # 构建市场概览描述
        market_desc = f"""市场概览：
上涨家数：{market_overview.get('up_stocks', 0)}
下跌家数：{market_overview.get('down_stocks', 0)}
平盘家数：{market_overview.get('flat_stocks', 0)}
上涨比例：{market_overview.get('up_ratio', 0)}%
下跌比例：{market_overview.get('down_ratio', 0)}%
"""
        
        # 构建行业板块描述
        industry_desc = "行业板块表现：\n"
        if not industry_data.empty:
            # 确保有需要的字段
            industry_data['涨跌幅'] = pd.to_numeric(industry_data.get('涨跌幅', 0.0), errors='coerce').fillna(0.0)
            if '板块名称' not in industry_data.columns:
                industry_data['板块名称'] = industry_data.get('name', industry_data.get('名称', '未知板块'))
            
            # 取前10个行业
            top_industries = industry_data.sort_values(by='涨跌幅', ascending=False).head(10)
            for _, row in top_industries.iterrows():
                industry_desc += f"{row.get('板块名称', '未知板块')}: {row.get('涨跌幅', 0):.2f}%\n"
        
        # 构建板块资金流向描述
        fund_flow_desc = "板块资金流向排名：\n"
        if sector_fund_flow is not None and not sector_fund_flow.empty:
            # 获取资金流入排名前5的板块
            top_fund_flow = sector_fund_flow.head(5)
            for _, row in top_fund_flow.iterrows():
                # 适配不同的字段名
                sector_name = row.get('板块名称', row.get('name', row.get('名称', '未知板块')))
                fund_flow = row.get('主力净流入净额', row.get('净额', row.get('主力净流入', 0)))
                fund_flow_desc += f"{sector_name}: 主力净流入 {fund_flow:.2f} 亿元\n"
        
        # 构建新闻描述
        news_desc = "最新财经新闻：\n"
        for news_item in news:
            news_desc += f"{news_item['date']} - {news_item['title']}: {news_item['snippet']}\n"
        
        prompt = f"""作为一名专业的股票市场分析师，请根据以下信息生成一份今日市场分析报告：

{index_desc}

{market_desc}

{industry_desc}

{fund_flow_desc}

{news_desc}

请严格按照以下纯文本格式输出分析结果，禁用所有Markdown语法，不要使用#、|、---等标记，不要使用任何装饰线：

一、市场综述

1. 主要指数表现
   - 上证指数：收于 [上证指数收盘] 点，涨跌幅 [上证指数涨跌幅]%
   - 深证成指：收于 [深证成指收盘] 点，涨跌幅 [深证成指涨跌幅]%
   - 两市成交额：[成交额] 亿元，较前一交易日 [放量/缩量] [变化量] 亿元

2. 市场情绪
   - 上涨家数：[上涨家数] 家
   - 下跌家数：[下跌家数] 家
   - 平盘家数：[平盘家数] 家
   - 上涨比例：[上涨比例]%，下跌比例：[下跌比例]%
   情绪判断：[市场情绪描述]

二、影响市场的主要因素

1. 内部因素
   - [因素1描述]
   - [因素2描述]

2. 外部因素
   - [因素描述]

三、强势板块

排名  板块名称      主力净流入(亿元)  表现分析                    上涨原因                                    龙头个股
----  ------------  ----------------  --------------------------  ----------------------------------------  ----------
1     [板块名]      [净流入]          [短评]                      [核心原因]                                 [个股示例]
2     [板块名]      [净流入]          [短评]                      [核心原因]                                 [个股示例]
3     [板块名]      [净流入]          [短评]                      [核心原因]                                 [个股示例]
4     [板块名]      [净流入]          [短评]                      [核心原因]                                 [个股示例]
5     [板块名]      [净流入]          [短评]                      [核心原因]                                 [个股示例]

四、板块轮动分析

- 资金聚焦方向：[描述资金集中的板块及金额]
- 资金流出方向：[描述资金流出的板块]
- 轮动特征：[总结性判断]

五、明日板块预测

1. 科技板块
   - 明日表现：[预测]
   - 关键因素：[因素]
   - 操作建议：[建议]

2. 新能源板块
   - 明日表现：[预测]
   - 关键因素：[因素]
   - 操作建议：[建议]

3. 医药板块
   - 明日表现：[预测]
   - 关键因素：[因素]
   - 操作建议：[建议]

4. 金融板块
   - 明日表现：[预测]
   - 关键因素：[因素]
   - 操作建议：[建议]

5. 消费板块
   - 明日表现：[预测]
   - 关键因素：[因素]
   - 操作建议：[建议]

请使用专业、客观的语言，避免主观臆断，确保分析有数据支持。

特别注意：
1. 严格使用纯文本格式，禁用所有Markdown语法
2. 不要使用任何装饰线（如===、---、***等）
3. 表格用空格对齐，列表用-加空格，换行后缩进两个空格
4. 百分比保留两位小数，金额单位统一为亿元
5. 章节之间用空行分隔
6. 所有数据占位符用{{}}表示
"""
        
        return prompt
    
    def _parse_analysis_result(self, analysis_text: str) -> Dict:
        """解析分析结果，增强容错性，处理重复内容和结构混乱问题"""
        result = {
            "market_summary": "",
            "strong_sectors": [],
            "rotation_analysis": "",
            "tomorrow_prediction": ""
        }
        
        logger.info(f"模型返回的原始内容: {analysis_text[:500]}...")
        
        # 检查模型返回的是否是错误信息
        if analysis_text and ("模型未加载成功" in analysis_text or "无法生成响应" in analysis_text or "error" in analysis_text.lower()):
            logger.warning("检测到模型返回的错误信息，解析将失败")
            return result
        
        import re
        
        # 1. 初步清理
        cleaned_text = analysis_text
        # 移除模型返回的特殊标记
        cleaned_text = re.sub(r'%QUERY%:.*?<.*?>', '', cleaned_text, flags=re.DOTALL)
        # 移除所有HTML标签
        cleaned_text = re.sub(r'<.*?>', '', cleaned_text)
        # 移除所有>符号（这些是生成过程中的指令分隔符）
        cleaned_text = re.sub(r'>', '', cleaned_text)
        # 移除多余的空白字符
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # 2. 识别章节并分配内容
        # 定义章节模式
        patterns = {
            "market_summary": r'(大盘综述)(.*?)(强势板块|板块轮动|明日板块预测|$)',
            "strong_sectors": r'(强势板块)(.*?)(板块轮动|明日板块预测|$)',
            "rotation_analysis": r'(板块轮动)(.*?)(明日板块预测|$)',
            "tomorrow_prediction": r'(明日板块预测|后市展望|未来预测)(.*?$)'
        }
        
        # 提取每个章节的内容
        for section, pattern in patterns.items():
            match = re.search(pattern, cleaned_text, flags=re.DOTALL)
            if match:
                # 获取章节内容（去掉章节标题）
                content = match.group(2).strip()
                result[section] = content
                logger.info(f"直接匹配到{section}: {content[:100]}...")
        
        # 3. 处理强势板块列表
        if result["strong_sectors"]:
            sectors = []
            content = result["strong_sectors"]
            
            # 处理多种格式的板块列表
            # 1. 处理以-分隔的格式，如 "- 油气开采及服务 - 能源金属 - 电池"
            if '- ' in content:
                # 拆分并清理
                parts = content.split('- ')
                for part in parts:
                    sector = part.strip()
                    if sector and not any(keyword in sector for keyword in ["大盘综述", "板块轮动", "明日板块预测"]):
                        sectors.append(sector)
            # 2. 处理每行一个板块的格式
            else:
                lines = content.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # 处理带-前缀的板块
                    if line.startswith('-'):
                        sector = line[1:].strip()
                        if sector:
                            sectors.append(sector)
                    # 处理其他可能的格式
                    elif not any(keyword in line for keyword in ["大盘综述", "板块轮动", "明日板块预测"]):
                        # 检查是否是逗号分隔的列表
                        if ',' in line:
                            sector_list = [s.strip() for s in line.split(',') if s.strip()]
                            sectors.extend(sector_list)
                        else:
                            # 可能是单个板块名称
                            if not line.startswith(('大盘', '板块', '明日', '根据', '最后')):
                                sectors.append(line)
            
            # 去重
            sectors = list(set(sectors))
            # 过滤空字符串
            sectors = [sector for sector in sectors if sector]
            result["strong_sectors"] = sectors
            logger.info(f"解析到强势板块: {sectors}")
        
        # 4. 清理所有文本内容中的重复指令
        def clean_text_content(text):
            """清理文本内容中的重复指令"""
            if not text:
                return ""
            
            # 定义要移除的指令模式
            instruction_patterns = [
                r'最后，记得在报告中加入一些实用的表格.*?获取所需的信息\.?',
                r'根据以上内容，分析每个板块的具体表现.*?判断。?',
                r'在实际操作中，你可以选择在每个板块前添加标题.*?理解你的分析内容。?',
                r'最后，如果你有一些关于未来的预测.*?预测能力。?',
                r'最后，记得将答案以清晰、标准化的格式呈现出来.*?结果。?'
            ]
            
            cleaned_text = text
            for pattern in instruction_patterns:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
            
            return cleaned_text.strip()
        
        # 5. 应用文本清理到非列表字段
        for section in ["market_summary", "rotation_analysis", "tomorrow_prediction"]:
            if result[section]:
                result[section] = clean_text_content(result[section])
        
        # 6. 确保所有字段都有值，避免前端显示加载中
        for field in result:
            if field == "strong_sectors" and not result[field]:
                result[field] = []
            elif not result[field]:
                result[field] = ""
        
        logger.info(f"最终解析结果: {result}")
        
        return result
    
    async def get_market_overview(self) -> Dict:
        """获取市场概览，用于前端展示"""
        logger.info("开始获取市场概览数据")
        
        # 初始化结果，不使用默认数据
        analysis_result = {}
        real_time_data = {}
        
        try:
            # 直接异步调用市场分析方法（会自动检查并使用缓存）
            analysis_result = await self.analyze_market()
            logger.info("市场分析完成")
        except Exception as e:
            logger.error(f"市场分析失败: {e}")
            logger.exception("详细错误信息:")
            # 获取一些基本数据用于生成分析结果
            index_data = data_agent.get_index_quotes(['sh000001', 'sz399001', 'sh000002', 'sh000688', 'sz399006', 'hkHSI', 'usSPX', 'usDJI', 'usIXIC', 'XAUUSD'])
            market_overview = data_agent.get_market_overview()
            
            # 清除行业数据的缓存，确保获取最新数据
            if hasattr(data_agent, 'cache'):
                cache_key = data_agent._get_cache_key("get_industry_data")
                if cache_key in data_agent.cache:
                    del data_agent.cache[cache_key]
            industry_data = data_agent.get_industry_data()
            
            analysis_result = {
                "market_summary": self._generate_detailed_market_summary(index_data, market_overview),
                "strong_sectors": self._get_strong_sectors(industry_data),
                "rotation_analysis": self._generate_rotation_analysis(industry_data),
                "tomorrow_prediction": self._generate_tomorrow_prediction(index_data, market_overview, industry_data),
                "market_data": {
                    "market_overview": market_overview,
                    "industry_performance": self._get_industry_performance(industry_data)
                }
            }
        
        # 添加实时数据
        try:
            # 获取国内和海外主要指数
            index_symbols = ['sh000001', 'sz399001', 'sh000002', 'sh000688', 'sz399006', 'hkHSI', 'usSPX', 'usDJI', 'usIXIC']
            
            # 使用异步方式获取实时数据，避免阻塞
            loop = asyncio.get_running_loop()
            index_data = await loop.run_in_executor(
                None,  # 使用默认线程池
                data_agent.get_index_quotes,
                index_symbols
            )
            
            logger.info(f"获取指数数据结果: {index_data}")
            
            for symbol, df in index_data.items():
                if not df.empty:
                    # 只使用真实数据，不使用默认值
                    if all(col in df.columns for col in ['名称', '最新价', '涨跌幅']):
                        real_time_data[symbol] = {
                            "名称": df.iloc[0]['名称'],
                            "price": df.iloc[0]['最新价'],
                            "change_pct": df.iloc[0]['涨跌幅']
                        }
                    else:
                        logger.warning(f"指数 {symbol} 数据缺少必要字段: {df.columns}")
        except Exception as e:
            logger.error(f"获取实时数据失败: {e}")
        
        # 获取黄金价格数据
        try:
            gold_price_data = data_agent.get_gold_price()
            if gold_price_data:
                analysis_result["gold_price"] = gold_price_data
                logger.info("成功获取黄金价格数据并添加到市场分析结果")
        except Exception as e:
            logger.warning(f"获取黄金价格数据失败: {e}")
        
        # 确保analysis_result中的数据都是可序列化的
        try:
            if "market_data" in analysis_result:
                market_data = analysis_result["market_data"]
                
                # 移除不可序列化的数据类型（如DataFrame）
                if "index_data" in market_data:
                    del market_data["index_data"]
                if "industry_data" in market_data:
                    del market_data["industry_data"]
        except Exception as e:
            logger.error(f"处理分析结果失败: {e}")
        
        result = {
            "real_time": real_time_data,
            "analysis": analysis_result
        }
        
        logger.info(f"市场概览数据准备完成，返回结构: {list(result.keys())}")
        logger.info(f"analysis字段结构: {list(analysis_result.keys())}")
        return result
    


# 全局市场分析Agent实例
market_analysis_agent = MarketAnalysisAgent()
