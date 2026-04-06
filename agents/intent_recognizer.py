#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图识别和需求分析模块
负责分析用户输入的意图和需求，提取关键信息
"""

import re
import logging
import json
import os
from typing import Dict, List, Optional, Tuple

from .intent_type import IntentType

logger = logging.getLogger(__name__)


class IntentRecognizer:
    """意图识别器"""
    
    def __init__(self):
        # 股票代码正则表达式
        self.stock_code_pattern = re.compile(r'\b(?:[630][013789]\d{4}|[84][03]\d{3}|00[012]\d{3})\b')
        
        # 中文股票名称正则表达式（2-4个汉字，不包含"的"、"股"等无意义字符）
        self.chinese_stock_name_pattern = re.compile(r'[\u4e00-\u9fa5]{2,4}(?![的股行析走])')
        
        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 数据文件路径
        self.data_dir = os.path.join(self.root_dir, 'data')
        
        # 加载映射表
        self.common_indices = self._load_common_indices()
        self.stock_name_code_map = self._load_stock_name_code_map()
        self.common_sectors = self._load_common_sectors()
        
        # 缓存映射表以提高性能
        self._cached_sorted_indices = sorted(self.common_indices, key=len, reverse=True)
        self._cached_sorted_sectors = sorted(self.common_sectors, key=len, reverse=True)
        self._cached_sorted_stock_names = sorted(self.stock_name_code_map.keys(), key=len, reverse=True)
        
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
    
    def recognize_intent(self, user_input: str) -> Tuple[IntentType, Dict]:
        """
        识别用户输入的意图
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Tuple[IntentType, Dict]: 意图类型和提取的实体信息
        """
        try:
            user_input = user_input.strip()
            logger.info(f"开始识别意图，用户输入: {user_input}")
            
            if not user_input:
                logger.warning("用户输入为空")
                return IntentType.UNKNOWN, {}
            
            # 提取实体信息
            entities = self._extract_entities(user_input)
            logger.debug(f"提取到实体: {entities}")
            
            # 识别意图
            intent = self._classify_intent(user_input, entities)
            
            # 验证意图类型
            if not hasattr(intent, 'value'):
                logger.error(f"识别到无效意图类型: {intent}")
                return IntentType.UNKNOWN, entities
            
            logger.info(f"成功识别到意图: {intent.value}，实体: {entities}")
            
            return intent, entities
        except Exception as e:
            logger.error(f"意图识别失败: {e}", exc_info=True)
            return IntentType.UNKNOWN, {}
    
    def _extract_entities(self, user_input: str) -> Dict:
        """提取用户输入中的实体信息"""
        entities = {
            "stock_code": None,
            "stock_name": None,
            "sector": None,
            "index_name": None,
            "time_period": None,
            "analysis_type": None,
            "investment_amount": None,
            "risk_tolerance": None
        }
        
        import difflib
        
        # 模糊匹配函数
        def fuzzy_match(query, choices, threshold=0.8):
            """模糊匹配字符串，返回相似度最高且超过阈值的结果"""
            if not query or not choices:
                return None
            
            # 使用get_close_matches获取相似度最高的前3个匹配
            matches = difflib.get_close_matches(query, choices, n=3, cutoff=threshold)
            if matches:
                return matches[0]
            
            # 如果没有找到匹配，尝试更宽松的匹配（检查是否包含关键词）
            for choice in choices:
                if query in choice or choice in query:
                    return choice
            
            return None
        
        # 提取投资金额
        investment_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*(万|亿|元|块|现金)')
        investment_match = investment_pattern.search(user_input)
        if investment_match:
            entities["investment_amount"] = investment_match.group(0)
            logger.debug(f"识别到投资金额：{entities['investment_amount']}")
        
        # 提取风险承受能力
        risk_keywords = {
            "低风险": ["保守", "稳健", "低风险", "风险厌恶"],
            "中风险": ["平衡", "中等风险", "稳健增长"],
            "高风险": ["激进", "高风险", "进取", "积极"],
        }
        
        for risk_level, keywords in risk_keywords.items():
            for keyword in keywords:
                if keyword in user_input:
                    entities["risk_tolerance"] = risk_level
                    logger.debug(f"识别到风险承受能力：{risk_level}")
                    break
            if entities["risk_tolerance"]:
                break
        
        # 先不提取板块名称，等股票识别完成后再处理
        
        # 提取指数名称（需要先于股票名称识别，避免将指数误认为股票）
        # 使用缓存的排序指数列表，提高性能
        for index_name in self._cached_sorted_indices:
            if index_name in user_input:
                # 这是指数，不是股票或板块
                entities["sector"] = None  # 清除可能误识别的板块
                entities["index_name"] = index_name
                logger.debug(f"识别到指数：{index_name}")
                break
        
        # 如果没有精确匹配到指数，尝试模糊匹配
        if not entities["index_name"]:
            # 从用户输入中提取可能的指数关键词
            possible_index = None
            for word in user_input.split():
                if len(word) > 2:  # 过滤掉太短的词
                    match = fuzzy_match(word, self.common_indices, threshold=0.8)
                    if match:
                        possible_index = match
                        break
            if possible_index:
                entities["sector"] = None  # 清除可能误识别的板块
                entities["index_name"] = possible_index
                logger.debug(f"模糊匹配到指数：{possible_index}")
        
        # 提取股票代码
        stock_code_match = self.stock_code_pattern.search(user_input)
        if stock_code_match:
            entities["stock_code"] = stock_code_match.group(0)
        
        # 提取股票名称并映射到代码
        # 1. 首先检查静态映射表
        # 使用更精确的匹配，确保是完整的股票名称匹配
        # 使用缓存的排序股票名称列表，提高性能
        for stock_name in self._cached_sorted_stock_names:
            if not entities["stock_code"]:
                # 使用更严格的匹配：确保股票名称是一个完整的词，不是其他词的一部分
                # 使用正则表达式进行边界匹配
                pattern = rf'\b{re.escape(stock_name)}\b'
                if re.search(pattern, user_input):
                    # 对于静态映射表中的股票名称，确保是精确匹配
                    
                    # 确保不是板块名称的一部分（例如"新能源板块"中的"新能源"不是股票名称）
                    if entities["sector"] and stock_name == entities["sector"]:
                        continue
                    
                    # 确保不是银行类股票被误认为板块
                    if entities["sector"] == "银行" and stock_name.endswith("银行"):
                        # 这是银行股票，不是银行板块
                        entities["sector"] = None
                    
                    entities["stock_name"] = stock_name
                    entities["stock_code"] = self.stock_name_code_map[stock_name]
                    logger.debug(f"从静态映射表中精确识别到股票：{stock_name} -> {self.stock_name_code_map[stock_name]}")
                    break
        
        # 如果没有精确匹配到股票名称，尝试模糊匹配
        if not entities["stock_name"] and not entities["stock_code"]:
            # 从用户输入中提取可能的股票名称关键词
            possible_stock_name = None
            for word in user_input.split():
                if len(word) > 1:  # 过滤掉太短的词
                    # 避免将已经识别的板块或指数作为股票名称
                    if entities["sector"] and word == entities["sector"]:
                        continue
                    if entities["index_name"] and word in entities["index_name"]:
                        continue
                    
                    match = fuzzy_match(word, self.stock_name_code_map.keys(), threshold=0.8)
                    if match:
                        possible_stock_name = match
                        break
            
            if possible_stock_name:
                entities["stock_name"] = possible_stock_name
                entities["stock_code"] = self.stock_name_code_map[possible_stock_name]
                logger.debug(f"模糊匹配到股票：{possible_stock_name} -> {self.stock_name_code_map[possible_stock_name]}")
                # 如果匹配到股票，清除可能误识别的板块
                entities["sector"] = None
        
        # 2. 如果静态映射表中没有找到，尝试识别中文股票名称
        if not entities["stock_name"]:
            # 如果已经识别到指数，不需要再识别股票名称
            if "index_name" in entities and entities["index_name"]:
                pass
            else:
                stock_name_match = self.chinese_stock_name_pattern.search(user_input)
                if stock_name_match:
                    candidate_name = stock_name_match.group(0)
                    
                    # 确保候选名称不是常见的非股票词汇
                    common_non_stock = [
                        "股票", "分析", "走势", "表现", "今天", "明天", "现在", "如何", "什么",
                        "请帮", "我看", "的股票", "估值", "基本面", "技术面", "资金面",
                        "查询", "推荐", "买入", "卖出", "持有", "表现", "走势", "分析",
                        "请帮我看", "分析一下", "股票走势", "股票估值", "今天的股", "请分析一",
                        "板块", "行业", "指数", "上证", "深证", "创业板", "沪深", "中证"
                    ]
                    
                    # 检查是否是常见非股票词汇或包含非股票关键词
                    is_non_stock = False
                    for non_stock in common_non_stock:
                        if candidate_name == non_stock or non_stock in candidate_name:
                            is_non_stock = True
                            break
                    
                    # 确保不是板块名称
                    if entities["sector"] and (candidate_name == entities["sector"] or entities["sector"] in candidate_name):
                        is_non_stock = True
                        
                    # 确保不是指数名称
                    for index_name in self.common_indices:
                        if candidate_name in index_name or index_name in candidate_name:
                            is_non_stock = True
                            break
                            
                    if not is_non_stock:
                        entities["stock_name"] = candidate_name
                        # 尝试通过API查询股票代码
                        try:
                            from agents.data_agent import data_agent
                            code = data_agent.get_stock_code_by_name(candidate_name)
                            if code:
                                entities["stock_code"] = code
                                # 将新识别的股票添加到映射表中，提高后续查询效率
                                self.stock_name_code_map[candidate_name] = code
                        except Exception as e:
                            logger.debug(f"API查询股票代码失败: {e}")
        
        # 提取板块名称（股票识别之后，避免冲突）
        # 如果已经识别到股票，不优先识别板块
        if not entities["stock_name"] and not entities["stock_code"]:
            for sector in self._cached_sorted_sectors:
                if sector in user_input:
                    entities["sector"] = sector
                    logger.debug(f"识别到板块：{sector}")
                    break
        
        # 如果没有精确匹配到板块，尝试模糊匹配
        if not entities["sector"] and not entities["stock_name"] and not entities["stock_code"]:
            # 从用户输入中提取可能的板块关键词
            possible_sector = None
            for word in user_input.split():
                if len(word) > 1:  # 过滤掉太短的词
                    match = fuzzy_match(word, self.common_sectors, threshold=0.75)
                    if match:
                        possible_sector = match
                        break
            if possible_sector:
                entities["sector"] = possible_sector
                logger.debug(f"模糊匹配到板块：{possible_sector}")
        
        # 提取时间周期
        # 优先匹配更长的时间周期表达
        time_periods = [
            "最近三天", "近三天", "最近3天", "近3天", "过去三天", "过去3天",
            "最近一周", "近一周", "最近7天", "近7天", "过去一周", "过去7天",
            "最近两周", "近两周", "最近14天", "近14天", "过去两周", "过去14天",
            "最近一个月", "近一个月", "最近30天", "近30天", "过去一个月", "过去30天",
            "最近两个月", "近两个月", "最近60天", "近60天", "过去两个月", "过去60天",
            "最近三个月", "近三个月", "最近90天", "近90天", "过去三个月", "过去90天",
            "最近半年", "近半年", "过去半年", "最近一年", "近一年", "过去一年",
            "今日", "昨日", "本周", "上周", "本月", "上月", "今年", "去年"
        ]
        
        # 按长度排序，确保先匹配较长的时间周期
        sorted_time_periods = sorted(time_periods, key=len, reverse=True)
        for period in sorted_time_periods:
            if period in user_input:
                entities["time_period"] = period
                logger.debug(f"识别到时间周期：{period}")
                break
        
        # 提取分析类型
        analysis_types = [
            "基本面", "技术面", "资金面", "情绪面", "政策面", "估值", 
            "投资建议", "走势预测", "行业分析", "公司分析", "财务分析",
            "技术分析", "基本面分析", "技术面分析", "资金面分析",
            "估值分析", "投资价值分析", "走势分析", "表现分析"
        ]
        
        # 按长度排序，确保先匹配较长的分析类型
        sorted_analysis_types = sorted(analysis_types, key=len, reverse=True)
        for analysis_type in sorted_analysis_types:
            if analysis_type in user_input:
                entities["analysis_type"] = analysis_type
                logger.debug(f"识别到分析类型：{analysis_type}")
                break
        
        return entities
    
    def _classify_intent(self, user_input: str, entities: Dict) -> IntentType:
        """根据用户输入和实体信息分类意图"""
        user_input_lower = user_input.lower()
        
        # 查询类意图（优先识别，避免被其他意图覆盖）
        query_keywords = ["查询", "查一下", "查一查", "请问", "价格", "多少钱", "股票价格", "股价"]
        if any(keyword in user_input for keyword in query_keywords):
            # 如果有股票代码或名称，直接识别为股票信息查询
            if entities.get("stock_code") or entities.get("stock_name"):
                return IntentType.QUERY_STOCK_INFO
            # 如果有指数名称，识别为指数信息查询
            elif entities.get("index_name"):
                return IntentType.QUERY_MARKET_INFO
            # 如果有板块名称，识别为板块信息查询
            elif entities.get("sector"):
                return IntentType.QUERY_SECTOR_INFO
        
        # 新闻查询意图
        news_keywords = ["新闻", "资讯", "消息", "动态", "报道", "最新消息", "最新资讯", "最新动态"]
        if any(keyword in user_input for keyword in news_keywords):
            return IntentType.QUERY_NEWS
        
        # 大盘分析意图
        market_keywords = ["大盘", "上证指数", "深证成指", "创业板", "沪深300", "市场分析", 
                          "市场走势", "市场表现", "市场动态", "市场行情"]
        # 如果包含大盘关键词或已识别到指数名称，则为大盘分析意图
        if any(keyword in user_input for keyword in market_keywords) or entities.get("index_name"):
            # 确定具体的大盘分析类型
            return self._determine_market_analysis_type(user_input, entities)
        
        # 板块分析意图
        if entities.get("sector"):
            # 确定具体的板块分析类型
            return self._determine_sector_analysis_type(user_input, entities)
        
        # 股票分析意图
        # 1. 如果有股票代码或名称，直接识别为股票分析
        if entities.get("stock_code") or entities.get("stock_name"):
            # 确定具体的股票分析类型
            return self._determine_stock_analysis_type(user_input, entities)
        
        # 2. 即使没有识别到具体股票，也可以根据关键词判断为股票分析意图
        stock_analysis_keywords = ["分析", "走势", "表现", "估值", "推荐", "买入", "卖出", "持有", 
                                  "涨幅", "跌幅", "涨了", "跌了", "反弹", "回调", "调整", 
                                  "压力位", "支撑位", "MACD", "KDJ", "均线", "成交量", "成交额", 
                                  "换手率", "市盈率", "市净率", "ROE", "净利润", "营收", "分红"]
        if "股票" in user_input and any(keyword in user_input for keyword in stock_analysis_keywords):
            # 确定具体的股票分析类型
            return self._determine_stock_analysis_type(user_input, entities)
        
        # 3. 包含分析类型但未明确指定股票的情况
        if entities["analysis_type"]:
            return IntentType.ANALYSIS
        
        # 股票列表请求
        list_keywords = ["股票池", "推荐", "列表", "哪些股票", "什么股票", "股票推荐", "值得买的股票"]
        if any(keyword in user_input for keyword in list_keywords):
            return IntentType.QUERY_STOCK_LIST
        
        # 股票相关但未明确指定的查询
        stock_related_keywords = ["股票", "行情", "个股", "股价", "股票价格", "股票代码"]
        if any(keyword in user_input for keyword in stock_related_keywords):
            return IntentType.ANALYSIS_STOCK
        
        # 指数相关但未被识别为大盘分析的查询
        if "指数" in user_input:
            return IntentType.ANALYSIS_MARKET
        
        # 分析类型相关的查询（如基本面、技术面）
        analysis_related_keywords = ["分析", "看看", "了解", "研究", "评估"]
        if entities["analysis_type"] and any(keyword in user_input for keyword in analysis_related_keywords):
            return IntentType.ANALYSIS_STOCK
        
        # 无法识别的意图
        return IntentType.UNKNOWN
    
    def _determine_stock_analysis_type(self, user_input: str, entities: Dict) -> IntentType:
        """
        确定具体的股票分析类型
        
        Args:
            user_input: 用户输入文本
            entities: 实体信息
            
        Returns:
            IntentType: 具体的股票分析意图类型
        """
        # 基本面分析关键词
        fundamental_keywords = ["基本面", "财务", "业绩", "营收", "净利润", "毛利率", "净利率", 
                               "ROE", "资产负债率", "现金流", "EPS", "分红", "股息率"]
        
        # 技术面分析关键词
        technical_keywords = ["技术面", "MACD", "KDJ", "RSI", "均线", "成交量", "换手率", 
                              "支撑位", "压力位", "K线", "趋势", "金叉", "死叉"]
        
        # 估值分析关键词
        valuation_keywords = ["估值", "市盈率", "PE", "市净率", "PB", "市销率", "PS", 
                             "估值分析", "合理估值", "高估", "低估"]
        
        # 表现分析关键词
        performance_keywords = ["表现", "走势", "涨", "跌", "涨幅", "跌幅", "反弹", "回调", 
                              "走势分析", "表现分析"]
        
        # 检查关键词
        for keyword in fundamental_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_STOCK_FUNDAMENTAL
        
        for keyword in technical_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_STOCK_TECHNICAL
        
        for keyword in valuation_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_STOCK_VALUATION
        
        for keyword in performance_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_STOCK_PERFORMANCE
        
        # 默认返回股票分析
        return IntentType.ANALYSIS_STOCK
    
    def _determine_sector_analysis_type(self, user_input: str, entities: Dict) -> IntentType:
        """
        确定具体的板块分析类型
        
        Args:
            user_input: 用户输入文本
            entities: 实体信息
            
        Returns:
            IntentType: 具体的板块分析意图类型
        """
        # 基本面分析关键词
        fundamental_keywords = ["基本面", "财务", "业绩", "营收", "净利润", "行业分析", 
                               "产业分析", "行业前景", "产业趋势"]
        
        # 技术面分析关键词
        technical_keywords = ["技术面", "MACD", "KDJ", "RSI", "均线", "成交量", "换手率", 
                              "支撑位", "压力位", "走势"]
        
        # 表现分析关键词
        performance_keywords = ["表现", "走势", "涨", "跌", "涨幅", "跌幅", "反弹", "回调"]
        
        # 检查关键词
        for keyword in fundamental_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_SECTOR_FUNDAMENTAL
        
        for keyword in technical_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_SECTOR_TECHNICAL
        
        for keyword in performance_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_SECTOR_PERFORMANCE
        
        # 默认返回板块分析
        return IntentType.ANALYSIS_SECTOR
    
    def _determine_market_analysis_type(self, user_input: str, entities: Dict) -> IntentType:
        """
        确定具体的大盘分析类型
        
        Args:
            user_input: 用户输入文本
            entities: 实体信息
            
        Returns:
            IntentType: 具体的大盘分析意图类型
        """
        # 基本面分析关键词
        fundamental_keywords = ["基本面", "经济数据", "政策", "宏观经济", "GDP", "CPI", "PPI", 
                               "货币政策", "财政政策", "经济分析"]
        
        # 技术面分析关键词
        technical_keywords = ["技术面", "MACD", "KDJ", "RSI", "均线", "成交量", "换手率", 
                              "支撑位", "压力位", "走势", "趋势"]
        
        # 表现分析关键词
        performance_keywords = ["表现", "走势", "涨", "跌", "涨幅", "跌幅", "反弹", "回调", 
                              "市场分析", "市场表现"]
        
        # 检查关键词
        for keyword in fundamental_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_MARKET_FUNDAMENTAL
        
        for keyword in technical_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_MARKET_TECHNICAL
        
        for keyword in performance_keywords:
            if keyword in user_input:
                return IntentType.ANALYSIS_MARKET_PERFORMANCE
        
        # 默认返回大盘分析
        return IntentType.ANALYSIS_MARKET
    
    def analyze_requirements(self, intent: IntentType, entities: Dict, user_input: str) -> Dict:
        """
        分析用户的具体需求
        
        Args:
            intent: 意图类型
            entities: 实体信息
            user_input: 用户输入文本
            
        Returns:
            Dict: 分析后的需求信息
        """
        requirements = {
            "intent": intent.value,
            "entities": entities,
            "user_input": user_input,
            "data_needed": [],
            "analysis_needed": [],
            "sentiment": self._analyze_sentiment(user_input)
        }
        
        # 根据意图确定需要的数据和分析类型
        if intent == IntentType.ANALYSIS_STOCK or intent == IntentType.STOCK_ANALYSIS:
            requirements["data_needed"] = ["stock_quote", "stock_history", "stock_news"]
            requirements["analysis_needed"] = ["fundamental", "technical", "valuation"]
        
        elif intent == IntentType.ANALYSIS_SECTOR or intent == IntentType.SECTOR_ANALYSIS:
            requirements["data_needed"] = ["sector_data", "sector_news"]
            requirements["analysis_needed"] = ["sector_performance", "sector_trend"]
        
        elif intent == IntentType.ANALYSIS_MARKET or intent == IntentType.MARKET_ANALYSIS:
            requirements["data_needed"] = ["market_overview", "index_data", "industry_data", "market_news"]
            requirements["analysis_needed"] = ["market_trend", "sector_rotation", "market_sentiment"]
        
        elif intent == IntentType.ANALYSIS_STOCK_FUNDAMENTAL:
            requirements["data_needed"] = ["stock_quote", "stock_fundamental", "stock_news"]
            requirements["analysis_needed"] = ["fundamental"]
        
        elif intent == IntentType.ANALYSIS_STOCK_TECHNICAL:
            requirements["data_needed"] = ["stock_quote", "stock_history", "stock_news"]
            requirements["analysis_needed"] = ["technical"]
        
        elif intent == IntentType.ANALYSIS_STOCK_VALUATION:
            requirements["data_needed"] = ["stock_quote", "stock_fundamental", "stock_news"]
            requirements["analysis_needed"] = ["valuation"]
        
        elif intent == IntentType.ANALYSIS_STOCK_PERFORMANCE:
            requirements["data_needed"] = ["stock_quote", "stock_history", "stock_news"]
            requirements["analysis_needed"] = ["performance"]
        
        elif intent == IntentType.ANALYSIS_SECTOR_FUNDAMENTAL:
            requirements["data_needed"] = ["sector_data", "sector_news"]
            requirements["analysis_needed"] = ["sector_fundamental"]
        
        elif intent == IntentType.ANALYSIS_SECTOR_TECHNICAL:
            requirements["data_needed"] = ["sector_data", "sector_news"]
            requirements["analysis_needed"] = ["sector_technical"]
        
        elif intent == IntentType.ANALYSIS_SECTOR_PERFORMANCE:
            requirements["data_needed"] = ["sector_data", "sector_news"]
            requirements["analysis_needed"] = ["sector_performance"]
        
        elif intent == IntentType.ANALYSIS_MARKET_FUNDAMENTAL:
            requirements["data_needed"] = ["market_overview", "index_data", "industry_data", "market_news"]
            requirements["analysis_needed"] = ["market_fundamental"]
        
        elif intent == IntentType.ANALYSIS_MARKET_TECHNICAL:
            requirements["data_needed"] = ["market_overview", "index_data", "industry_data", "market_news"]
            requirements["analysis_needed"] = ["market_technical"]
        
        elif intent == IntentType.ANALYSIS_MARKET_PERFORMANCE:
            requirements["data_needed"] = ["market_overview", "index_data", "industry_data", "market_news"]
            requirements["analysis_needed"] = ["market_performance"]
        
        elif intent == IntentType.QUERY_NEWS:
            requirements["data_needed"] = ["market_news"]
            if entities.get("stock_code") or entities.get("stock_name"):
                requirements["data_needed"] = ["stock_news"]
            elif entities.get("sector"):
                requirements["data_needed"] = ["sector_news"]
        
        elif intent == IntentType.QUERY_STOCK_INFO:
            requirements["data_needed"] = ["stock_quote", "stock_fundamental", "stock_news"]
        
        elif intent == IntentType.QUERY_SECTOR_INFO:
            requirements["data_needed"] = ["sector_data", "sector_news"]
        
        elif intent == IntentType.QUERY_MARKET_INFO:
            requirements["data_needed"] = ["market_overview", "index_data", "industry_data"]
        
        elif intent == IntentType.QUERY_STOCK_LIST:
            requirements["data_needed"] = ["industry_data"]
            requirements["analysis_needed"] = ["stock_selection"]
        
        elif intent == IntentType.RECOMMEND_STOCK:
            requirements["data_needed"] = ["industry_data", "stock_quote", "stock_news"]
            requirements["analysis_needed"] = ["stock_selection", "valuation"]
        
        elif intent == IntentType.RECOMMEND_SECTOR:
            requirements["data_needed"] = ["industry_data", "sector_news"]
            requirements["analysis_needed"] = ["sector_selection"]
        
        elif intent == IntentType.RECOMMEND_PORTFOLIO:
            requirements["data_needed"] = ["market_overview", "industry_data", "stock_quote", "stock_news"]
            requirements["analysis_needed"] = ["portfolio_construction"]
        
        elif intent == IntentType.GENERATE_REPORT:
            requirements["data_needed"] = ["market_overview", "index_data", "industry_data", "stock_quote", "stock_fundamental", "stock_news"]
            requirements["analysis_needed"] = ["comprehensive_analysis"]
        
        return requirements
    
    def _analyze_sentiment(self, user_input: str) -> str:
        """
        分析用户输入的情感倾向
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            str: 情感倾向（看涨、看跌、中性）
        """
        # 看涨关键词
        bullish_keywords = ["涨", "上涨", "反弹", "突破", "买入", "做多", "看好", "强劲", "上升", "牛市"]
        # 看跌关键词
        bearish_keywords = ["跌", "下跌", "回调", "跌破", "卖出", "做空", "看空", "疲软", "下降", "熊市"]
        
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in user_input)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in user_input)
        
        if bullish_count > bearish_count:
            return "看涨"
        elif bearish_count > bullish_count:
            return "看跌"
        else:
            return "中性"


# 全局意图识别器实例
intent_recognizer = IntentRecognizer()
