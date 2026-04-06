from enum import Enum
from typing import List, Optional


class IntentType(Enum):
    """意图类型枚举"""
    # 根意图
    ROOT = "root"  # 根意图
    UNKNOWN = "unknown"  # 未知意图
    
    # 一级意图
    ANALYSIS = "analysis"  # 分析类意图
    QUERY = "query"  # 查询类意图
    RECOMMENDATION = "recommendation"  # 推荐类意图
    
    # 二级意图 - 分析类
    ANALYSIS_STOCK = "stock_analysis"  # 股票分析
    STOCK_ANALYSIS = "stock_analysis"  # 股票分析（别名）
    ANALYSIS_SECTOR = "sector_analysis"  # 板块分析
    SECTOR_ANALYSIS = "sector_analysis"  # 板块分析（别名）
    ANALYSIS_MARKET = "market_analysis"  # 大盘分析
    MARKET_ANALYSIS = "market_analysis"  # 大盘分析（别名）
    
    # 三级意图 - 股票分析
    ANALYSIS_STOCK_FUNDAMENTAL = "analysis_stock_fundamental"  # 股票基本面分析
    ANALYSIS_STOCK_TECHNICAL = "analysis_stock_technical"  # 股票技术面分析
    ANALYSIS_STOCK_VALUATION = "analysis_stock_valuation"  # 股票估值分析
    ANALYSIS_STOCK_PERFORMANCE = "analysis_stock_performance"  # 股票表现分析
    
    # 三级意图 - 板块分析
    ANALYSIS_SECTOR_FUNDAMENTAL = "analysis_sector_fundamental"  # 板块基本面分析
    ANALYSIS_SECTOR_TECHNICAL = "analysis_sector_technical"  # 板块技术面分析
    ANALYSIS_SECTOR_PERFORMANCE = "analysis_sector_performance"  # 板块表现分析
    
    # 三级意图 - 大盘分析
    ANALYSIS_MARKET_FUNDAMENTAL = "analysis_market_fundamental"  # 大盘基本面分析
    ANALYSIS_MARKET_TECHNICAL = "analysis_market_technical"  # 大盘技术面分析
    ANALYSIS_MARKET_PERFORMANCE = "analysis_market_performance"  # 大盘表现分析
    
    # 二级意图 - 查询类
    QUERY_NEWS = "news_query"  # 新闻查询
    QUERY_STOCK_LIST = "stock_list"  # 股票列表请求
    QUERY_STOCK_INFO = "stock_info_query"  # 股票信息查询
    QUERY_SECTOR_INFO = "sector_info_query"  # 板块信息查询
    QUERY_MARKET_INFO = "market_info_query"  # 大盘信息查询
    
    # 二级意图 - 推荐类
    RECOMMEND_STOCK = "stock_recommendation"  # 股票推荐
    RECOMMEND_SECTOR = "sector_recommendation"  # 板块推荐
    RECOMMEND_PORTFOLIO = "portfolio_recommendation"  # 投资组合推荐
    
    # 新增：报告生成意图
    GENERATE_REPORT = "generate_report"  # 生成投资报告
    
    @classmethod
    def get_hierarchy(cls, intent_type: 'IntentType') -> List['IntentType']:
        """
        获取意图的层次结构
        
        Args:
            intent_type: 意图类型
            
        Returns:
            List[IntentType]: 意图的层次结构，从根意图到当前意图
        """
        hierarchy_map = {
            # 股票分析相关
            cls.ANALYSIS_STOCK_FUNDAMENTAL: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_STOCK, cls.ANALYSIS_STOCK_FUNDAMENTAL],
            cls.ANALYSIS_STOCK_TECHNICAL: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_STOCK, cls.ANALYSIS_STOCK_TECHNICAL],
            cls.ANALYSIS_STOCK_VALUATION: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_STOCK, cls.ANALYSIS_STOCK_VALUATION],
            cls.ANALYSIS_STOCK_PERFORMANCE: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_STOCK, cls.ANALYSIS_STOCK_PERFORMANCE],
            
            # 板块分析相关
            cls.ANALYSIS_SECTOR_FUNDAMENTAL: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_SECTOR, cls.ANALYSIS_SECTOR_FUNDAMENTAL],
            cls.ANALYSIS_SECTOR_TECHNICAL: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_SECTOR, cls.ANALYSIS_SECTOR_TECHNICAL],
            cls.ANALYSIS_SECTOR_PERFORMANCE: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_SECTOR, cls.ANALYSIS_SECTOR_PERFORMANCE],
            
            # 大盘分析相关
            cls.ANALYSIS_MARKET_FUNDAMENTAL: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_MARKET, cls.ANALYSIS_MARKET_FUNDAMENTAL],
            cls.ANALYSIS_MARKET_TECHNICAL: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_MARKET, cls.ANALYSIS_MARKET_TECHNICAL],
            cls.ANALYSIS_MARKET_PERFORMANCE: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_MARKET, cls.ANALYSIS_MARKET_PERFORMANCE],
            
            # 查询相关
            cls.QUERY_NEWS: [cls.ROOT, cls.QUERY, cls.QUERY_NEWS],
            cls.QUERY_STOCK_LIST: [cls.ROOT, cls.QUERY, cls.QUERY_STOCK_LIST],
            cls.QUERY_STOCK_INFO: [cls.ROOT, cls.QUERY, cls.QUERY_STOCK_INFO],
            cls.QUERY_SECTOR_INFO: [cls.ROOT, cls.QUERY, cls.QUERY_SECTOR_INFO],
            cls.QUERY_MARKET_INFO: [cls.ROOT, cls.QUERY, cls.QUERY_MARKET_INFO],
            
            # 推荐相关
            cls.RECOMMEND_STOCK: [cls.ROOT, cls.RECOMMENDATION, cls.RECOMMEND_STOCK],
            cls.RECOMMEND_SECTOR: [cls.ROOT, cls.RECOMMENDATION, cls.RECOMMEND_SECTOR],
            cls.RECOMMEND_PORTFOLIO: [cls.ROOT, cls.RECOMMENDATION, cls.RECOMMEND_PORTFOLIO],
            
            # 报告生成
            cls.GENERATE_REPORT: [cls.ROOT, cls.GENERATE_REPORT],
            
            # 默认情况
            cls.ANALYSIS: [cls.ROOT, cls.ANALYSIS],
            cls.ANALYSIS_STOCK: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_STOCK],
            cls.ANALYSIS_SECTOR: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_SECTOR],
            cls.ANALYSIS_MARKET: [cls.ROOT, cls.ANALYSIS, cls.ANALYSIS_MARKET],
            cls.QUERY: [cls.ROOT, cls.QUERY],
            cls.RECOMMENDATION: [cls.ROOT, cls.RECOMMENDATION],
            cls.ROOT: [cls.ROOT],
            cls.UNKNOWN: [cls.ROOT, cls.UNKNOWN]
        }
        
        return hierarchy_map.get(intent_type, [cls.ROOT, cls.UNKNOWN])
    
    @classmethod
    def get_parent(cls, intent_type: 'IntentType') -> Optional['IntentType']:
        """
        获取意图的父意图
        
        Args:
            intent_type: 意图类型
            
        Returns:
            Optional[IntentType]: 父意图类型，如果是根意图则返回None
        """
        hierarchy = cls.get_hierarchy(intent_type)
        if len(hierarchy) > 1:
            return hierarchy[-2]
        return None
    
    @classmethod
    def is_analysis_intent(cls, intent):
        """判断是否为分析类意图"""
        if isinstance(intent, str):
            # 如果是字符串，转换为IntentType
            try:
                intent = cls(intent)
            except ValueError:
                return False
        hierarchy = cls.get_hierarchy(intent)
        return cls.ANALYSIS in hierarchy
    
    @classmethod
    def is_query_intent(cls, intent):
        """判断是否为查询类意图"""
        if isinstance(intent, str):
            # 如果是字符串，转换为IntentType
            try:
                intent = cls(intent)
            except ValueError:
                return False
        hierarchy = cls.get_hierarchy(intent)
        return cls.QUERY in hierarchy
    
    @classmethod
    def is_recommendation_intent(cls, intent):
        """判断是否为推荐类意图"""
        if isinstance(intent, str):
            # 如果是字符串，转换为IntentType
            try:
                intent = cls(intent)
            except ValueError:
                return False
        hierarchy = cls.get_hierarchy(intent)
        return cls.RECOMMENDATION in hierarchy
