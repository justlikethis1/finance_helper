#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
互联网信息搜集模块
负责从互联网上搜索和获取相关的金融信息
"""

import logging
import requests
import json
import re
import time
from typing import Dict, List, Optional
from bs4 import BeautifulSoup
from functools import wraps

logger = logging.getLogger(__name__)


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


class InternetInfoAgent:
    """互联网信息搜集代理
    负责从互联网上搜索和获取相关的金融信息
    """
    
    def __init__(self):
        # 配置请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.5,en;q=0.3',
        })
        
        # 支持的搜索引擎和金融网站
        self.search_engines = {
            'sogou': self._sogou_search,
            'baidu': self._baidu_search
        }
        
        self.financial_sites = {
            'eastmoney': '东方财富网',
            'xueqiu': '雪球',
            'hexun': '和讯网',
            'ifeng': '凤凰财经',
            'wallstreetcn': '华尔街见闻'
        }
    
    @rate_limited(max_calls=5, period=60)  # 每分钟最多5次请求
    def search_internet(self, query: str, sources: List[str] = None, limit: int = 5) -> List[Dict]:
        """
        在互联网上搜索相关信息
        
        Args:
            query: 搜索查询
            sources: 搜索来源列表
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        if not sources:
            sources = ['sogou']  # 默认使用搜狗搜索
        
        all_results = []
        
        for source in sources:
            try:
                if source in self.search_engines:
                    results = self.search_engines[source](query, limit)
                    all_results.extend(results)
                else:
                    logger.warning(f"不支持的搜索来源: {source}")
            except Exception as e:
                logger.error(f"搜索失败 ({source}): {e}")
                continue
        
        return all_results
    
    def _sogou_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        使用搜狗搜索
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        url = f"https://www.sogou.com/web?query={query}&page=1&num={limit}"
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # 提取搜索结果
        for item in soup.select('.vrwrap')[:limit]:
            title_tag = item.select_one('.vrTitle a')
            if not title_tag:
                continue
            
            title = title_tag.get_text(strip=True)
            url = title_tag.get('href')
            summary_tag = item.select_one('.vrSummary')
            summary = summary_tag.get_text(strip=True) if summary_tag else ''
            
            # 提取来源信息
            source_tag = item.select_one('.siteinfo')
            source = source_tag.get_text(strip=True) if source_tag else '未知'
            
            results.append({
                'title': title,
                'url': url,
                'summary': summary,
                'source': source,
                'engine': 'sogou'
            })
        
        return results
    
    def _baidu_search(self, query: str, limit: int = 5) -> List[Dict]:
        """
        使用百度搜索
        
        Args:
            query: 搜索查询
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 搜索结果列表
        """
        url = f"https://www.baidu.com/s?wd={query}&pn=0&rn={limit}"
        response = self.session.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        results = []
        
        # 提取搜索结果
        for item in soup.select('.result')[:limit]:
            title_tag = item.select_one('h3 a')
            if not title_tag:
                continue
            
            title = title_tag.get_text(strip=True)
            url = title_tag.get('href')
            summary_tag = item.select_one('.c-abstract')
            summary = summary_tag.get_text(strip=True) if summary_tag else ''
            
            # 提取来源信息
            source_tag = item.select_one('.c-showurl')
            source = source_tag.get_text(strip=True) if source_tag else '未知'
            
            results.append({
                'title': title,
                'url': url,
                'summary': summary,
                'source': source,
                'engine': 'baidu'
            })
        
        return results
    
    def fetch_stock_news(self, stock_name: str, limit: int = 10) -> List[Dict]:
        """
        获取股票相关新闻
        
        Args:
            stock_name: 股票名称
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 新闻列表
        """
        query = f"{stock_name} 股票 新闻 最新"
        results = self.search_internet(query, limit=limit)
        
        # 过滤与股票相关的新闻
        stock_news = []
        for result in results:
            if stock_name in result['title'] or stock_name in result['summary']:
                stock_news.append(result)
        
        return stock_news[:limit]
    
    def fetch_sector_news(self, sector_name: str, limit: int = 10) -> List[Dict]:
        """
        获取板块相关新闻
        
        Args:
            sector_name: 板块名称
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 新闻列表
        """
        query = f"{sector_name} 板块 新闻 最新"
        results = self.search_internet(query, limit=limit)
        
        return results
    
    def fetch_market_news(self, limit: int = 10) -> List[Dict]:
        """
        获取市场相关新闻
        
        Args:
            limit: 返回结果数量限制
            
        Returns:
            List[Dict]: 新闻列表
        """
        query = "中国股市 最新消息 市场动态"
        results = self.search_internet(query, limit=limit)
        
        return results


# 创建单例实例
internet_info_agent = InternetInfoAgent()