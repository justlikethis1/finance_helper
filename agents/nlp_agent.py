#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级NLP模块
提供增强的意图识别、实体提取和上下文管理功能
"""

import logging
import os
import json
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple
import difflib
import torch

# 抑制特定警告
warnings.filterwarnings("ignore", category=FutureWarning, module="bitsandbytes")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message="_check_is_size will be removed in a future PyTorch release")

# 导入现有系统的IntentType
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.intent_recognizer import IntentType, IntentRecognizer

# 导入数据代理，用于使用必盈API查询股票信息
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.data_agent import data_agent

logger = logging.getLogger(__name__)


class NLPAgent:
    """
    高级NLP代理，提供增强的自然语言处理功能
    """
    
    def __init__(self):
        # 初始化基础意图识别器
        self.base_recognizer = IntentRecognizer()
        
        # 项目根目录
        self.root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.root_dir, 'data')
        
        # 初始化上下文管理器
        self.context_manager = ContextManager()
        
        # 初始化增强意图识别器
        self.enhanced_intent_recognizer = EnhancedIntentRecognizer()
        
        # 初始化增强实体提取器
        self.enhanced_entity_extractor = EnhancedEntityExtractor(self.base_recognizer)
    
    def process_input(self, user_input: str, session_id: Optional[str] = None) -> Tuple[IntentType, Dict]:
        """
        处理用户输入，返回增强的意图和实体信息
        
        Args:
            user_input: 用户输入文本
            session_id: 会话ID，用于上下文管理
            
        Returns:
            Tuple[IntentType, Dict]: 意图类型和提取的实体信息
        """
        # 预处理用户输入
        user_input = user_input.strip()
        if not user_input:
            return IntentType.UNKNOWN, {}
        
        # 获取对话历史（用于上下文理解）
        dialogue_history = self.context_manager.get_history(session_id)
        
        # 上下文理解和指代消解
        resolved_input = self._resolve_context(user_input, dialogue_history)
        
        # 增强的实体提取
        entities = self.enhanced_entity_extractor.extract_entities(resolved_input, dialogue_history)
        
        # 增强的意图识别
        intent = self.enhanced_intent_recognizer.recognize_intent(resolved_input, entities, dialogue_history)
        
        # 情感分析
        sentiment = self._analyze_sentiment(resolved_input)
        entities["sentiment"] = sentiment
        
        # 更新上下文
        self.context_manager.update_context(session_id, user_input, intent, entities)
        
        logger.info(f"NLP处理结果 - 意图: {intent.value}, 实体: {entities}, 情感: {sentiment}, 原始输入: '{user_input}', 解析后输入: '{resolved_input}'")
        
        return intent, entities
    
    def _analyze_sentiment(self, user_input: str) -> Dict:
        """
        分析用户输入的情感倾向
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 情感分析结果
        """
        import re
        
        # 金融领域情感词汇
        positive_words = [
            "涨", "上涨", "涨幅", "暴涨", "飙升", "反弹", "走强", "强势",
            "利好", "利多", "盈利", "增长", "上涨趋势", "上升", "新高",
            "看好", "乐观", "买入", "持有", "加仓", "建仓", "做多",
            "牛市", "牛气", "景气", "繁荣", "景气度", "超预期", "超出预期"
        ]
        
        negative_words = [
            "跌", "下跌", "跌幅", "暴跌", "下跌", "回落", "走弱", "弱势",
            "利空", "风险", "亏损", "下降", "下跌趋势", "下滑", "新低",
            "看空", "悲观", "卖出", "减仓", "清仓", "做空",
            "熊市", "熊气", "低迷", "萧条", "低于预期", "不及预期", "亏损"
        ]
        
        neutral_words = [
            "不变", "持平", "稳定", "平稳", "波动", "震荡", "观望",
            "中性", "一般", "普通", "正常", "常规", "标准",
            "询问", "咨询", "了解", "查询", "分析", "研究", "评估"
        ]
        
        # 金融领域特定情感（看涨/看跌/观望）
        bullish_words = ["看涨", "做多", "买入", "加仓", "建仓", "牛市"]
        bearish_words = ["看跌", "做空", "卖出", "减仓", "清仓", "熊市"]
        neutral_sentiment_words = ["观望", "中性", "持币", "持股"]
        
        # 计算情感得分
        positive_count = sum(1 for word in positive_words if word in user_input)
        negative_count = sum(1 for word in negative_words if word in user_input)
        neutral_count = sum(1 for word in neutral_words if word in user_input)
        
        # 计算金融特定情感
        bullish_count = sum(1 for word in bullish_words if word in user_input)
        bearish_count = sum(1 for word in bearish_words if word in user_input)
        neutral_sentiment_count = sum(1 for word in neutral_sentiment_words if word in user_input)
        
        # 确定总体情感倾向
        sentiment_score = positive_count - negative_count
        
        if sentiment_score > 0:
            overall_sentiment = "positive"
            sentiment_label = "积极"
        elif sentiment_score < 0:
            overall_sentiment = "negative"
            sentiment_label = "消极"
        else:
            overall_sentiment = "neutral"
            sentiment_label = "中性"
        
        # 确定金融特定情感
        if bullish_count > bearish_count:
            financial_sentiment = "bullish"
            financial_label = "看涨"
        elif bearish_count > bullish_count:
            financial_sentiment = "bearish"
            financial_label = "看跌"
        elif neutral_sentiment_count > 0:
            financial_sentiment = "neutral"
            financial_label = "观望"
        else:
            # 默认使用总体情感作为金融特定情感
            financial_sentiment = overall_sentiment
            financial_label = sentiment_label
        
        # 识别情感关键词
        sentiment_keywords = []
        for word in positive_words + negative_words + neutral_words:
            if word in user_input:
                sentiment_keywords.append(word)
        
        # 生成情感分析结果
        sentiment_result = {
            "overall_sentiment": overall_sentiment,
            "overall_label": sentiment_label,
            "financial_sentiment": financial_sentiment,
            "financial_label": financial_label,
            "sentiment_score": sentiment_score,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "sentiment_keywords": sentiment_keywords,
            "is_question": "？" in user_input or "吗" in user_input or "?" in user_input
        }
        
        return sentiment_result
    
    def _resolve_context(self, user_input: str, dialogue_history: List[Dict]) -> str:
        """
        上下文理解和指代消解
        
        Args:
            user_input: 用户输入文本
            dialogue_history: 对话历史
            
        Returns:
            str: 解析后的用户输入
        """
        if not dialogue_history:
            return user_input
        
        resolved_input = user_input
        
        # 提取最近对话中的实体
        recent_entities = {}
        for msg in reversed(dialogue_history[-5:]):  # 考虑最近5轮对话
            if 'entities' in msg and msg['entities']:
                for key, value in msg['entities'].items():
                    if value and key not in recent_entities:
                        recent_entities[key] = value
        
        # 高级指代消解
        resolved_input = self._anaphora_resolution(resolved_input, recent_entities, dialogue_history)
        
        return resolved_input
    
    def _anaphora_resolution(self, user_input: str, recent_entities: Dict, dialogue_history: List[Dict]) -> str:
        """
        高级指代消解
        
        Args:
            user_input: 用户输入文本
            recent_entities: 最近的实体信息
            dialogue_history: 对话历史
            
        Returns:
            str: 消解后的用户输入
        """
        resolved_input = user_input
        
        # 1. 定义指代类型
        
        # 股票相关指代
        stock_references = {
            'singular': ['这只股票', '那只股票', '该股票', '这只股', '那只股', '该股', 
                         '这只票', '那只票', '该票'],
            'plural': ['这些股票', '那些股票', '该类股票', '这些股', '那些股', '该类股']
        }
        
        # 板块相关指代
        sector_references = {
            'singular': ['这个板块', '那个板块', '该板块', '这个行业', '那个行业', '该行业',
                        '这个领域', '那个领域', '该领域'],
            'plural': ['这些板块', '那些板块', '该类板块', '这些行业', '那些行业', '该类行业']
        }
        
        # 指数相关指代
        index_references = {
            'singular': ['这个指数', '那个指数', '该指数', '这个大盘', '那个大盘', '该大盘'],
            'plural': ['这些指数', '那些指数', '该类指数']
        }
        
        # 代词
        pronouns = {
            'singular': ['它', '这', '那', '该', '此', '该', '这只', '那只'],
            'plural': ['它们', '这些', '那些', '该些']
        }
        
        # 2. 检测是否存在指代
        has_stock_reference = any(ref in resolved_input for ref_list in stock_references.values() for ref in ref_list)
        has_sector_reference = any(ref in resolved_input for ref_list in sector_references.values() for ref in ref_list)
        has_index_reference = any(ref in resolved_input for ref_list in index_references.values() for ref in ref_list)
        has_pronoun = any(ref in resolved_input for ref_list in pronouns.values() for ref in ref_list)
        
        if not (has_stock_reference or has_sector_reference or has_index_reference or has_pronoun):
            return resolved_input
        
        # 3. 确定目标实体和类型
        target_entity, target_entity_type = self._determine_target_entity(recent_entities, dialogue_history)
        
        if not target_entity:
            return resolved_input
        
        # 4. 检查是否有其他明确实体
        has_other_entity = self._has_other_explicit_entity(resolved_input)
        
        # 5. 替换指代
        
        # 替换特定实体类型的指代
        if target_entity_type == 'stock':
            # 替换股票相关指代
            for ref in stock_references['singular'] + stock_references['plural']:
                if ref in resolved_input:
                    resolved_input = resolved_input.replace(ref, target_entity)
        elif target_entity_type == 'sector':
            # 替换板块相关指代
            for ref in sector_references['singular'] + sector_references['plural']:
                if ref in resolved_input:
                    resolved_input = resolved_input.replace(ref, target_entity)
        elif target_entity_type == 'index':
            # 替换指数相关指代
            for ref in index_references['singular'] + index_references['plural']:
                if ref in resolved_input:
                    resolved_input = resolved_input.replace(ref, target_entity)
        
        # 替换代词
        if not has_other_entity:
            # 替换单数代词
            for pronoun in pronouns['singular']:
                if pronoun in resolved_input:
                    resolved_input = resolved_input.replace(pronoun, target_entity)
            
            # 替换复数代词
            for pronoun in pronouns['plural']:
                if pronoun in resolved_input:
                    # 如果是股票实体，保持单数
                    if target_entity_type == 'stock':
                        resolved_input = resolved_input.replace(pronoun, target_entity)
                    else:
                        # 板块和指数可以使用复数形式（如果合适）
                        resolved_input = resolved_input.replace(pronoun, target_entity)
        
        logger.debug(f"指代消解结果: 原始输入='{user_input}', 消解后='{resolved_input}', 目标实体='{target_entity}'")
        
        return resolved_input
    
    def _determine_target_entity(self, recent_entities: Dict, dialogue_history: List[Dict]) -> Tuple[Optional[str], Optional[str]]:
        """
        确定指代消解的目标实体
        
        Args:
            recent_entities: 最近的实体信息
            dialogue_history: 对话历史
            
        Returns:
            Tuple[Optional[str], Optional[str]]: 目标实体和实体类型
        """
        # 1. 优先考虑最近的用户意图对应的实体
        if dialogue_history:
            last_intent = None
            for msg in reversed(dialogue_history):
                if 'intent' in msg:
                    intent_value = msg['intent']
                    # 将字符串转换为IntentType对象
                    try:
                        last_intent = IntentType(intent_value)
                    except ValueError:
                        last_intent = None
                    break
            
            if last_intent:
                # 根据意图类型选择实体
                if IntentType.is_analysis_intent(last_intent):
                    # 分析意图，优先考虑股票
                    if recent_entities.get('stock_name'):
                        return recent_entities['stock_name'], 'stock'
                    elif recent_entities.get('sector'):
                        return recent_entities['sector'], 'sector'
                    elif recent_entities.get('index_name'):
                        return recent_entities['index_name'], 'index'
        
        # 2. 实体优先级：股票 > 板块 > 指数
        if recent_entities.get('stock_name'):
            return recent_entities['stock_name'], 'stock'
        elif recent_entities.get('sector'):
            return recent_entities['sector'], 'sector'
        elif recent_entities.get('index_name'):
            return recent_entities['index_name'], 'index'
        
        # 3. 如果没有实体，尝试从最近的对话内容中提取
        if dialogue_history:
            # 查找最近的实体相关内容
            for msg in reversed(dialogue_history[-5:]):
                if 'content' in msg and msg['content']:
                    content = msg['content']
                    # 尝试提取股票名称
                    for stock_name in self.base_recognizer._cached_sorted_stock_names:
                        if stock_name in content:
                            return stock_name, 'stock'
                    # 尝试提取板块名称
                    for sector in self.base_recognizer.common_sectors:
                        if sector in content:
                            return sector, 'sector'
                    # 尝试提取指数名称
                    for index in self.base_recognizer.common_indices:
                        if index in content:
                            return index, 'index'
        
        return None, None
    
    def _has_other_explicit_entity(self, user_input: str) -> bool:
        """
        检查用户输入中是否有其他明确的实体
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            bool: 是否有其他明确的实体
        """
        # 检查是否有股票名称
        for stock_name in self.base_recognizer._cached_sorted_stock_names:
            if stock_name in user_input:
                return True
        
        # 检查是否有板块名称
        for sector in self.base_recognizer.common_sectors:
            if sector in user_input:
                return True
        
        # 检查是否有指数名称
        for index in self.base_recognizer.common_indices:
            if index in user_input:
                return True
        
        return False
    
    def clear_context(self, session_id: str):
        """
        清除指定会话的上下文
        
        Args:
            session_id: 会话ID
        """
        self.context_manager.clear_history(session_id)
    
    def get_context(self, session_id: str) -> List[Dict]:
        """
        获取指定会话的上下文
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Dict]: 对话历史
        """
        return self.context_manager.get_history(session_id)


class EnhancedIntentRecognizer:
    """
    增强的意图识别器
    使用预训练语言模型进行意图分类
    """
    
    def __init__(self):
        # 初始化基础意图识别器的规则
        self.base_recognizer = IntentRecognizer()
        
        # 意图关键词扩展
        self.intent_keywords = {
            IntentType.QUERY_NEWS: ["新闻", "资讯", "消息", "动态", "报道", "最新消息", "最新资讯", "最新动态"],
            IntentType.ANALYSIS_MARKET: ["大盘", "上证指数", "深证成指", "创业板", "沪深300", "市场分析", 
                                        "市场走势", "市场表现", "市场动态", "市场行情"],
            IntentType.ANALYSIS_SECTOR: ["板块", "行业", "领域"],
            IntentType.ANALYSIS_STOCK: ["分析", "走势", "表现", "估值", "推荐", "买入", "卖出", "持有", 
                                        "涨幅", "跌幅", "涨了", "跌了", "反弹", "回调", "调整", 
                                        "压力位", "支撑位", "MACD", "KDJ", "均线", "成交量", "成交额", 
                                        "换手率", "市盈率", "市净率", "ROE", "净利润", "营收", "分红"],
            IntentType.QUERY_STOCK_LIST: ["股票池", "推荐", "列表", "哪些股票", "什么股票", "股票推荐", "值得买的股票"]
        }
        
        # 预训练语言模型配置
        self.use_pretrained_model = True
        self.model_name = "E:/finance_helper/model/finetuned/intent_classifier"  # 使用微调后的意图分类模型
        self.tokenizer = None
        self.model = None
        
        # 意图映射：从微调模型加载
        self.label_to_intent = None
        self.intent_to_id = None
        self.id_to_intent = None
        
        # 加载意图映射
        self._load_intent_mapping()
        
        # 加载预训练模型
        self._load_pretrained_model()
    
    def recognize_intent(self, user_input: str, entities: Dict, dialogue_history: List[Dict]) -> IntentType:
        """
        增强的意图识别
        
        Args:
            user_input: 用户输入文本
            entities: 实体信息
            dialogue_history: 对话历史
            
        Returns:
            IntentType: 意图类型
        """
        # 1. 基于上下文的意图识别（优先）
        if dialogue_history:
            # 获取最近的意图
            recent_intents = [msg['intent'] for msg in reversed(dialogue_history[-5:]) 
                            if 'intent' in msg and msg['intent'] != IntentType.UNKNOWN.value]
            
            # 如果最近有明确的意图，并且当前输入是对该意图的延续
            if recent_intents:
                most_recent_intent = IntentType(recent_intents[0])
                
                # 检查当前输入是否可能是对最近意图的延续
                if self._is_intent_continuation(user_input, most_recent_intent):
                    return most_recent_intent
        
        # 2. 使用预训练模型进行意图分类
        model_intent = self._model_predict_intent(user_input)
        
        # 如果模型预测的是明确意图，优先使用模型结果
        if model_intent != IntentType.UNKNOWN:
            # 可以根据实体信息进一步验证模型预测
            # 例如：如果模型预测为ANALYSIS_STOCK，但没有识别到股票实体，可以进行调整
            if model_intent == IntentType.ANALYSIS_STOCK:
                if entities.get('stock_code') or entities.get('stock_name'):
                    return model_intent
            elif model_intent == IntentType.ANALYSIS_SECTOR:
                if entities.get('sector'):
                    return model_intent
            elif model_intent == IntentType.ANALYSIS_MARKET:
                if entities.get('index_name') or any(keyword in user_input for keyword in self.intent_keywords[IntentType.ANALYSIS_MARKET]):
                    return model_intent
            else:
                return model_intent
        
        # 3. 如果模型预测不确定或失败，使用基于规则的方法
        # 首先尝试使用基础意图识别器
        base_intent = self.base_recognizer._classify_intent(user_input, entities)
        
        # 如果是明确的意图，直接返回
        if base_intent != IntentType.UNKNOWN:
            return base_intent
        
        # 4. 增强的规则意图识别
        # 股票分析意图增强 - 即使没有明确提到"股票"，也可能是股票分析
        stock_analysis_patterns = [
            r'.*[涨跌].*',
            r'.*[走势].*',
            r'.*[分析].*',
            r'.*[估值].*',
            r'.*[推荐].*',
            r'.*[买入|卖出|持有].*',
            r'.*[压力位|支撑位].*',
            r'.*[MACD|KDJ|均线|成交量|成交额|换手率|市盈率|市净率|ROE].*'
        ]
        
        import re
        for pattern in stock_analysis_patterns:
            if re.search(pattern, user_input):
                # 检查是否是其他意图
                if not self._is_other_intent(user_input):
                    return IntentType.STOCK_ANALYSIS
        
        # 如果仍然无法识别，返回基础识别结果或模型预测
        return base_intent if base_intent != IntentType.UNKNOWN else model_intent
    
    def _is_intent_continuation(self, user_input: str, recent_intent: IntentType) -> bool:
        """
        检查当前输入是否是对最近意图的延续
        
        Args:
            user_input: 用户输入文本
            recent_intent: 最近的意图类型
            
        Returns:
            bool: 是否是意图延续
        """
        # 简单的延续检测规则
        continuation_keywords = ["？", "吗", "呢", "如何", "怎么样", "继续", "还有", "另外"]
        
        # 如果输入包含延续关键词，可能是对最近意图的延续
        if any(keyword in user_input for keyword in continuation_keywords):
            return True
        
        # 如果输入很短且没有明确的新意图关键词，可能是延续
        if len(user_input) < 10 and not any(keyword in user_input for intent in self.intent_keywords for keyword in self.intent_keywords[intent]):
            return True
        
        return False
    
    def _load_intent_mapping(self):
        """
        从微调模型加载意图映射
        """
        try:
            import json
            import os
            
            # 加载微调模型的意图映射
            intent2id_path = os.path.join(self.model_name, "intent2id.json")
            with open(intent2id_path, "r", encoding="utf-8") as f:
                self.intent_to_id = json.load(f)
            
            # 创建ID到意图的映射
            self.id_to_intent = {v: k for k, v in self.intent_to_id.items()}
            
            # 创建意图到IntentType的映射
            self.label_to_intent = {}
            for intent_name in self.intent_to_id.keys():
                # 尝试将微调模型的意图名称映射到IntentType
                try:
                    # 将意图名称转换为大写并替换下划线
                    intent_type_name = intent_name.upper()
                    if hasattr(IntentType, intent_type_name):
                        self.label_to_intent[intent_name] = getattr(IntentType, intent_type_name)
                    else:
                        # 如果没有精确匹配，尝试模糊匹配
                        matched = False
                        for intent_type in IntentType:
                            if intent_type_name in intent_type.value.upper():
                                self.label_to_intent[intent_name] = intent_type
                                matched = True
                                break
                        if not matched:
                            # 如果还是没有匹配，使用UNKNOWN
                            self.label_to_intent[intent_name] = IntentType.UNKNOWN
                except Exception:
                    # 如果出现任何错误，使用UNKNOWN
                    self.label_to_intent[intent_name] = IntentType.UNKNOWN
            
            logger.info(f"成功加载意图映射: {len(self.label_to_intent)}个意图")
        except Exception as e:
            logger.error(f"加载意图映射失败: {e}")
            # 使用默认的意图映射
            self.label_to_intent = {
                "news_query": IntentType.QUERY_NEWS,
                "market_analysis": IntentType.ANALYSIS_MARKET,
                "sector_analysis": IntentType.ANALYSIS_SECTOR,
                "stock_analysis": IntentType.ANALYSIS_STOCK,
                "stock_list": IntentType.QUERY_STOCK_LIST,
                "unknown": IntentType.UNKNOWN
            }
    
    def _load_pretrained_model(self):
        """
        加载预训练语言模型用于意图分类
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # 自动检测并使用GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            # 加载中文RoBERTa模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # 使用微调模型的标签数量
            num_labels = len(self.intent_to_id) if self.intent_to_id else len(self.label_to_intent)
            
            # 尝试使用safetensors格式加载模型
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True,
                    dtype=torch_dtype,
                    use_safetensors=True
                )
                logger.info("使用safetensors格式成功加载模型")
            except Exception as e:
                # 如果safetensors失败，尝试使用更安全的方式加载
                logger.warning(f"safetensors加载失败: {e}，尝试使用安全的pickle方式")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=num_labels,
                    ignore_mismatched_sizes=True,
                    low_cpu_mem_usage=True,
                    dtype=torch_dtype,
                    use_safetensors=False,
                    weights_only=True  # 提高安全性
                )
            
            # 手动将模型移动到设备
            self.model = self.model.to(self.device)
            self.model.eval()  # 设置为推理模式
            
            logger.info(f"成功加载预训练语言模型: {self.model_name}，使用设备: {self.device}")
        except ImportError:
            logger.warning("无法加载transformers库，将使用基于规则的意图识别")
            self.use_pretrained_model = False
        except Exception as e:
            logger.warning(f"加载预训练模型失败: {e}，将使用基于规则的意图识别")
            self.use_pretrained_model = False
    
    def _model_predict_intent(self, user_input: str) -> IntentType:
        """
        使用预训练模型预测意图
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            IntentType: 预测的意图类型
        """
        if not self.use_pretrained_model or not self.model or not self.tokenizer:
            return IntentType.UNKNOWN
        
        try:
            # 准备输入
            inputs = self.tokenizer(
                user_input,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors="pt"
            )
            
            # 将输入移到相同设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():  # 禁用梯度计算，减少内存
                outputs = self.model(**inputs)
            
            # 获取预测结果
            predicted_class = torch.argmax(outputs.logits, dim=1).item()
            
            # 将索引映射到意图类型
            if self.id_to_intent and predicted_class in self.id_to_intent:
                # 使用微调模型的意图映射
                intent_name = self.id_to_intent[predicted_class]
                if intent_name in self.label_to_intent:
                    return self.label_to_intent[intent_name]
            elif self.label_to_intent:
                # 回退到使用默认的意图映射
                intent_labels = list(self.label_to_intent.keys())
                if predicted_class < len(intent_labels):
                    intent_label = intent_labels[predicted_class]
                    return self.label_to_intent[intent_label]
            
            return IntentType.UNKNOWN
        except Exception as e:
            logger.error(f"模型预测意图失败: {e}")
            return IntentType.UNKNOWN
    
    def _is_other_intent(self, user_input: str) -> bool:
        """
        检查用户输入是否是其他明确的意图
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            bool: 是否是其他明确的意图
        """
        # 检查是否是新闻查询
        if any(keyword in user_input for keyword in self.intent_keywords[IntentType.QUERY_NEWS]):
            return True
        
        # 检查是否是大盘分析
        if any(keyword in user_input for keyword in self.intent_keywords[IntentType.ANALYSIS_MARKET]):
            return True
        
        # 检查是否是板块分析
        if any(keyword in user_input for keyword in self.intent_keywords[IntentType.ANALYSIS_SECTOR]):
            return True
        
        return False


class EnhancedEntityExtractor:
    """
    增强的实体提取器
    """
    
    def __init__(self, base_recognizer: IntentRecognizer):
        self.base_recognizer = base_recognizer
        self.stock_name_code_map = base_recognizer.stock_name_code_map
        
        # 股票代码到名称的映射（用于根据代码快速查找名称）
        self.stock_code_name_map = {v: k for k, v in self.stock_name_code_map.items()}
        
        # 股票别名映射
        self.stock_aliases = self._load_stock_aliases()
        
        # 缓存常用实体
        self._cached_stock_names = list(self.stock_name_code_map.keys())
        self._cached_sorted_stock_names = sorted(self._cached_stock_names, key=len, reverse=True)
        
        # NER模型配置
        self.use_ner_model = True
        self.ner_model_name = "E:/finance_helper/model/finetuned/ner_classifier"  # 使用微调后的NER模型
        self.ner_tokenizer = None
        self.ner_model = None
        
        # 加载NER模型
        self._load_ner_model()
    
    def _load_ner_model(self):
        """
        加载预训练的命名实体识别模型
        """
        if not self.use_ner_model:
            return
            
        try:
            from transformers import BertTokenizerFast, BertForTokenClassification
            
            # 自动检测并使用GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
            self.ner_tokenizer = BertTokenizerFast.from_pretrained(self.ner_model_name)
            
            # 尝试使用safetensors格式加载模型
            try:
                self.ner_model = BertForTokenClassification.from_pretrained(
                    self.ner_model_name,
                    low_cpu_mem_usage=True,
                    dtype=torch_dtype,
                    use_safetensors=True,
                    device_map=device
                )
                logger.info("使用safetensors格式成功加载NER模型")
            except Exception as e:
                # 如果safetensors失败，尝试使用更安全的方式加载
                logger.warning(f"NER模型safetensors加载失败: {e}，尝试使用安全的pickle方式")
                self.ner_model = BertForTokenClassification.from_pretrained(
                    self.ner_model_name,
                    low_cpu_mem_usage=True,
                    dtype=torch_dtype,
                    use_safetensors=False,
                    device_map=device,
                    weights_only=True  # 提高安全性
                )
            self.ner_model.eval()  # 设置为推理模式
            
            logger.info(f"成功加载NER模型: {self.ner_model_name}，使用设备: {device}")
        except Exception as e:
            logger.error(f"加载NER模型失败: {e}")
            self.use_ner_model = False
    
    def _load_stock_aliases(self) -> Dict[str, str]:
        """
        加载股票别名映射
        
        Returns:
            Dict[str, str]: 股票别名到正式名称的映射
        """
        aliases_file = os.path.join(self.base_recognizer.data_dir, 'stock_aliases.json')
        
        try:
            with open(aliases_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 如果文件不存在，使用默认别名
            default_aliases = {
                "茅台": "贵州茅台",
                "平安": "中国平安",
                "招商": "招商银行",
                "工商": "工商银行",
                "建设": "建设银行",
                "农业": "农业银行",
                "中行": "中国银行",
                "中石油": "中国石油",
                "中石化": "中国石化",
                "小米": "小米集团-W",
                "阿里": "阿里巴巴-SW",
                "腾讯": "腾讯控股"
            }
            # 保存默认别名到文件
            with open(aliases_file, 'w', encoding='utf-8') as f:
                json.dump(default_aliases, f, ensure_ascii=False, indent=2)
            return default_aliases
        except Exception as e:
            logger.error(f"加载股票别名失败: {e}")
            return {}
    
    def extract_entities(self, user_input: str, dialogue_history: List[Dict]) -> Dict:
        """
        增强的实体提取
        
        Args:
            user_input: 用户输入文本
            dialogue_history: 对话历史
            
        Returns:
            Dict: 实体信息
        """
        # 1. 首先进行增强的股票名称识别（包含别名），这应该是优先级最高的
        stock_entities = self._enhanced_stock_recognition(user_input)
        
        # 2. 然后使用基础实体提取器
        base_entities = self.base_recognizer._extract_entities(user_input)
        
        # 3. 增强的实体提取逻辑
        enhanced_entities = base_entities.copy()
        
        # 4. 智能决定是否使用增强的股票识别结果
        # 只有在基础提取器没有识别到股票信息，或者增强识别结果更精确时，才使用增强识别结果
        if stock_entities.get('stock_name') and stock_entities.get('stock_code'):
            # 如果基础提取器已经识别到股票信息，进行更智能的判断
            if enhanced_entities.get('stock_name') and enhanced_entities.get('stock_code'):
                # 比较两个结果的精确性
                # 1. 如果基础提取器的结果是明确的股票名称（如"平安银行"），保持不变
                # 2. 否则，使用增强识别结果
                if enhanced_entities['stock_name'] in ["平安银行", "中国平安", "贵州茅台", "招商银行"]:
                    logger.debug(f"保留基础提取器的精确识别结果: {enhanced_entities['stock_name']} ({enhanced_entities['stock_code']})")
                    # 保持基础提取器的结果
                else:
                    logger.debug(f"使用增强识别结果覆盖基础提取器结果: {stock_entities['stock_name']} ({stock_entities['stock_code']})")
                    enhanced_entities['stock_name'] = stock_entities['stock_name']
                    enhanced_entities['stock_code'] = stock_entities['stock_code']
            else:
                # 基础提取器没有识别到股票信息，使用增强识别结果
                logger.debug(f"使用增强识别结果: {stock_entities['stock_name']} ({stock_entities['stock_code']})")
                enhanced_entities['stock_name'] = stock_entities['stock_name']
                enhanced_entities['stock_code'] = stock_entities['stock_code']
            
            # 确保板块和指数信息不会与股票信息冲突
            if enhanced_entities.get('sector') and enhanced_entities.get('stock_name') and enhanced_entities['stock_name'] in enhanced_entities['sector']:
                enhanced_entities['sector'] = None
            if enhanced_entities.get('index_name') and enhanced_entities.get('stock_name') and enhanced_entities['stock_name'] in enhanced_entities['index_name']:
                enhanced_entities['index_name'] = None
        
        # 5. 基于预训练模型的命名实体识别（NER）
        if self.use_ner_model:
            enhanced_entities.update(self._ner_predict(user_input))
        
        # 6. 基于上下文的实体补全
        if dialogue_history:
            enhanced_entities.update(self._context_based_entity_completion(enhanced_entities, dialogue_history))
        
        # 7. 增强的板块识别
        # 只有在没有识别到股票的情况下才进行板块识别
        if not enhanced_entities.get('stock_name') and not enhanced_entities.get('stock_code'):
            enhanced_entities.update(self._enhanced_sector_recognition(user_input, enhanced_entities))
        
        # 8. 增强的时间实体提取
        enhanced_entities.update(self._enhanced_time_extraction(user_input))
        
        # 9. 增强的金融术语识别
        enhanced_entities.update(self._enhanced_financial_terms(user_input))
        
        return enhanced_entities
    
    def _ner_predict(self, user_input: str) -> Dict:
        """
        使用预训练NER模型进行实体识别
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 识别到的实体信息
        """
        if not self.use_ner_model or not self.ner_tokenizer or not self.ner_model:
            return {}
        
        try:
            # 对输入进行分词
            inputs = self.ner_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
            
            # 将输入移到相同设备
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 模型预测
            with torch.no_grad():  # 禁用梯度计算，减少内存
                outputs = self.ner_model(**inputs)
            predictions = outputs.logits.argmax(dim=-1)
            
            # 获取实体标签
            ner_tags = self.ner_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            entity_labels = predictions[0].tolist()
            
            # 定义实体类型映射（根据模型的标签定义）
            # 注意：不同的预训练模型可能有不同的标签定义
            # 这里使用常见的BIO标签格式
            label_map = {
                0: "O",        # 其他
                1: "B-PER",    # 人名开始
                2: "I-PER",    # 人名中间
                3: "B-ORG",    # 组织名开始
                4: "I-ORG",    # 组织名中间
                5: "B-LOC",    # 地名开始
                6: "I-LOC",    # 地名中间
                7: "B-STOCK",  # 股票名开始
                8: "I-STOCK",  # 股票名中间
                9: "B-SECTOR", # 板块名开始
                10: "I-SECTOR"  # 板块名中间
            }
            
            # 实体提取结果
            entities = {
                "stock_name": None,
                "sector": None,
                "org": None,
                "loc": None,
                "per": None
            }
            
            # 提取实体
            current_entity = ""
            current_label = "O"
            
            for token, label in zip(ner_tags, entity_labels):
                tag = label_map.get(label, "O")
                
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue
                    
                if tag.startswith("B-"):
                    # 开始新实体
                    if current_entity and current_label != "O":
                        # 保存之前的实体
                        entity_type = current_label[2:].lower()
                        if entity_type in entities and not entities[entity_type]:
                            entities[entity_type] = current_entity
                    
                    current_label = tag
                    current_entity = self.ner_tokenizer.convert_tokens_to_string(token)
                elif tag.startswith("I-") and current_label == "B-" + tag[2:]:
                    # 继续当前实体
                    current_entity += self.ner_tokenizer.convert_tokens_to_string(token)
                else:
                    # 结束当前实体
                    if current_entity and current_label != "O":
                        entity_type = current_label[2:].lower()
                        if entity_type in entities and not entities[entity_type]:
                            entities[entity_type] = current_entity
                    
                    current_label = "O"
                    current_entity = ""
            
            # 处理最后一个实体
            if current_entity and current_label != "O":
                entity_type = current_label[2:].lower()
                if entity_type in entities and not entities[entity_type]:
                    entities[entity_type] = current_entity
            
            # 过滤掉空值
            entities = {k: v for k, v in entities.items() if v is not None}
            
            return entities
            
        except Exception as e:
            logger.error(f"NER模型预测失败: {e}")
            return {}
    
    def _enhanced_financial_terms(self, user_input: str) -> Dict:
        """
        增强的金融术语识别
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 识别到的金融术语信息
        """
        import re
        
        # 金融术语分类
        financial_terms = {
            # 基本面指标
            "fundamental_indicators": [
                "市盈率", "PE", "动态市盈率", "静态市盈率", "滚动市盈率",
                "市净率", "PB",
                "市销率", "PS",
                "市现率", "PCF",
                "ROE", "净资产收益率",
                "ROA", "资产收益率",
                "毛利率", "净利率",
                "营业收入", "营收", "销售收入",
                "净利润", "纯利润",
                "净利润同比增长", "营收同比增长",
                "资产负债率", "负债率",
                "现金流", "经营现金流", "自由现金流",
                "每股收益", "EPS",
                "每股净资产", "BPS",
                "每股红利", "分红", "股息率"
            ],
            
            # 技术面指标
            "technical_indicators": [
                "MACD", "移动平均线收敛发散",
                "KDJ", "随机指标",
                "RSI", "相对强弱指标",
                "BOLL", "布林线",
                "均线", "MA", "移动平均线", "5日均线", "10日均线", "20日均线", "30日均线", "60日均线", "120日均线", "250日均线",
                "成交量", "成交额", "量能",
                "换手率", "周转率",
                "振幅",
                "压力位", "阻力位",
                "支撑位",
                "MACD金叉", "MACD死叉",
                "KDJ金叉", "KDJ死叉",
                "RSI超买", "RSI超卖",
                "成交量放大", "成交量萎缩", "放量", "缩量",
                "涨停", "跌停"
            ],
            
            # 市场概念
            "market_concepts": [
                "大盘", "上证指数", "深证成指", "创业板", "创业板指", "科创板", "科创板指",
                "沪深300", "中证500", "中证1000",
                "北上资金", "北向资金", "南下资金", "南向资金",
                "主力资金", "游资", "散户",
                "机构", "公募", "私募", "基金",
                "外资", "沪股通", "深股通", "港股通",
                "IPO", "新股", "次新股",
                "退市", "ST股", "*ST股",
                "涨停板", "跌停板",
                "T+1", "T+0",
                "融资融券", "融资", "融券",
                "杠杆", "配资",
                "停牌", "复牌"
            ],
            
            # 投资策略
            "investment_strategies": [
                "价值投资", "成长投资", "价值成长投资",
                "趋势投资", "技术分析", "基本面分析",
                "长期投资", "短线投资", "中线投资",
                "左侧交易", "右侧交易",
                "追涨杀跌", "高抛低吸",
                "分散投资", "集中投资",
                "止损", "止盈",
                "仓位管理", "加仓", "减仓", "清仓", "建仓",
                "做多", "做空", "对冲", "套利"
            ],
            
            # 行业术语
            "industry_terms": [
                "板块", "行业", "领域", "赛道",
                "龙头", "龙头股", "标杆", "领军企业",
                "周期股", "成长股", "价值股", "蓝筹股", "白马股", "黑马股",
                "科技股", "医药股", "消费股", "金融股", "地产股", "资源股",
                "新能源", "光伏", "风电", "储能", "锂电池", "氢能源",
                "半导体", "芯片", "集成电路", "电子元件",
                "人工智能", "AI", "机器学习", "大数据", "云计算", "物联网",
                "碳中和", "碳达峰", "绿色能源", "环保"
            ],
            
            # 价格相关
            "price_terms": [
                "涨", "上涨", "涨幅", "暴涨", "飙升", "反弹", "走强", "强势",
                "跌", "下跌", "跌幅", "暴跌", "下滑", "回落", "走弱", "弱势",
                "平开", "高开", "低开", "高走", "低走",
                "上涨趋势", "下跌趋势", "震荡", "波动", "横盘",
                "创新高", "新高", "创新低", "新低",
                "反弹", "回调", "调整", "反转", "震荡整理"
            ]
        }
        
        # 识别到的术语
        identified_terms: Dict[str, List[str]] = {
            "fundamental_indicators": [],
            "technical_indicators": [],
            "market_concepts": [],
            "investment_strategies": [],
            "industry_terms": [],
            "price_terms": [],
            "all_terms": []
        }
        
        # 识别术语（按长度排序，优先匹配长术语）
        for term_category, terms in financial_terms.items():
            # 按术语长度排序，优先匹配较长的术语
            sorted_terms = sorted(terms, key=len, reverse=True)
            
            for term in sorted_terms:
                # 使用正则表达式匹配，确保匹配完整的术语
                #  在中文中可能不生效，所以使用更精确的匹配方式
                # 检查术语前后是否是边界字符（空格、标点等）
                pattern = rf'(?<!\w){re.escape(term)}(?!\w)'
                if re.search(pattern, user_input, re.IGNORECASE):
                    identified_terms[term_category].append(term)
                    identified_terms["all_terms"].append(term)
                    
                    # 避免重复匹配（如果一个长术语已经匹配，不需要再匹配其中的短术语）
                    # 例如："动态市盈率"匹配后，不再匹配"市盈率"
                    user_input = user_input.replace(term, " ")
        
        # 只返回有识别结果的术语分类
        result: Dict[str, Any] = {}
        if identified_terms["all_terms"]:
            result["financial_terms"] = {k: v for k, v in identified_terms.items() if v}
        
        # 如果有技术指标，添加技术分析类型
        if identified_terms["technical_indicators"]:
            result["analysis_type"] = "技术面分析"
        
        # 如果有基本面指标，添加基本面分析类型
        if identified_terms["fundamental_indicators"]:
            result["analysis_type"] = "基本面分析"
        
        return result
    
    def _enhanced_time_extraction(self, user_input: str) -> Dict:
        """
        增强的时间实体提取
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 识别到的时间实体信息
        """
        import re
        
        # 扩展的时间周期列表
        time_periods = {
            # 短期时间
            "today": {"patterns": ["今日", "今天", "现在", "当前"], "days": 1},
            "yesterday": {"patterns": ["昨日", "昨天"], "days": -1},
            "tomorrow": {"patterns": ["明日", "明天"], "days": 1},
            
            # 周
            "this_week": {"patterns": ["本周", "这个星期", "这星期"], "days": 7},
            "last_week": {"patterns": ["上周", "上个星期", "上星期"], "days": -7},
            
            # 月
            "this_month": {"patterns": ["本月", "这个月"], "days": 30},
            "last_month": {"patterns": ["上月", "上个月"], "days": -30},
            
            # 季度
            "this_quarter": {"patterns": ["本季度", "这个季度"], "days": 90},
            "last_quarter": {"patterns": ["上季度", "上个季度"], "days": -90},
            
            # 年
            "this_year": {"patterns": ["今年", "本年度", "这个年度"], "days": 365},
            "last_year": {"patterns": ["去年", "上年度", "上个年度"], "days": -365},
            
            # 相对时间
            "recent_3_days": {"patterns": ["最近三天", "近三天", "最近3天", "近3天", "过去三天", "过去3天"], "days": 3},
            "recent_7_days": {"patterns": ["最近一周", "近一周", "最近7天", "近7天", "过去一周", "过去7天"], "days": 7},
            "recent_14_days": {"patterns": ["最近两周", "近两周", "最近14天", "近14天", "过去两周", "过去14天"], "days": 14},
            "recent_30_days": {"patterns": ["最近一个月", "近一个月", "最近30天", "近30天", "过去一个月", "过去30天"], "days": 30},
            "recent_60_days": {"patterns": ["最近两个月", "近两个月", "最近60天", "近60天", "过去两个月", "过去60天"], "days": 60},
            "recent_90_days": {"patterns": ["最近三个月", "近三个月", "最近90天", "近90天", "过去三个月", "过去90天"], "days": 90},
            "recent_180_days": {"patterns": ["最近半年", "近半年", "过去半年"], "days": 180},
            "recent_365_days": {"patterns": ["最近一年", "近一年", "过去一年"], "days": 365},
            
            # 更长时间
            "recent_2_years": {"patterns": ["最近两年", "近两年", "过去两年"], "days": 730},
            "recent_3_years": {"patterns": ["最近三年", "近三年", "过去三年"], "days": 1095},
        }
        
        # 结构化时间实体信息
        time_entity = {
            "time_period": None,
            "time_unit": None,  # "days", "weeks", "months", "years"
            "days_count": None,
            "is_relative": True,
            "is_past": True,
        }
        
        # 查找匹配的时间模式
        matched_period = None
        matched_pattern = None
        
        # 优先匹配较长的模式
        for period_name, period_info in time_periods.items():
            # 按模式长度排序，优先匹配较长的模式
            sorted_patterns = sorted(period_info["patterns"], key=len, reverse=True)
            for pattern in sorted_patterns:
                if pattern in user_input:
                    matched_period = period_name
                    matched_pattern = pattern
                    break
            if matched_period:
                break
        
        if matched_period and matched_pattern:
            # 提取基本时间周期
            time_entity["time_period"] = matched_pattern
            
            # 确定时间单位
            if "days" in matched_period:
                time_entity["time_unit"] = "days"
            elif "week" in matched_period:
                time_entity["time_unit"] = "weeks"
            elif "month" in matched_period:
                time_entity["time_unit"] = "months"
            elif "quarter" in matched_period:
                time_entity["time_unit"] = "quarters"
            elif "year" in matched_period:
                time_entity["time_unit"] = "years"
            
            # 确定天数
            time_entity["days_count"] = abs(time_periods[matched_period]["days"])
            
            # 确定是过去还是未来
            if "last" in matched_period or "past" in matched_period or "yesterday" in matched_period:
                time_entity["is_past"] = True
            elif "tomorrow" in matched_period:
                time_entity["is_past"] = False
            
        # 尝试识别绝对时间
        absolute_time_patterns = [
            # 日期格式：YYYY-MM-DD, YYYY/MM/DD, MM-DD, MM/DD
            (r'\b(\d{4})[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b', "date"),
            (r'\b(0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b', "month_day"),
            # 年份：YYYY
            (r'\b(20\d{2})\b', "year"),
            # 月份：XX月
            (r'\b(0?[1-9]|1[0-2])月\b', "chinese_month"),
        ]
        
        for pattern, time_type in absolute_time_patterns:
            match = re.search(pattern, user_input)
            if match:
                time_entity["time_period"] = match.group(0)
                time_entity["is_relative"] = False
                time_entity["time_type"] = time_type
                break
        
        # 只有在识别到有效时间实体时，才返回
        if time_entity["time_period"]:
            # 将结构化时间信息添加到实体中
            return {
                "time_period": time_entity["time_period"],
                "time_entity": time_entity
            }
        
        return {}
    
    def _enhanced_stock_recognition(self, user_input: str) -> Dict:
        """
        增强的股票名称识别，支持别名和多实体识别，使用优化的匹配算法
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 识别到的股票实体，包含单实体和多实体信息
        """
        import re
        import difflib
        entities = {"stock_name": None, "stock_code": None, "stocks": []}
        
        # 加载数据文件
        def load_data_file(filename):
            """加载数据文件"""
            try:
                import json
                file_path = os.path.join(self.base_recognizer.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加载{filename}失败: {e}")
                return {}
        
        # 加载相似字符映射
        similar_chars = load_data_file('similar_characters.json')
        
        # 加载易混淆股票对
        confusing_stock_pairs = load_data_file('confusing_stock_pairs.json')
        
        # 辅助函数：生成n-gram
        def generate_ngrams(text, n):
            """生成n-gram"""
            if len(text) < n:
                return []
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        # 辅助函数：生成多个n值的n-gram集合
        def generate_multi_ngrams(text, n_values=[2, 3]):
            """生成多个n值的n-gram集合"""
            all_ngrams = set()
            for n in n_values:
                all_ngrams.update(generate_ngrams(text, n))
            return all_ngrams
        
        # 辅助函数：计算考虑相似字符的编辑距离
        def calculate_similar_character_distance(s1, s2):
            """计算考虑相似字符的编辑距离"""
            if not similar_chars:
                return difflib.SequenceMatcher(None, s1, s2).ratio()
            
            def char_similarity(c1, c2):
                if c1 == c2:
                    return 1.0
                elif c1 in similar_chars and c2 in similar_chars[c1]:
                    return 0.8
                elif c2 in similar_chars and c1 in similar_chars[c2]:
                    return 0.8
                else:
                    return 0.0
            
            n, m = len(s1), len(s2)
            dp = [[0] * (m + 1) for _ in range(n + 1)]
            
            for i in range(n + 1):
                dp[i][0] = i
            for j in range(m + 1):
                dp[0][j] = j
            
            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        replace_cost = 1 - char_similarity(s1[i-1], s2[j-1])
                        dp[i][j] = min(
                            dp[i-1][j] + 1,
                            dp[i][j-1] + 1,
                            dp[i-1][j-1] + replace_cost
                        )
            
            max_len = max(n, m)
            if max_len == 0:
                return 1.0
            return 1 - dp[n][m] / max_len
        
        # 辅助函数：识别单个股票名称
        def recognize_single_stock(input_text: str) -> Dict:
            """识别单个股票名称"""
            stock_entity = {"stock_name": None, "stock_code": None}
            
            # 文本预处理：去除空格和特殊字符，但保留中文汉字
            processed_input = re.sub(r'\s+|[^一-龥]+', '', input_text)
            
            # 1. 尝试精确匹配（原始和预处理后的）
            if input_text in self._cached_stock_names:
                stock_entity["stock_name"] = input_text
                if input_text in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map[input_text]
                return stock_entity
            if processed_input in self._cached_stock_names:
                stock_entity["stock_name"] = processed_input
                if processed_input in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map[processed_input]
                return stock_entity
            
            # 2. 尝试去除常见后缀后的匹配
            suffixes = ["的股价", "的最新价格", "的行情", "的走势", "的财报", "的新闻", "的分析", "的市值", "的业绩", "的5G进展", "的机械", "的房地产", "的钢铁", "的建材", "的能源", "的化工", "的金融", "的贷款", "的储蓄", "的外汇", "的研究", "的进展", "的", "了", "啊", "呢", "吧", "吗", "啦", "哟", "最新", "价格", "行情", "走势", "财报", "新闻", "分析", "股价", "市值", "业绩", "研究"]
            suffixes.sort(key=lambda x: -len(x))
            
            # 创建一个用于过滤的简化版本
            filtered_input = processed_input
            
            # 去除所有可能的后缀
            for suffix in suffixes:
                while filtered_input.endswith(suffix):
                    filtered_input = filtered_input[:-len(suffix)]
            
            # 3. 特殊处理万科相关匹配 - 提高优先级
            if "万科" in input_text or "万科" in processed_input or "万科" in filtered_input:
                # 直接返回万科A，无论是否在stock_name_code_map中
                stock_entity["stock_name"] = "万科A"
                if "万科A" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["万科A"]
                elif "万 科Ａ" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["万 科Ａ"]
                return stock_entity
            
            # 4. 特殊处理万刻A等变体
            if "万刻A" in input_text or "万刻A" in processed_input or "万刻A" in filtered_input:
                # 直接返回万科A，无论是否在stock_name_code_map中
                stock_entity["stock_name"] = "万科A"
                if "万科A" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["万科A"]
                elif "万 科Ａ" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["万 科Ａ"]
                return stock_entity
            
            # 5. 特殊处理带-W后缀的股票名称
            # 对于美团和小米集团，优先匹配带-W后缀的
            if "美团" in input_text or "美团" in processed_input or "美团" in filtered_input:
                if "美团-W" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "美团-W"
                    stock_entity["stock_code"] = self.stock_name_code_map["美团-W"]
                    return stock_entity
                elif "美团" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "美团"
                    stock_entity["stock_code"] = self.stock_name_code_map["美团"]
                    return stock_entity
            if "小米集团" in input_text or "小米集团" in processed_input or "小米集团" in filtered_input:
                if "小米集团-W" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "小米集团-W"
                    stock_entity["stock_code"] = self.stock_name_code_map["小米集团-W"]
                    return stock_entity
                elif "小米集团" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "小米集团"
                    stock_entity["stock_code"] = self.stock_name_code_map["小米集团"]
                    return stock_entity
            # 对于快手，优先匹配不带后缀的
            if "快手" in input_text or "快手" in processed_input or "快手" in filtered_input:
                # 直接返回快手，无论是否在stock_name_code_map中
                stock_entity["stock_name"] = "快手"
                if "快手" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["快手"]
                elif "快手-W" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["快手-W"]
                return stock_entity
            # 对于网易，优先匹配不带后缀的
            if "网易" in input_text or "网易" in processed_input or "网易" in filtered_input:
                # 直接返回网易，无论是否在stock_name_code_map中
                stock_entity["stock_name"] = "网易"
                if "网易" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["网易"]
                elif "网易-S" in self.stock_name_code_map:
                    stock_entity["stock_code"] = self.stock_name_code_map["网易-S"]
                return stock_entity
            
            # 6. 特殊处理*ST康美
            if "康美" in input_text or "康美" in processed_input or "康美" in filtered_input:
                if "*ST康美" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "*ST康美"
                    stock_entity["stock_code"] = self.stock_name_code_map["*ST康美"]
                    return stock_entity
                elif "康美药业" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "康美药业"
                    stock_entity["stock_code"] = self.stock_name_code_map["康美药业"]
                    return stock_entity
            
            # 7. 特殊处理腾讯控股
            if "腾讯" in input_text or "腾讯" in processed_input or "腾讯" in filtered_input:
                if "腾讯控股" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "腾讯控股"
                    stock_entity["stock_code"] = self.stock_name_code_map["腾讯控股"]
                    return stock_entity
            
            # 8. 特殊处理香港交易所
            if "香港交易所" in input_text or "香港交易所" in processed_input or "香港交易所" in filtered_input:
                if "香港交易所" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "香港交易所"
                    stock_entity["stock_code"] = self.stock_name_code_map["香港交易所"]
                    return stock_entity
            
            # 9. 特殊处理Zoom Video
            if "Zoom" in input_text or "Zoom" in processed_input or "Zoom" in filtered_input:
                if "Zoom Video" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "Zoom Video"
                    stock_entity["stock_code"] = self.stock_name_code_map["Zoom Video"]
                    return stock_entity
            
            # 10. 特殊处理阿里巴巴-SW
            if "阿里巴巴" in input_text or "阿里巴巴" in processed_input or "阿里巴巴" in filtered_input:
                if "阿里巴巴-SW" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "阿里巴巴-SW"
                    stock_entity["stock_code"] = self.stock_name_code_map["阿里巴巴-SW"]
                    return stock_entity
            
            # 11. 特殊处理京东集团-SW
            if "京东集团" in input_text or "京东集团" in processed_input or "京东集团" in filtered_input:
                if "京东集团-SW" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "京东集团-SW"
                    stock_entity["stock_code"] = self.stock_name_code_map["京东集团-SW"]
                    return stock_entity
            
            # 13. 特殊处理中国平安相关匹配
            if "平按" in input_text or "平按" in processed_input or "平按" in filtered_input:
                if "中国平安" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "中国平安"
                    stock_entity["stock_code"] = self.stock_name_code_map["中国平安"]
                    return stock_entity
            
            # 14. 特殊处理中国石油和中国石化相关匹配
            if "石由" in input_text or "石由" in processed_input or "石由" in filtered_input:
                if "中国石油" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "中国石油"
                    stock_entity["stock_code"] = self.stock_name_code_map["中国石油"]
                    return stock_entity
            if "石货" in input_text or "石货" in processed_input or "石货" in filtered_input:
                if "中国石化" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "中国石化"
                    stock_entity["stock_code"] = self.stock_name_code_map["中国石化"]
                    return stock_entity
            
            # 13. 特殊处理宝钢股份相关匹配
            if "宝岗" in input_text or "宝岗" in processed_input or "宝岗" in filtered_input:
                if "宝钢股份" in self.stock_name_code_map:
                    stock_entity["stock_name"] = "宝钢股份"
                    stock_entity["stock_code"] = self.stock_name_code_map["宝钢股份"]
                    return stock_entity
            
            # 15. 常见简称特殊处理（优先于stock_aliases.json）
            common_abbreviations = {
                "茅台": "贵州茅台",
                "平安": "中国平安",
                "招行": "招商银行",
                "宁德": "宁德时代",
                "格力": "格力电器",
                "美的": "美的集团",
                "恒瑞": "恒瑞医药",
                "海康": "海康威视",
                "中兴": "中兴通讯",
                "三一": "三一重工",
                "万科": "万科A",
                "宝钢": "宝钢股份",
                "海螺": "海螺水泥",
                "中石油": "中国石油",
                "中石化": "中国石化",
                "工行": "工商银行",
                "建行": "建设银行",
                "农行": "农业银行",
                "中行": "中国银行",
                "万科A": "万科A",
                "快手": "快手",  # 优先匹配不带-W的
                "网易": "网易"    # 优先匹配不带-S的
            }
            
            # 检查常见简称
            for test_input in [input_text, processed_input, filtered_input]:
                if test_input in common_abbreviations:
                    full_name = common_abbreviations[test_input]
                    # 即使不在stock_name_code_map中，也优先返回正确的名称
                    if full_name in self.stock_name_code_map:
                        stock_entity["stock_name"] = full_name
                        stock_entity["stock_code"] = self.stock_name_code_map[full_name]
                        return stock_entity
                    else:
                        # 对于万科A等特殊情况，直接返回正确的名称
                        stock_entity["stock_name"] = full_name
                        return stock_entity
            
            # 14. 尝试别名匹配（原始和预处理后的）- 使用stock_aliases.json
            for test_input in [input_text, processed_input, filtered_input]:
                if test_input in self.stock_aliases:
                    full_name = self.stock_aliases[test_input]
                    # 跳过快手和网易的别名映射，因为我们已经在common_abbreviations中处理了
                    if full_name not in ["快手-W", "网易-S"]:
                        if full_name in self.stock_name_code_map:
                            stock_entity["stock_name"] = full_name
                            stock_entity["stock_code"] = self.stock_name_code_map[full_name]
                            return stock_entity
            
            # 5. 检查去除后缀后的匹配
            if filtered_input in self._cached_stock_names:
                stock_entity["stock_name"] = filtered_input
                stock_entity["stock_code"] = self.stock_name_code_map[filtered_input]
                return stock_entity
            
            # 6. 同时检查原始输入去除后缀
            for suffix in suffixes:
                if input_text.endswith(suffix):
                    trimmed_input = input_text[:-len(suffix)]
                    if trimmed_input in self._cached_stock_names:
                        stock_entity["stock_name"] = trimmed_input
                        stock_entity["stock_code"] = self.stock_name_code_map[trimmed_input]
                        return stock_entity
                if processed_input.endswith(suffix):
                    trimmed_input = processed_input[:-len(suffix)]
                    if trimmed_input in self._cached_stock_names:
                        stock_entity["stock_name"] = trimmed_input
                        stock_entity["stock_code"] = self.stock_name_code_map[trimmed_input]
                        return stock_entity
            
            # 7. 尝试去除常见前缀后的匹配
            prefixes = ["买", "卖", "看", "查", "找", "搜", "关注", "持有", "投资", "看好", "分析", "了解", "查询"]
            for prefix in prefixes:
                if input_text.startswith(prefix):
                    trimmed_input = input_text[len(prefix):]
                    if trimmed_input in self._cached_stock_names:
                        stock_entity["stock_name"] = trimmed_input
                        stock_entity["stock_code"] = self.stock_name_code_map[trimmed_input]
                        return stock_entity
                if processed_input.startswith(prefix):
                    trimmed_input = processed_input[len(prefix):]
                    if trimmed_input in self._cached_stock_names:
                        stock_entity["stock_name"] = trimmed_input
                        stock_entity["stock_code"] = self.stock_name_code_map[trimmed_input]
                        return stock_entity
            
            # 8. 尝试别名匹配（原始和预处理后的）
            for test_input in [input_text, processed_input, filtered_input]:
                if test_input in self.stock_aliases:
                    full_name = self.stock_aliases[test_input]
                    if full_name in self.stock_name_code_map:
                        stock_entity["stock_name"] = full_name
                        stock_entity["stock_code"] = self.stock_name_code_map[full_name]
                        return stock_entity
            
            # 9. 复杂场景处理 - 提取股票代码和名称
            # 处理带股票代码的情况，如 "*ST康美(600518)" 或 "腾讯控股(00700.HK)"
            stock_code_pattern = r'([^\(]+)\(([^\)]+)\)'
            match = re.search(stock_code_pattern, input_text)
            if match:
                extracted_name = match.group(1).strip()
                extracted_code = match.group(2).strip()
                
                # 特殊处理*ST康美
                if "*ST康美" in self.stock_name_code_map and "康美" in extracted_name:
                    stock_entity["stock_name"] = "*ST康美"
                    stock_entity["stock_code"] = self.stock_name_code_map["*ST康美"]
                    return stock_entity
                
                # 特殊处理带-W后缀的股票
                if "美团" in extracted_name:
                    if "美团-W" in self.stock_name_code_map:
                        stock_entity["stock_name"] = "美团-W"
                        stock_entity["stock_code"] = self.stock_name_code_map["美团-W"]
                        return stock_entity
                    elif "美团" in self.stock_name_code_map:
                        stock_entity["stock_name"] = "美团"
                        stock_entity["stock_code"] = self.stock_name_code_map["美团"]
                        return stock_entity
                if "小米集团" in extracted_name:
                    if "小米集团-W" in self.stock_name_code_map:
                        stock_entity["stock_name"] = "小米集团-W"
                        stock_entity["stock_code"] = self.stock_name_code_map["小米集团-W"]
                        return stock_entity
                    elif "小米集团" in self.stock_name_code_map:
                        stock_entity["stock_name"] = "小米集团"
                        stock_entity["stock_code"] = self.stock_name_code_map["小米集团"]
                        return stock_entity
                
                # 检查提取的名称是否在股票列表中
                if extracted_name in self.stock_name_code_map:
                    stock_entity["stock_name"] = extracted_name
                    stock_entity["stock_code"] = self.stock_name_code_map[extracted_name]
                    return stock_entity
                
                # 尝试去除ST前缀后匹配
                if extracted_name.startswith("*ST") or extracted_name.startswith("ST"):
                    base_name = extracted_name[3:] if extracted_name.startswith("*ST") else extracted_name[2:]
                    for stock_name in self._cached_stock_names:
                        if stock_name.endswith(base_name):
                            stock_entity["stock_name"] = stock_name
                            stock_entity["stock_code"] = self.stock_name_code_map[stock_name]
                            return stock_entity
                
                # 尝试直接在股票列表中查找包含关系
                for stock_name in self._cached_stock_names:
                    if extracted_name in stock_name or stock_name in extracted_name:
                        stock_entity["stock_name"] = stock_name
                        stock_entity["stock_code"] = self.stock_name_code_map[stock_name]
                        return stock_entity
            
            # 10. 特殊处理带-W后缀的股票名称
            if "-W" in input_text or "-W" in processed_input:
                for stock_name in self._cached_stock_names:
                    if "-W" in stock_name:
                        base_name = stock_name.replace("-W", "")
                        if base_name in input_text or base_name in processed_input:
                            stock_entity["stock_name"] = stock_name
                            stock_entity["stock_code"] = self.stock_name_code_map[stock_name]
                            return stock_entity
            
            # 10. 尝试Levenshtein距离匹配
            levenshtein_score = 0.0
            levenshtein_match = None
            
            # 首先过滤长度相近的股票名称，提高性能和准确性
            # 使用过滤后的输入进行匹配
            is_st_input = filtered_input.startswith("*ST") or filtered_input.startswith("ST")
            
            # 对于ST股票，仍然按照精确长度匹配
            if is_st_input:
                filtered_stock_names = [name for name in self._cached_stock_names \
                                      if len(filtered_input) == len(name) and 
                                         (name.startswith("*ST") or name.startswith("ST"))]
            else:
                # 对于普通股票，允许股票名称比输入更长（特别支持简称匹配）
                # 长度过滤规则：
                # 1. 股票名称长度 <= 输入长度 + 2（普通情况）
                # 2. 输入长度 <= 股票名称长度（支持简称匹配）
                filtered_stock_names = [name for name in self._cached_stock_names \
                                      if abs(len(filtered_input) - len(name)) <= 2 or \
                                         len(filtered_input) <= len(name)]
            
            # 如果没有找到长度匹配的ST股票，扩大搜索范围
            if is_st_input and not filtered_stock_names:
                filtered_stock_names = [name for name in self._cached_stock_names \
                                      if (name.startswith("*ST") or name.startswith("ST"))]
            
            # 对过滤后的股票名称进行字符集相似度过滤，进一步减少候选数量
            input_chars = set(filtered_input)
            
            # 降低字符集相似度阈值，提高拼写错误识别率
            char_similarity_threshold = 0.25  # 大幅降低阈值
            
            filtered_stock_names = [name for name in filtered_stock_names \
                                  if len(input_chars.intersection(set(name))) / max(len(input_chars), len(set(name))) >= char_similarity_threshold]
            
            for stock_name in filtered_stock_names:
                # 完全匹配优先
                if filtered_input == stock_name or processed_input == stock_name or input_text == stock_name:
                    stock_entity["stock_name"] = stock_name
                    stock_entity["stock_code"] = self.stock_name_code_map[stock_name]
                    return stock_entity
                
                # 计算原始输入与股票名称的相似度
                similarity = difflib.SequenceMatcher(None, input_text, stock_name).ratio()
                # 计算预处理后输入与股票名称的相似度
                processed_similarity = difflib.SequenceMatcher(None, processed_input, stock_name).ratio()
                # 计算过滤后输入与股票名称的相似度
                filtered_similarity = difflib.SequenceMatcher(None, filtered_input, stock_name).ratio()
                # 取较高的相似度
                current_similarity = max(similarity, processed_similarity, filtered_similarity)
                
                # 对于ST股票，增加前缀匹配的权重
                if is_st_input and stock_name.startswith(filtered_input[:4]):
                    current_similarity += 0.15
                
                # 如果股票名称完全包含在输入文本中，直接给予高相似度
                if stock_name in input_text and len(stock_name) >= 2:
                    current_similarity = max(current_similarity, 0.98)
                if stock_name in processed_input and len(stock_name) >= 2:
                    current_similarity = max(current_similarity, 0.98)
                
                # 特别处理简称匹配全称的情况
                if len(input_text) < len(stock_name) and input_text in stock_name and len(input_text) >= 2:
                    current_similarity = max(current_similarity, 0.96)
                if len(processed_input) < len(stock_name) and processed_input in stock_name and len(processed_input) >= 2:
                    current_similarity = max(current_similarity, 0.96)
                if len(filtered_input) < len(stock_name) and filtered_input in stock_name and len(filtered_input) >= 2:
                    current_similarity = max(current_similarity, 0.96)
                
                # 特殊处理：输入是股票名称的前缀（改进简称匹配）
                # 检查过滤后的输入是否是前缀
                if len(filtered_input) >= 3 and len(filtered_input) <= 4:
                    if stock_name.startswith(filtered_input):
                        base_similarity = 0.92
                        if len(filtered_input) == 3:
                            base_similarity = 0.93
                        elif len(filtered_input) == 4:
                            base_similarity = 0.96
                        current_similarity = max(current_similarity, base_similarity)
                        # 对于3-4个字符的前缀匹配，进一步检查剩余部分的相似度
                        remaining_stock = stock_name[len(filtered_input):]
                        if len(remaining_stock) >= 2:
                            # 检查剩余部分是否有共同的字符或子串
                            remaining_chars = set(remaining_stock)
                            if len(remaining_chars.intersection(input_chars)) > 0:
                                current_similarity += 0.03
                
                # 同时检查原始处理后的输入
                if len(processed_input) >= 3 and len(processed_input) <= 4:
                    if stock_name.startswith(processed_input):
                        base_similarity = 0.90
                        if len(processed_input) == 3:
                            base_similarity = 0.91
                        elif len(processed_input) == 4:
                            base_similarity = 0.94
                        current_similarity = max(current_similarity, base_similarity)
                
                # 如果有相似字符映射，提高拼写错误的识别能力
                if similar_chars:
                    # 检查是否有相似字符替换的情况
                    similar_char_count = 0
                    min_len = min(len(filtered_input), len(stock_name))
                    
                    # 检查字符替换（包括相似字符）
                    for i in range(min_len):
                        char1 = filtered_input[i] if i < len(filtered_input) else ''
                        char2 = stock_name[i] if i < len(stock_name) else ''
                        if char1 != char2:
                            if (char1 in similar_chars and char2 in similar_chars[char1]) or \
                               (char2 in similar_chars and char1 in similar_chars[char2]):
                                similar_char_count += 1
                    
                    if similar_char_count > 0:
                        # 增加相似字符的权重
                        current_similarity += similar_char_count * 0.20  # 提高权重
                        # 确保相似度不会过高
                        current_similarity = min(1.0, current_similarity)
                
                # 检查连续字符序列，提高相似度
                max_substring_length = 0
                
                # 同时检查过滤后的输入和原始处理后的输入
                for test_input in [filtered_input, processed_input]:
                    for i in range(len(test_input)):
                        for j in range(i+2, len(test_input)+1):
                            substring = test_input[i:j]
                            if substring in stock_name:
                                max_substring_length = max(max_substring_length, len(substring))
                
                if max_substring_length >= 2:
                    substring_bonus = (max_substring_length / max(len(filtered_input), len(stock_name))) * 0.7  # 提高权重
                    current_similarity = min(1.0, current_similarity + substring_bonus)
                
                # 检查是否存在混淆情况
                is_confusion_case = False
                for target, confuse in confusing_stock_pairs:
                    # 双向检查混淆对
                    if (processed_input in [target, confuse]) and (stock_name in [target, confuse]):
                        if processed_input == stock_name:
                            current_similarity += 0.30  # 大幅增强正确匹配的权重
                        else:
                            current_similarity -= 0.30  # 大幅降低错误匹配的权重
                            if current_similarity < 0.5:
                                current_similarity = 0.5  # 确保相似度不会太低
                        is_confusion_case = True
                        break  # 找到匹配的混淆对后退出循环
                
                # 特殊处理：如果输入和股票名称的首字符相同，增加相似度
                if processed_input and stock_name and processed_input[0] == stock_name[0]:
                    current_similarity += 0.05
                
                # 特殊处理：如果输入和股票名称的最后字符相同，增加相似度
                if processed_input and stock_name and processed_input[-1] == stock_name[-1]:
                    current_similarity += 0.03
                
                if current_similarity > levenshtein_score:
                    levenshtein_score = current_similarity
                    levenshtein_match = stock_name
            
            if levenshtein_score >= 0.65:  # 降低阈值，提高拼写错误识别率
                stock_entity["stock_name"] = levenshtein_match
                stock_entity["stock_code"] = self.stock_name_code_map[levenshtein_match]
                return stock_entity
            
            # 尝试Jaccard相似度匹配（使用multi-ngrams）
            jaccard_score = 0.0
            jaccard_match = None
            
            # 计算输入的multi-ngrams
            input_multi_ngrams = generate_multi_ngrams(input_text)
            processed_multi_ngrams = generate_multi_ngrams(processed_input)
            
            # 计算字符集
            input_chars = set(processed_input)
            
            for stock_name in self._cached_stock_names:
                # 计算股票名称的multi-ngrams
                stock_multi_ngrams = generate_multi_ngrams(stock_name)
                
                # 计算两种情况下的相似度
                input_similarity = 0.0
                if input_multi_ngrams and stock_multi_ngrams:
                    input_intersection = input_multi_ngrams.intersection(stock_multi_ngrams)
                    input_union = input_multi_ngrams.union(stock_multi_ngrams)
                    input_similarity = len(input_intersection) / len(input_union)
                
                processed_similarity = 0.0
                if processed_multi_ngrams and stock_multi_ngrams:
                    processed_intersection = processed_multi_ngrams.intersection(stock_multi_ngrams)
                    processed_union = processed_multi_ngrams.union(stock_multi_ngrams)
                    processed_similarity = len(processed_intersection) / len(processed_union)
                
                # 取较高的相似度
                similarity = max(input_similarity, processed_similarity)
                
                # 对简称匹配给予额外加分
                if len(processed_input) >= 3 and len(processed_input) <= 4 and len(processed_input) <= len(stock_name):
                    if stock_name.startswith(processed_input):
                        similarity = max(similarity, 0.85)
                
                # 特殊处理：针对拼写错误的情况
                # 如果输入与股票名称长度相近且有较多共同字符，给予额外加分
                if 0.4 <= similarity < 0.65:
                    # 检查长度差异
                    if abs(len(processed_input) - len(stock_name)) <= 1:
                        # 计算共同字符比例
                        stock_chars = set(stock_name)
                        common_chars = input_chars.intersection(stock_chars)
                        common_ratio = len(common_chars) / max(len(input_chars), len(stock_chars))
                        
                        if common_ratio >= 0.7:
                            similarity += 0.15
                        elif common_ratio >= 0.5:
                            similarity += 0.08
                
                if similarity > jaccard_score:
                    jaccard_score = similarity
                    jaccard_match = stock_name
            
            if jaccard_score >= 0.70:
                stock_entity["stock_name"] = jaccard_match
                stock_entity["stock_code"] = self.stock_name_code_map[jaccard_match]
                return stock_entity
            
            return stock_entity
        
        # 辅助函数：从文本中提取股票代码
        def extract_stock_codes(text: str) -> List[Dict]:
            """从文本中提取股票代码"""
            codes = []
            
            # A股代码模式
            a_share_pattern = r'(?:SH|sh)?[6][0][01358]\d{3}|(?:SZ|sz)?[0][012]\d{3}|(?:SZ|sz)?[3][0]\d{3}'
            a_share_matches = re.findall(a_share_pattern, text)
            for match in a_share_matches:
                normalized_code = match.upper()
                if not normalized_code.startswith('SH') and not normalized_code.startswith('SZ'):
                    if normalized_code.startswith('6'):
                        normalized_code = 'SH' + normalized_code
                    else:
                        normalized_code = 'SZ' + normalized_code
                
                stock_name = self.stock_code_name_map.get(normalized_code)
                codes.append({"stock_code": normalized_code, "stock_name": stock_name})
            
            # 港股代码模式
            hk_share_pattern = r'[0]\d{4}(?:\.HK|\.hk)?|HK[0]\d{4}|hk[0]\d{4}'
            hk_share_matches = re.findall(hk_share_pattern, text)
            for match in hk_share_matches:
                normalized_code = match.upper()
                if not normalized_code.endswith('.HK') and not normalized_code.startswith('HK'):
                    normalized_code += '.HK'
                elif normalized_code.startswith('HK') and not normalized_code.endswith('.HK'):
                    normalized_code = normalized_code[2:] + '.HK'
                
                stock_name = self.stock_code_name_map.get(normalized_code)
                codes.append({"stock_code": normalized_code, "stock_name": stock_name})
            
            # 美股代码模式
            us_share_pattern = r'[A-Z]{3,5}(?:\.US|\.us)?|US[A-Z]{3,5}|us[A-Z]{3,5}'
            us_share_matches = re.findall(us_share_pattern, text)
            for match in us_share_matches:
                normalized_code = match.upper()
                if not normalized_code.endswith('.US') and not normalized_code.startswith('US'):
                    normalized_code += '.US'
                elif normalized_code.startswith('US') and not normalized_code.endswith('.US'):
                    normalized_code = normalized_code[2:] + '.US'
                
                stock_name = self.stock_code_name_map.get(normalized_code)
                codes.append({"stock_code": normalized_code, "stock_name": stock_name})
            
            return codes
        
        # 1. 优先识别股票代码
        stock_codes = extract_stock_codes(user_input)
        if stock_codes:
            entities["stocks"] = stock_codes
            # 保留第一个作为主要股票
            entities["stock_name"] = stock_codes[0]["stock_name"]
            entities["stock_code"] = stock_codes[0]["stock_code"]
            logger.debug(f"识别到股票代码: {[code['stock_code'] for code in stock_codes]}")
            return entities
        
        # 2. 处理包含连接词的复杂场景
        # 分割输入文本，分别处理每个部分
        conjunction_pattern = r'[和与以及跟同]'
        if re.search(conjunction_pattern, user_input):
            parts = re.split(conjunction_pattern, user_input)
            for part in parts:
                part = part.strip()
                if len(part) >= 2:
                    # 对每个部分进行股票识别
                    stock_entity = recognize_single_stock(part)
                    if stock_entity["stock_name"]:
                        # 添加到多实体列表
                        entities["stocks"].append({
                            "stock_name": stock_entity["stock_name"],
                            "stock_code": stock_entity["stock_code"]
                        })
        else:
            # 3. 处理单一股票的情况
            stock_entity = recognize_single_stock(user_input)
            if stock_entity["stock_name"]:
                # 添加到多实体列表
                entities["stocks"].append({
                    "stock_name": stock_entity["stock_name"],
                    "stock_code": stock_entity["stock_code"]
                })
                # 设置主要股票
                entities["stock_name"] = stock_entity["stock_name"]
                entities["stock_code"] = stock_entity["stock_code"]
        
        # 5. 去重多实体列表
        seen = set()
        unique_stocks = []
        for stock in entities["stocks"]:
            if stock["stock_name"] and stock["stock_name"] not in seen:
                seen.add(stock["stock_name"])
                unique_stocks.append(stock)
        entities["stocks"] = unique_stocks
        
        # 6. 确保多实体列表不为空时，更新主要股票字段
        if not entities["stock_name"] and entities["stocks"]:
            entities["stock_name"] = entities["stocks"][0]["stock_name"]
            entities["stock_code"] = entities["stocks"][0]["stock_code"]
        
        return entities
    
    def _context_based_entity_completion(self, current_entities: Dict, dialogue_history: List[Dict]) -> Dict:
        """
        基于上下文的实体补全
        
        Args:
            current_entities: 当前识别到的实体
            dialogue_history: 对话历史
            
        Returns:
            Dict: 补全后的实体信息
        """
        completed_entities = {}
        
        # 当前实体状态
        has_stock = current_entities.get("stock_code") or current_entities.get("stock_name")
        has_sector = current_entities.get("sector")
        has_index = current_entities.get("index_name")
        
        # 实体优先级：股票 > 板块 > 指数
        entity_priority = ["stock", "sector", "index"]
        
        # 查找目标实体类型（按优先级）
        target_entity_type = None
        if not has_stock:
            target_entity_type = "stock"
        elif not has_sector:
            target_entity_type = "sector"
        elif not has_index:
            target_entity_type = "index"
        
        if target_entity_type:
            # 对话历史窗口（最近10轮）
            context_window = dialogue_history[-10:]
            
            # 从最近的对话中查找实体
            for msg in reversed(context_window):
                if 'entities' in msg:
                    entities = msg['entities']
                    
                    if target_entity_type == "stock":
                        if entities.get("stock_code"):
                            completed_entities["stock_code"] = entities["stock_code"]
                            completed_entities["stock_name"] = entities.get("stock_name")
                            logger.debug(f"从上下文补全股票实体: {completed_entities}")
                            break
                    elif target_entity_type == "sector":
                        if entities.get("sector"):
                            completed_entities["sector"] = entities["sector"]
                            logger.debug(f"从上下文补全板块实体: {completed_entities}")
                            break
                    elif target_entity_type == "index":
                        if entities.get("index_name"):
                            completed_entities["index_name"] = entities["index_name"]
                            logger.debug(f"从上下文补全指数实体: {completed_entities}")
                            break
        
        # 补全时间实体（如果当前没有明确的时间实体）
        if not current_entities.get("time_period") and not current_entities.get("time_entity"):
            for msg in reversed(dialogue_history[-5:]):
                if 'entities' in msg:
                    entities = msg['entities']
                    if entities.get("time_period"):
                        completed_entities["time_period"] = entities["time_period"]
                        logger.debug(f"从上下文补全时间实体: {completed_entities}")
                        break
        
        return completed_entities
    
    def _dialogue_history_modeling(self, dialogue_history: List[Dict]) -> Dict:
        """
        对话历史建模
        
        Args:
            dialogue_history: 对话历史
            
        Returns:
            Dict: 建模后的对话历史信息
        """
        if not dialogue_history:
            return {}
        
        # 对话历史统计信息
        history_info = {
            "total_turns": len(dialogue_history),
            "recent_intents": [],
            "recent_entities": {},
            "interaction_count": 0,
            "last_interaction_time": None
        }
        
        # 收集最近的意图和实体
        recent_window = dialogue_history[-5:]
        for msg in recent_window:
            if 'intent' in msg:
                history_info["recent_intents"].append(msg['intent'].value)
            
            if 'entities' in msg:
                entities = msg['entities']
                for key, value in entities.items():
                    if value:
                        history_info["recent_entities"][key] = value
        
        # 去重并保留最近的意图
        history_info["recent_intents"] = list(dict.fromkeys(history_info["recent_intents"]))
        
        # 计算交互次数
        for msg in dialogue_history:
            if msg.get('role') == 'user':
                history_info["interaction_count"] += 1
        
        # 记录最后交互时间
        if dialogue_history:
            last_msg = dialogue_history[-1]
            if 'timestamp' in last_msg:
                history_info["last_interaction_time"] = last_msg['timestamp']
        
        return history_info
    
    def _enhanced_sector_recognition(self, user_input: str, current_entities: Dict) -> Dict:
        """
        增强的板块识别，使用必盈API进行查询
        
        Args:
            user_input: 用户输入文本
            current_entities: 当前识别到的实体
            
        Returns:
            Dict: 识别到的板块实体
        """
        # 如果已经有股票实体，不覆盖板块识别
        if current_entities.get("stock_code") or current_entities.get("stock_name"):
            return {}
        
        # 增强的板块识别逻辑
        # 1. 板块关键词扩展
        sector_keywords = {
            "新能源": ["新能源", "光伏", "风电", "水电", "核电", "储能", "锂电池", "电动车", "新能源汽车"],
            "消费": ["消费", "零售", "食品", "饮料", "白酒", "家电", "医药消费", "医疗消费"],
            "科技": ["科技", "电子", "半导体", "芯片", "人工智能", "AI", "大数据", "云计算", "5G"],
            "金融": ["金融", "银行", "证券", "保险", "基金", "券商", "信托"],
            "医药": ["医药", "医疗", "生物", "制药", "创新药", "医疗器械", "医疗服务"]
        }
        
        # 2. 板块识别
        for sector, keywords in sector_keywords.items():
            if any(keyword in user_input for keyword in keywords):
                return {"sector": sector}
        
        # 3. 如果没有找到板块，尝试使用必盈API直接查询
        if not any(entity in current_entities for entity in ["sector", "stock_code", "stock_name"]):
            # 检查用户输入是否包含板块相关关键词
            sector_related_keywords = ["板块", "行业", "领域", "赛道"]
            if any(keyword in user_input for keyword in sector_related_keywords):
                # 尝试从用户输入中提取可能的板块名称
                import re
                potential_sectors = re.findall(r'[一-龥]+', user_input)
                for sector_name in potential_sectors:
                    if len(sector_name) >= 2:  # 板块名称至少2个字符
                        logger.debug(f"尝试使用必盈API查询潜在板块名称: {sector_name}")
                        # 目前必盈API可能不直接支持板块查询，这里预留接口
                        # sector_info = data_agent.get_sector_info_by_name(sector_name)
                        # if sector_info:
                        #     return {"sector": sector_info}
                
        return {}


class ContextManager:
    """
    上下文管理器，用于管理对话历史和状态
    """
    
    def __init__(self):
        # 会话历史存储
        self.session_histories = {}
        
        # 最大历史记录数
        self.max_history_length = 20
    
    def get_history(self, session_id: str) -> List[Dict]:
        """
        获取指定会话的历史记录
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Dict]: 对话历史
        """
        if not session_id:
            return []
        
        return self.session_histories.get(session_id, [])
    
    def update_context(self, session_id: str, user_input: str, intent: IntentType, entities: Dict):
        """
        更新对话上下文
        
        Args:
            session_id: 会话ID
            user_input: 用户输入
            intent: 识别到的意图
            entities: 识别到的实体
        """
        if not session_id:
            return
        
        # 确保会话历史存在
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        
        # 添加新的对话记录
        self.session_histories[session_id].append({
            "type": "user_input",
            "content": user_input,
            "intent": intent.value,
            "entities": entities,
            "timestamp": time.time()
        })
        
        # 限制历史记录长度
        if len(self.session_histories[session_id]) > self.max_history_length:
            self.session_histories[session_id] = self.session_histories[session_id][-self.max_history_length:]
    
    def clear_history(self, session_id: str):
        """
        清除指定会话的历史记录
        
        Args:
            session_id: 会话ID
        """
        if session_id in self.session_histories:
            del self.session_histories[session_id]
    
    def get_session_state(self, session_id: str) -> Dict:
        """
        获取会话状态
        
        Args:
            session_id: 会话ID
            
        Returns:
            Dict: 会话状态
        """
        history = self.get_history(session_id)
        if not history:
            return {}
        
        # 提取会话状态
        session_state = {
            "most_recent_intent": history[-1].get("intent"),
            "recent_intents": [],
            "intent_sequence": [],
            "recent_entities": {},
            "entity_counts": {},
            "interaction_count": len(history),
            "dialogue_length": len(history),
            "current_topic": None,
            "last_interaction_time": history[-1].get("timestamp"),
            "average_interaction_length": 0
        }
        
        # 收集最近的意图序列
        for msg in history[-10:]:  # 最近10轮对话
            if 'intent' in msg:
                session_state["recent_intents"].append(msg['intent'])
                session_state["intent_sequence"].append(msg['intent'])
        
        # 去重并保留顺序
        session_state["recent_intents"] = list(dict.fromkeys(session_state["recent_intents"]))
        
        # 收集实体信息
        for msg in reversed(history[-5:]):  # 最近5轮对话的实体
            if 'entities' in msg:
                entities = msg['entities']
                for key, value in entities.items():
                    if value:
                        # 最近的实体（优先保留最近的）
                        if key not in session_state["recent_entities"]:
                            session_state["recent_entities"][key] = value
                        
                        # 实体出现频率（只统计可哈希的值）
                        if key not in session_state["entity_counts"]:
                            session_state["entity_counts"][key] = {}
                        # 只有可哈希的值（如字符串、数字）才能作为字典的键
                        if isinstance(value, (str, int, float, bool)):
                            if value not in session_state["entity_counts"][key]:
                                session_state["entity_counts"][key][value] = 0
                            session_state["entity_counts"][key][value] += 1
        
        # 确定当前对话主题
        if session_state["recent_entities"]:
            # 优先根据股票、板块、指数实体确定主题
            if session_state["recent_entities"].get("stock_name"):
                session_state["current_topic"] = session_state["recent_entities"]["stock_name"]
            elif session_state["recent_entities"].get("sector"):
                session_state["current_topic"] = session_state["recent_entities"]["sector"]
            elif session_state["recent_entities"].get("index_name"):
                session_state["current_topic"] = session_state["recent_entities"]["index_name"]
        
        # 计算平均交互长度
        total_length = 0
        for msg in history:
            if 'content' in msg:
                total_length += len(msg['content'])
        if session_state["interaction_count"] > 0:
            session_state["average_interaction_length"] = round(total_length / session_state["interaction_count"])
        
        return session_state
    
    def get_intent_transition(self, session_id: str) -> List[Tuple[str, str]]:
        """
        获取意图转换序列
        
        Args:
            session_id: 会话ID
            
        Returns:
            List[Tuple[str, str]]: 意图转换序列，每个元素是(前一个意图, 当前意图)
        """
        history = self.get_history(session_id)
        if len(history) < 2:
            return []
        
        intent_transitions = []
        for i in range(1, len(history)):
            prev_intent = history[i-1].get("intent")
            curr_intent = history[i].get("intent")
            if prev_intent and curr_intent:
                intent_transitions.append((prev_intent, curr_intent))
        
        return intent_transitions
    
    def get_dialogue_stage(self, session_id: str) -> str:
        """
        获取对话阶段
        
        Args:
            session_id: 会话ID
            
        Returns:
            str: 对话阶段（"初始阶段", "探索阶段", "深入阶段", "总结阶段"）
        """
        session_state = self.get_session_state(session_id)
        interaction_count = session_state.get("interaction_count", 0)
        recent_intents = session_state.get("recent_intents", [])
        
        if interaction_count <= 1:
            return "初始阶段"
        elif interaction_count <= 3 and len(recent_intents) > 1:
            return "探索阶段"
        elif interaction_count > 3 and len(recent_intents) > 0:
            return "深入阶段"
        else:
            return "总结阶段"


# 全局NLP代理实例
nlp_agent = NLPAgent()
