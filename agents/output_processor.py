#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型输出微调和后处理模块
负责对模型生成的输出进行微调和后处理，确保输出质量
"""

import logging
import re
from typing import Dict, Optional
from agents.intent_recognizer import IntentType

logger = logging.getLogger(__name__)


class OutputProcessor:
    """模型输出处理器"""
    
    def __init__(self):
        # 正则表达式模式
        self.error_patterns = [
            re.compile(r"模型未加载成功", re.IGNORECASE),
            re.compile(r"无法生成响应", re.IGNORECASE),
            re.compile(r"error", re.IGNORECASE),
            re.compile(r"timeout", re.IGNORECASE),
            re.compile(r"请求失败", re.IGNORECASE),
        ]
        
        # 输出风格配置（扩展到所有意图类型）
        self.style_configs = {
            # 分析类意图
            IntentType.ANALYSIS_STOCK: {
                "tone": "professional",
                "max_length": 1500,
                "allow_technical_terms": True,
                "template": "股票分析"
            },
            IntentType.ANALYSIS_STOCK_FUNDAMENTAL: {
                "tone": "detail-oriented",
                "max_length": 1800,
                "allow_technical_terms": True,
                "template": "股票基本面分析"
            },
            IntentType.ANALYSIS_STOCK_TECHNICAL: {
                "tone": "technical",
                "max_length": 1500,
                "allow_technical_terms": True,
                "template": "股票技术面分析"
            },
            IntentType.ANALYSIS_SECTOR: {
                "tone": "analytical",
                "max_length": 1200,
                "allow_technical_terms": True,
                "template": "板块分析"
            },
            IntentType.ANALYSIS_MARKET: {
                "tone": "comprehensive",
                "max_length": 2000,
                "allow_technical_terms": True,
                "template": "大盘分析"
            },
            IntentType.ANALYSIS_STOCK_VALUATION: {
                "tone": "detail-oriented",
                "max_length": 1500,
                "allow_technical_terms": True,
                "template": "股票估值分析"
            },
            IntentType.ANALYSIS_STOCK_PERFORMANCE: {
                "tone": "data-driven",
                "max_length": 1200,
                "allow_technical_terms": True,
                "template": "股票表现分析"
            },
            IntentType.ANALYSIS_SECTOR_PERFORMANCE: {
                "tone": "data-driven",
                "max_length": 1200,
                "allow_technical_terms": True,
                "template": "板块表现分析"
            },
            IntentType.ANALYSIS_MARKET_PERFORMANCE: {
                "tone": "data-driven",
                "max_length": 1500,
                "allow_technical_terms": True,
                "template": "大盘表现分析"
            },
            
            # 查询类意图
            IntentType.QUERY_NEWS: {
                "tone": "informative",
                "max_length": 1000,
                "allow_technical_terms": False,
                "template": "相关新闻"
            },
            IntentType.QUERY_STOCK_LIST: {
                "tone": "concise",
                "max_length": 800,
                "allow_technical_terms": False,
                "template": "股票列表"
            },
            IntentType.QUERY_STOCK_INFO: {
                "tone": "data-driven",
                "max_length": 600,
                "allow_technical_terms": True,
                "template": "股票报价"
            },
            IntentType.QUERY_MARKET_INFO: {
                "tone": "data-driven",
                "max_length": 500,
                "allow_technical_terms": True,
                "template": "指数报价"
            },
            IntentType.QUERY_SECTOR_INFO: {
                "tone": "informative",
                "max_length": 800,
                "allow_technical_terms": True,
                "template": "板块信息"
            },
            
            # 推荐类意图
            IntentType.RECOMMEND_STOCK: {
                "tone": "advisory",
                "max_length": 1200,
                "allow_technical_terms": True,
                "template": "股票推荐"
            },
            IntentType.RECOMMEND_SECTOR: {
                "tone": "advisory",
                "max_length": 1000,
                "allow_technical_terms": True,
                "template": "板块推荐"
            },
            IntentType.RECOMMEND_PORTFOLIO: {
                "tone": "advisory",
                "max_length": 1500,
                "allow_technical_terms": True,
                "template": "投资组合推荐"
            },
            
            # 其他意图
            IntentType.UNKNOWN: {
                "tone": "friendly",
                "max_length": 500,
                "allow_technical_terms": False,
                "template": "信息回复"
            }
        }
    
    def process_output(self, model_output: str, intent: IntentType, entities: Dict) -> str:
        """
        处理模型输出
        
        Args:
            model_output: 模型生成的原始输出
            intent: 意图类型
            entities: 实体信息
            
        Returns:
            str: 处理后的输出
        """
        if not model_output:
            return self._get_default_response(intent)
        
        # 检测错误
        if self._is_error_response(model_output):
            logger.warning("检测到模型返回错误信息，将生成默认响应")
            return self._get_default_response(intent)
        
        # 获取风格配置
        config = self.style_configs.get(intent, self.style_configs[IntentType.UNKNOWN])
        
        # 进行各种处理
        processed = model_output
        
        # 清理格式
        processed = self._clean_formatting(processed)
        
        # 调整长度
        processed = self._adjust_length(processed, config["max_length"])
        
        # 调整语气和风格
        processed = self._adjust_tone(processed, config["tone"])
        
        # 处理技术术语
        if not config["allow_technical_terms"]:
            processed = self._simplify_technical_terms(processed)
        
        # 根据意图进行特定处理
        processed = self._intent_specific_processing(processed, intent, entities)
        
        # 确保响应自然流畅
        processed = self._make_natural(processed)
        
        logger.info(f"模型输出处理完成，处理前后长度: {len(model_output)} -> {len(processed)} 字符")
        
        return processed
    
    def _is_error_response(self, text: str) -> bool:
        """检测是否为错误响应"""
        for pattern in self.error_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _clean_formatting(self, text: str) -> str:
        """清理格式"""
        # 去除多余的空行
        text = re.sub(r"\n\s*\n", "\n\n", text)
        
        # 去除行首多余的空格
        text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)
        
        # 去除连续的空格
        text = re.sub(r"\s+", " ", text)
        
        # 恢复换行符
        text = text.replace("。 ", "。\n")
        text = text.replace("！ ", "！\n")
        text = text.replace("？ ", "？\n")
        
        return text.strip()
    
    def _adjust_length(self, text: str, max_length: int) -> str:
        """调整文本长度"""
        if len(text) <= max_length:
            return text
        
        # 截断到最近的句子结束
        truncated = text[:max_length]
        last_period = truncated.rfind("。")
        last_exclamation = truncated.rfind("！")
        last_question = truncated.rfind("？")
        
        last_punctuation = max(last_period, last_exclamation, last_question)
        
        if last_punctuation > max_length - 200:  # 确保至少保留最后200个字符
            truncated = truncated[:last_punctuation + 1]
        
        truncated += "...\n\n(内容过长已自动截断，如需完整分析请提出更具体的问题。)"
        
        logger.warning(f"模型输出被截断，原始长度: {len(text)}，截断后长度: {len(truncated)}")
        
        return truncated
    
    def _adjust_tone(self, text: str, tone: str) -> str:
        """调整语气和风格"""
        if tone == "professional":
            # 专业语气：保持正式、准确，突出数据和分析
            text = text.replace("我认为", "分析表明")
            text = text.replace("应该", "建议")
            text = text.replace("不错", "表现良好")
            text = text.replace("很好", "表现优异")
            return text
        elif tone == "analytical":
            # 分析性语气：强调逻辑关系，突出因果联系
            text = text.replace("因为", "由于")
            text = text.replace("所以", "因此")
            text = text.replace("但是", "然而")
            text = text.replace("同时", "此外")
            return text
        elif tone == "comprehensive":
            # 全面性语气：确保内容覆盖多个方面
            if "总结" not in text and "总体" not in text:
                text += "\n\n总体来看，市场表现呈现多元化特点，投资者应综合考虑多种因素。"
            return text
        elif tone == "informative":
            # 信息传递语气：简洁明了，突出事实
            text = re.sub(r"\n\s*\n", "\n", text)  # 减少空行
            text = re.sub(r"\n\s*[1-9]\s*\.", "\n•", text)  # 使用项目符号
            return text
        elif tone == "concise":
            # 简洁语气：删除冗余信息，只保留核心内容
            sentences = re.split(r"[。！？]", text)
            key_sentences = []
            keywords = ["主要", "核心", "关键", "重要", "数据", "表现", "建议"]
            
            for sentence in sentences:
                if any(keyword in sentence for keyword in keywords) and sentence.strip():
                    key_sentences.append(sentence.strip())
            
            if key_sentences:
                return "。".join(key_sentences) + "。"
            return text
        elif tone == "friendly":
            # 友好语气：使用亲切的表达，避免生硬
            text = text.replace("您需要了解", "您想知道")
            text = text.replace("请知悉", "希望对您有帮助")
            text = text.replace("请注意", "建议关注")
            return text
        elif tone == "detail-oriented":
            # 注重细节语气：突出具体数据和详细信息
            text = text.replace("一些", "多项")
            text = text.replace("很多", "大量")
            text = text.replace("大概", "约")
            return text
        elif tone == "technical":
            # 技术性语气：保留专业术语，突出技术分析
            text = text.replace("趋势", "技术趋势")
            text = text.replace("指标", "技术指标")
            text = text.replace("形态", "技术形态")
            return text
        elif tone == "advisory":
            # 建议性语气：强调建议和指导
            if "建议" not in text:
                text += "\n\n综合考虑，建议投资者根据自身风险承受能力制定投资策略。"
            return text
        elif tone == "data-driven":
            # 数据驱动语气：突出数据和事实
            text = text.replace("看起来", "根据数据")
            text = text.replace("似乎", "数据显示")
            return text
        else:
            return text
    
    def _simplify_technical_terms(self, text: str) -> str:
        """简化技术术语"""
        # 术语替换字典
        term_replacements = {
            "市盈率": "股价与每股盈利的比率",
            "市净率": "股价与每股净资产的比率",
            "ROE": "净资产收益率",
            "MACD": "平滑异同移动平均线",
            "KDJ": "随机指标",
            "RSI": "相对强弱指标",
            "成交量": "交易量",
            "成交额": "交易金额",
            "涨跌幅": "涨跌幅度",
            "换手率": "股票转手买卖的频率"
        }
        
        for term, replacement in term_replacements.items():
            text = text.replace(term, f"{term}（{replacement}）")
        
        return text
    
    def _intent_specific_processing(self, text: str, intent: IntentType, entities: Dict) -> str:
        """根据意图进行特定处理"""
        try:
            logger.info(f"开始意图特定处理，意图: {intent.value if hasattr(intent, 'value') else str(intent)}")
            
            # 验证意图类型
            if not hasattr(intent, 'value'):
                logger.error(f"无效的意图类型: {intent}")
                return text
            
            # 获取意图对应的模板
            config = self.style_configs.get(intent, self.style_configs[IntentType.UNKNOWN])
            template = config.get("template", "信息回复")
            logger.debug(f"意图模板: {template}")
            
            # 通用处理：确保包含模板标题
            if template not in text:
                logger.debug(f"添加模板标题: {template}")
                text = f"{template}：\n{text}"
            
            # 特定意图的额外处理
            if IntentType.is_analysis_intent(intent):
                logger.info(f"处理分析类意图: {intent.value}")
                # 所有分析类意图的通用处理
                if intent == IntentType.ANALYSIS_STOCK:
                    # 股票分析特定处理
                    logger.debug("处理股票分析特定意图")
                    stock_code = entities.get("stock_code")
                    if stock_code and f"{stock_code}" not in text:
                        logger.debug(f"添加股票代码到标题: {stock_code}")
                        text = text.replace(f"{template}：", f"{stock_code}{template}：")
                
                elif intent == IntentType.ANALYSIS_STOCK_FUNDAMENTAL:
                    # 股票基本面分析特定处理
                    logger.debug("处理股票基本面分析特定意图")
                    stock_name = entities.get("stock_name")
                    if stock_name and f"{stock_name}" not in text:
                        logger.debug(f"添加股票名称到标题: {stock_name}")
                        text = text.replace(f"{template}：", f"{stock_name}{template}：")
                
                elif intent == IntentType.ANALYSIS_STOCK_TECHNICAL:
                    # 股票技术面分析特定处理
                    logger.debug("处理股票技术面分析特定意图")
                    stock_name = entities.get("stock_name")
                    if stock_name and f"{stock_name}" not in text:
                        logger.debug(f"添加股票名称到标题: {stock_name}")
                        text = text.replace(f"{template}：", f"{stock_name}{template}：")
                
                elif intent == IntentType.ANALYSIS_SECTOR:
                    # 板块分析特定处理
                    logger.debug("处理板块分析特定意图")
                    sector = entities.get("sector")
                    if sector and f"{sector}" not in text:
                        logger.debug(f"添加板块名称到标题: {sector}")
                        text = text.replace(f"{template}：", f"{sector}{template}：")
                
                elif intent == IntentType.ANALYSIS_MARKET:
                    # 大盘分析特定处理
                    logger.debug("处理大盘分析特定意图")
                    # 大盘分析无需额外处理
                    pass
            
            elif IntentType.is_query_intent(intent):
                logger.info(f"处理查询类意图: {intent.value}")
                # 查询类意图的通用处理
                if intent == IntentType.QUERY_NEWS:
                    # 新闻查询特定处理
                    logger.debug("处理新闻查询特定意图")
                    if "摘要" not in text and "解读" not in text:
                        logger.debug("添加新闻摘要与解读前缀")
                        text = text.replace("相关新闻：", "相关新闻摘要与解读：")
            
            elif intent == IntentType.QUERY_STOCK_INFO:
                # 股票报价特定处理
                logger.debug("处理股票报价特定意图")
                stock_name = entities.get("stock_name")
                if stock_name and f"{stock_name}" not in text:
                    logger.debug(f"添加股票名称到标题: {stock_name}")
                    text = text.replace(f"{template}：", f"{stock_name}{template}：")
            
            elif intent == IntentType.QUERY_MARKET_INFO:
                # 指数报价特定处理
                logger.debug("处理指数报价特定意图")
                index_name = entities.get("index_name")
                if index_name and f"{index_name}" not in text:
                    logger.debug(f"添加指数名称到标题: {index_name}")
                    text = text.replace(f"{template}：", f"{index_name}{template}：")
            
            elif IntentType.is_recommendation_intent(intent):
                logger.info(f"处理推荐类意图: {intent.value}")
                # 推荐类意图的通用处理
                if "风险提示" not in text:
                    logger.debug("添加风险提示")
                    text += "\n\n风险提示：投资有风险，入市需谨慎。以上建议仅供参考，不构成投资决策。"
            
            if intent == IntentType.RECOMMEND_STOCK:
                # 股票推荐特定处理
                logger.debug("处理股票推荐特定意图")
                if "推荐理由" not in text:
                    logger.debug("添加推荐理由前缀")
                    text = text.replace("股票推荐：", "股票推荐及理由：")
            
            elif intent == IntentType.RECOMMEND_SECTOR:
                # 板块推荐特定处理
                logger.debug("处理板块推荐特定意图")
                if "推荐理由" not in text:
                    logger.debug("添加推荐理由前缀")
                    text = text.replace("板块推荐：", "板块推荐及理由：")
            
            logger.info(f"意图特定处理完成")
            return text
        except Exception as e:
            logger.error(f"意图特定处理失败: {e}", exc_info=True)
            return text
    
    def _make_natural(self, text: str) -> str:
        """确保文本自然流畅"""
        # 修复不自然的标点和空格
        text = re.sub(r"([。！？,，；;])\s*([。！？,，；;])", r"\1\2", text)
        text = re.sub(r"\s+([。！？,，；:：])", r"\1", text)
        
        # 确保开头自然
        if not text.startswith(("您好", "感谢", "根据", "关于", "以下", "我", "​")):
            text = f"您好！{text}"
        
        # 确保结尾友好
        if not text.endswith(("。", "！", "？", "建议", "分析", "参考")):
            text = f"{text}，希望对您有帮助！"
        
        return text
    
    def _get_default_response(self, intent: IntentType) -> str:
        """获取默认响应"""
        default_responses = {
            # 分析类意图
            IntentType.ANALYSIS_STOCK: "抱歉，暂时无法为您生成股票分析报告，请稍后重试。您也可以提供更具体的股票代码和问题。",
            IntentType.ANALYSIS_STOCK_FUNDAMENTAL: "抱歉，暂时无法为您生成股票基本面分析，请稍后重试。",
            IntentType.ANALYSIS_STOCK_TECHNICAL: "抱歉，暂时无法为您生成股票技术面分析，请稍后重试。",
            IntentType.ANALYSIS_SECTOR: "抱歉，暂时无法为您生成板块分析，请稍后重试。您也可以提供更具体的板块名称。",
            IntentType.ANALYSIS_MARKET: "抱歉，暂时无法为您生成大盘分析，请稍后重试。",
            IntentType.ANALYSIS_STOCK_VALUATION: "抱歉，暂时无法为您生成股票估值分析，请稍后重试。",
            IntentType.ANALYSIS_STOCK_PERFORMANCE: "抱歉，暂时无法为您生成股票表现分析，请稍后重试。",
            IntentType.ANALYSIS_SECTOR_PERFORMANCE: "抱歉，暂时无法为您生成板块表现分析，请稍后重试。",
            IntentType.ANALYSIS_MARKET_PERFORMANCE: "抱歉，暂时无法为您生成大盘表现分析，请稍后重试。",
            
            # 查询类意图
            IntentType.QUERY_NEWS: "抱歉，暂时无法为您查询相关新闻，请稍后重试。",
            IntentType.QUERY_STOCK_LIST: "抱歉，暂时无法为您提供股票列表，请稍后重试。",
            IntentType.QUERY_STOCK_INFO: "抱歉，暂时无法为您查询股票报价，请稍后重试。",
            IntentType.QUERY_MARKET_INFO: "抱歉，暂时无法为您查询指数报价，请稍后重试。",
            IntentType.QUERY_SECTOR_INFO: "抱歉，暂时无法为您查询板块信息，请稍后重试。",
            
            # 推荐类意图
            IntentType.RECOMMEND_STOCK: "抱歉，暂时无法为您提供股票推荐，请稍后重试。",
            IntentType.RECOMMEND_SECTOR: "抱歉，暂时无法为您提供板块推荐，请稍后重试。",
            IntentType.RECOMMEND_PORTFOLIO: "抱歉，暂时无法为您提供投资组合推荐，请稍后重试。",
            
            # 其他意图
            IntentType.UNKNOWN: "抱歉，我暂时无法理解您的问题，请尝试提供更具体的金融相关问题。"
        }
        
        # 如果找不到特定意图的响应，尝试使用父意图的响应
        current_intent = intent
        while current_intent not in default_responses and current_intent != IntentType.ROOT:
            parent = IntentType.get_parent(current_intent)
            if parent == current_intent:  # 已经到达根节点
                break
            current_intent = parent
        
        return default_responses.get(current_intent, default_responses[IntentType.UNKNOWN])
    
    def detect_error(self, text: str) -> Optional[str]:
        """检测错误并返回错误信息"""
        for pattern in self.error_patterns:
            match = pattern.search(text)
            if match:
                return match.group(0)
        return None
    
    def format_as_markdown(self, text: str, intent: IntentType) -> str:
        """将文本格式化为Markdown"""
        # 根据意图添加标题
        if intent == IntentType.ANALYSIS_STOCK:
            return f"# 股票分析\n\n{text}"
        elif intent == IntentType.ANALYSIS_SECTOR:
            return f"# 板块分析\n\n{text}"
        elif intent == IntentType.ANALYSIS_MARKET:
            return f"# 大盘分析\n\n{text}"
        elif intent == IntentType.QUERY_NEWS:
            return f"# 相关新闻\n\n{text}"
        elif intent == IntentType.QUERY_STOCK_LIST:
            return f"# 股票列表\n\n{text}"
        else:
            return text
    
    def extract_key_points(self, text: str, max_points: int = 3) -> list:
        """从文本中提取关键点"""
        # 简单的关键点提取逻辑
        sentences = re.split(r"[。！？]", text)
        key_points = []
        
        # 优先选择包含关键词的句子
        keywords = ["分析", "建议", "趋势", "特点", "数据", "表现", "预测"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # 检查是否包含关键词
            has_keyword = any(keyword in sentence for keyword in keywords)
            
            if has_keyword or len(key_points) < max_points:
                key_points.append(sentence)
                
                if len(key_points) >= max_points:
                    break
        
        return key_points


# 创建全局输出处理器实例
output_processor = OutputProcessor()
