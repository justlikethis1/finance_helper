from typing import Dict, Optional, List
import logging

from .intent_type import IntentType

logger = logging.getLogger(__name__)

class InformationRetriever:
    """信息检索器"""
    
    def __init__(self):
        """初始化信息检索器"""
        pass
    
    def generate_prompt(self, intent: IntentType, entities: Dict, retrieved_info: Dict, user_input: str) -> str:
        """
        根据检索到的信息生成提示词
        
        Args:
            intent: 意图类型
            entities: 实体信息
            retrieved_info: 检索到的信息
            user_input: 用户原始输入
            
        Returns:
            str: 生成的提示词
        """
        # 获取情感分析结果
        sentiment = entities.get('sentiment', {})
        financial_sentiment = sentiment.get('financial_label', '中性')
        is_question = sentiment.get('is_question', False)
        
        # 获取用户投资经验和风险偏好
        investment_experience = entities.get('investment_experience', '一般')
        risk_preference = entities.get('risk_preference', '稳健')
        
        # 获取用户投资金额
        investment_amount = entities.get('investment_amount', '未知')
        
        # 基础模板 - 更具个性化
        base_prompt = f"""
# 角色定位
你是一位专业的金融分析师，拥有丰富的金融知识和市场经验。你的回答应该专业、准确、客观，同时易于理解。

# 用户问题
{user_input}

# 任务要求
1. 基于以下提供的数据，提供专业、准确的分析和回答
2. 确保信息的准确性和时效性，对于不确定的数据明确说明
3. 保持回答的逻辑性和条理性，使用结构化的格式
4. 根据用户的语言风格调整回答语气
5. 不要编造信息，对于未知数据说明"未知"
6. {'直接回答用户的问题，保持简洁明了' if is_question else '提供全面详细的分析和建议'}
7. 根据用户的投资经验和风险偏好调整分析深度和建议风格
8. 考虑用户的投资金额，提供适合的投资建议
"""
        
        # 添加用户个性化信息
        base_prompt += f"\n# 用户信息\n"
        base_prompt += f"投资经验：{investment_experience}\n"
        base_prompt += f"风险偏好：{risk_preference}\n"
        base_prompt += f"投资金额：{investment_amount}\n"
        
        # 根据用户投资经验和风险偏好调整分析深度
        if investment_experience == '新手':
            base_prompt += "\n# 分析深度\n请使用通俗易懂的语言，避免专业术语，提供基础的投资知识和简单明了的分析。"
        elif investment_experience == '专业':
            base_prompt += "\n# 分析深度\n请提供详细的专业分析，包括技术指标、财务数据和市场趋势的深入解读。"
        else:
            base_prompt += "\n# 分析深度\n请提供适度的专业分析，平衡专业性和可读性。"
        
        # 根据风险偏好调整建议风格
        if risk_preference == '保守':
            base_prompt += "\n# 建议风格\n请优先考虑风险控制，提供稳健的投资建议，重点关注低风险、高流动性的投资机会。"
        elif risk_preference == '激进':
            base_prompt += "\n# 建议风格\n请提供进取型的投资建议，关注高成长、高收益的投资机会，同时提示潜在风险。"
        else:
            base_prompt += "\n# 建议风格\n请提供平衡的投资建议，在风险和收益之间取得合理平衡。"
        
        # 添加时间信息
        from datetime import datetime
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        base_prompt += f"\n# 当前时间\n{current_time}\n"
        
        # 添加互联网检索到的信息
        internet_info_prompt = self._build_internet_info_prompt(retrieved_info)
        if internet_info_prompt:
            base_prompt += internet_info_prompt
        
        # 基于意图类型的个性化提示词
        if intent == IntentType.ANALYSIS_STOCK_FUNDAMENTAL:
            prompt = self._build_stock_fundamental_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_STOCK_TECHNICAL:
            prompt = self._build_stock_technical_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_STOCK_VALUATION:
            prompt = self._build_stock_valuation_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_STOCK_PERFORMANCE:
            prompt = self._build_stock_performance_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_SECTOR_FUNDAMENTAL:
            prompt = self._build_sector_fundamental_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_SECTOR_TECHNICAL:
            prompt = self._build_sector_technical_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_SECTOR_PERFORMANCE:
            prompt = self._build_sector_performance_analysis_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_MARKET_FUNDAMENTAL:
            prompt = self._build_market_fundamental_analysis_prompt(retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_MARKET_TECHNICAL:
            prompt = self._build_market_technical_analysis_prompt(retrieved_info, base_prompt)
        elif intent == IntentType.ANALYSIS_MARKET_PERFORMANCE:
            prompt = self._build_market_performance_analysis_prompt(retrieved_info, base_prompt)
        elif intent == IntentType.QUERY_NEWS:
            prompt = self._build_news_query_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.QUERY_STOCK_INFO:
            prompt = self._build_stock_info_query_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.QUERY_SECTOR_INFO:
            prompt = self._build_sector_info_query_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.QUERY_MARKET_INFO:
            prompt = self._build_market_info_query_prompt(retrieved_info, base_prompt)
        elif intent == IntentType.QUERY_STOCK_LIST:
            prompt = self._build_stock_list_prompt(retrieved_info, base_prompt)
        elif intent == IntentType.RECOMMEND_STOCK:
            prompt = self._build_stock_recommendation_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.RECOMMEND_SECTOR:
            prompt = self._build_sector_recommendation_prompt(entities, retrieved_info, base_prompt)
        elif intent == IntentType.RECOMMEND_PORTFOLIO:
            prompt = self._build_portfolio_recommendation_prompt(retrieved_info, base_prompt)
        elif intent == IntentType.GENERATE_REPORT:
            prompt = self._build_report_generation_prompt(entities, retrieved_info, base_prompt)
        elif IntentType.is_analysis_intent(intent):
            # 通用分析意图模板
            prompt = base_prompt + f"\n# 分析范围\n请对{entities.get('stock_name', entities.get('sector', '相关金融数据'))}进行全面分析"
            prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n1. 概况分析\n2. 详细分析\n3. 结论和建议"
        elif IntentType.is_query_intent(intent):
            # 通用查询意图模板
            prompt = base_prompt + f"\n# 查询要求\n请提供关于{entities.get('stock_name', entities.get('sector', '相关金融信息'))}的详细查询结果"
            prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n1. 查询结果摘要\n2. 详细信息\n3. 相关建议"
        elif IntentType.is_recommendation_intent(intent):
            # 通用推荐意图模板
            prompt = base_prompt + f"\n# 推荐要求\n请根据您的专业知识，提供关于{entities.get('stock_name', entities.get('sector', '相关金融产品'))}的合理推荐"
            prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n1. 推荐理由\n2. 具体推荐\n3. 风险提示"
        else:
            prompt = base_prompt + "\n# 提示\n请提供更具体的金融相关问题，以便我能为您提供更有针对性的分析。"
        
        # 根据情感分析结果调整提示词
        if financial_sentiment == "看涨":
            prompt += "\n\n# 情感提示\n用户对市场持看涨态度，请在回答中适当考虑这一点，但保持专业客观的分析立场。"
        elif financial_sentiment == "看跌":
            prompt += "\n\n# 情感提示\n用户对市场持看跌态度，请在回答中适当考虑这一点，但保持专业客观的分析立场。"
        
        # 限制提示词长度，避免超过模型限制
        max_prompt_length = 4000
        if len(prompt) > max_prompt_length:
            # 保留关键部分，缩短数据部分
            parts = prompt.split('# 输出结构要求')
            if len(parts) == 2:
                prompt = parts[0][:max_prompt_length - len(parts[1]) - 20] + '# 输出结构要求' + parts[1]
            else:
                prompt = prompt[:max_prompt_length - 100] + '\n\n# 提示\n由于内容过长，部分数据已省略。'
        
        logger.debug(f"生成的提示词长度: {len(prompt)} 字符")
        
        return prompt
    
    def _build_internet_info_prompt(self, retrieved_info: Dict) -> str:
        """构建互联网信息提示词"""
        prompt = ""
        
        internet_data = retrieved_info.get("data", {})
        internet_info_sections = []
        
        # 添加股票相关互联网新闻
        if internet_data.get("internet_stock_news"):
            news_list = internet_data["internet_stock_news"]
            if news_list:
                internet_info_sections.append("## 互联网股票新闻")
                for i, news in enumerate(news_list[:3], 1):
                    internet_info_sections.append(f"{i}. {news.get('title', '')}")
                    internet_info_sections.append(f"   来源: {news.get('source', '')}")
                    internet_info_sections.append(f"   摘要: {news.get('summary', '')}")
        
        # 添加板块相关互联网新闻
        if internet_data.get("internet_sector_news"):
            news_list = internet_data["internet_sector_news"]
            if news_list:
                internet_info_sections.append("## 互联网板块新闻")
                for i, news in enumerate(news_list[:3], 1):
                    internet_info_sections.append(f"{i}. {news.get('title', '')}")
                    internet_info_sections.append(f"   来源: {news.get('source', '')}")
                    internet_info_sections.append(f"   摘要: {news.get('summary', '')}")
        
        # 添加市场相关互联网新闻
        if internet_data.get("internet_market_news"):
            news_list = internet_data["internet_market_news"]
            if news_list:
                internet_info_sections.append("## 互联网市场新闻")
                for i, news in enumerate(news_list[:3], 1):
                    internet_info_sections.append(f"{i}. {news.get('title', '')}")
                    internet_info_sections.append(f"   来源: {news.get('source', '')}")
                    internet_info_sections.append(f"   摘要: {news.get('summary', '')}")
        
        # 添加行业相关互联网新闻
        if internet_data.get("internet_industry_news"):
            news_list = internet_data["internet_industry_news"]
            if news_list:
                internet_info_sections.append("## 互联网行业新闻")
                for i, news in enumerate(news_list[:3], 1):
                    internet_info_sections.append(f"{i}. {news.get('title', '')}")
                    internet_info_sections.append(f"   来源: {news.get('source', '')}")
                    internet_info_sections.append(f"   摘要: {news.get('summary', '')}")
        
        # 添加财经相关互联网新闻
        if internet_data.get("internet_finance_news"):
            news_list = internet_data["internet_finance_news"]
            if news_list:
                internet_info_sections.append("## 互联网财经新闻")
                for i, news in enumerate(news_list[:3], 1):
                    internet_info_sections.append(f"{i}. {news.get('title', '')}")
                    internet_info_sections.append(f"   来源: {news.get('source', '')}")
                    internet_info_sections.append(f"   摘要: {news.get('summary', '')}")
        
        if internet_info_sections:
            prompt = "\n\n# 互联网相关信息\n" + "\n".join(internet_info_sections)
        
        return prompt
    
    def _build_stock_fundamental_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票基本面分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票基本面分析\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        
        prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        
        # 股票报价 - 更详细展示
        stock_quote = retrieved_info.get("data", {}).get("stock_quote", {})
        if stock_quote and stock_code in stock_quote:
            df = stock_quote[stock_code]
            if not df.empty and len(df) > 0:
                try:
                    latest = df.iloc[0]
                    prompt += "\n## 最新行情\n"
                    prompt += f"当前报价：{latest.get('最新价', '未知')} 元\n"
                    prompt += f"涨跌幅：{latest.get('涨跌幅', '未知')}%\n"
                    prompt += f"开盘价：{latest.get('今开', '未知')} 元\n"
                    prompt += f"最高价：{latest.get('最高', '未知')} 元\n"
                    prompt += f"最低价：{latest.get('最低', '未知')} 元\n"
                    prompt += f"成交量：{latest.get('成交量', '未知')} 手\n"
                    prompt += f"成交额：{latest.get('成交额', '未知')} 元\n"
                    # 添加更多数据字段
                    prompt += f"昨收价：{latest.get('昨收', '未知')} 元\n"
                except IndexError:
                    logger.warning(f"获取股票 {stock_code} 报价数据时发生索引错误")
        
        # 股票新闻 - 更结构化展示
        stock_news = retrieved_info.get("data", {}).get("stock_news")
        if stock_news is not None and not stock_news.empty:
            prompt += "\n## 相关新闻\n"
            for i, (_, row) in enumerate(stock_news.head(3).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                summary = row.get('摘要', '') or row.get('summary', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 互联网股票新闻
        internet_stock_news = retrieved_info.get("data", {}).get("internet_stock_news")
        if internet_stock_news:
            prompt += "\n## 互联网相关新闻\n"
            for i, news in enumerate(internet_stock_news[:2], 1):
                title = news.get('title', '')
                source = news.get('source', '')
                summary = news.get('summary', '')
                prompt += f"{i}. {title}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 行业数据对比
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业对比\n"
            # 显示行业整体表现
            top_sectors = industry_data.head(5)
            for i, (_, row) in enumerate(top_sectors.iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 公司基本情况\n"
        prompt += "   - 公司简介与核心业务\n"
        prompt += "   - 所属行业与板块定位\n"
        prompt += "   - 市场地位与竞争优势\n"
        prompt += "2. 财务状况分析\n"
        prompt += "   - 盈利能力（营收、利润、毛利率等）\n"
        prompt += "   - 运营能力（周转率、存货等）\n"
        prompt += "   - 偿债能力（负债率、现金流等）\n"
        prompt += "3. 行业分析\n"
        prompt += "   - 行业发展趋势\n"
        prompt += "   - 竞争格局\n"
        prompt += "   - 政策环境\n"
        prompt += "4. 基本面综合评估\n"
        prompt += "   - 优势分析\n"
        prompt += "   - 风险因素\n"
        prompt += "   - 成长性评估\n"
        prompt += "5. 投资建议\n"
        prompt += "   - 估值分析\n"
        prompt += "   - 目标价位\n"
        prompt += "   - 投资时机与仓位建议\n"
        
        return prompt
    
    def _build_stock_technical_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票技术面分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票技术面分析\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        
        prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        
        # 股票报价 - 更详细展示
        stock_quote = retrieved_info.get("data", {}).get("stock_quote", {})
        if stock_quote and stock_code in stock_quote:
            df = stock_quote[stock_code]
            if not df.empty and len(df) > 0:
                try:
                    latest = df.iloc[0]
                    prompt += "\n## 最新行情\n"
                    prompt += f"当前报价：{latest.get('最新价', '未知')} 元\n"
                    prompt += f"涨跌幅：{latest.get('涨跌幅', '未知')}%\n"
                    prompt += f"开盘价：{latest.get('今开', '未知')} 元\n"
                    prompt += f"最高价：{latest.get('最高', '未知')} 元\n"
                    prompt += f"最低价：{latest.get('最低', '未知')} 元\n"
                    prompt += f"成交量：{latest.get('成交量', '未知')} 手\n"
                    prompt += f"成交额：{latest.get('成交额', '未知')} 元\n"
                    # 添加更多数据字段
                    prompt += f"昨收价：{latest.get('昨收', '未知')} 元\n"
                except IndexError:
                    logger.warning(f"获取股票 {stock_code} 报价数据时发生索引错误")
        
        # 股票历史数据 - 更详细分析
        stock_history = retrieved_info.get("data", {}).get("stock_history")
        if stock_history is not None and not stock_history.empty:
            prompt += "\n## 近期走势数据\n"
            # 计算不同周期的表现
            try:
                # 最近5天
                if len(stock_history) >= 5:
                    last_5_days = stock_history.tail(5)
                    if '收盘' in last_5_days.columns and '开盘' in last_5_days.columns:
                        change_5d = ((last_5_days['收盘'].iloc[-1] - last_5_days['开盘'].iloc[0]) / last_5_days['开盘'].iloc[0]) * 100
                        prompt += f"5日涨跌幅：{change_5d:.2f}%\n"
                # 最近10天
                if len(stock_history) >= 10:
                    last_10_days = stock_history.tail(10)
                    if '收盘' in last_10_days.columns and '开盘' in last_10_days.columns:
                        change_10d = ((last_10_days['收盘'].iloc[-1] - last_10_days['开盘'].iloc[0]) / last_10_days['开盘'].iloc[0]) * 100
                        prompt += f"10日涨跌幅：{change_10d:.2f}%\n"
                # 最近20天
                if len(stock_history) >= 20:
                    last_20_days = stock_history.tail(20)
                    if '收盘' in last_20_days.columns and '开盘' in last_20_days.columns:
                        change_20d = ((last_20_days['收盘'].iloc[-1] - last_20_days['开盘'].iloc[0]) / last_20_days['开盘'].iloc[0]) * 100
                        prompt += f"20日涨跌幅：{change_20d:.2f}%\n"
                # 计算最高价、最低价和平均成交量
                if '最高' in stock_history.columns:
                    prompt += f"近期最高价：{stock_history['最高'].max():.2f} 元\n"
                if '最低' in stock_history.columns:
                    prompt += f"近期最低价：{stock_history['最低'].min():.2f} 元\n"
                if '成交量' in stock_history.columns:
                    prompt += f"近期平均成交量：{stock_history['成交量'].mean():.2f} 手\n"
                if '成交额' in stock_history.columns:
                    prompt += f"近期平均成交额：{stock_history['成交额'].mean():.2f} 元\n"
                
                # 计算简单的技术指标
                if len(stock_history) >= 20:
                    try:
                        # 计算不同周期的均线
                        if '收盘' in stock_history.columns:
                            ma5 = stock_history['收盘'].rolling(5).mean().iloc[-1]
                            ma10 = stock_history['收盘'].rolling(10).mean().iloc[-1]
                            ma20 = stock_history['收盘'].rolling(20).mean().iloc[-1]
                            ma60 = stock_history['收盘'].rolling(60).mean().iloc[-1] if len(stock_history) >= 60 else '未知'
                            prompt += f"5日均线：{ma5:.2f} 元\n"
                            prompt += f"10日均线：{ma10:.2f} 元\n"
                            prompt += f"20日均线：{ma20:.2f} 元\n"
                            if ma60 != '未知':
                                prompt += f"60日均线：{ma60:.2f} 元\n"
                    except Exception as e:
                        logger.warning(f"计算技术指标时发生错误: {e}")
            except Exception as e:
                logger.warning(f"计算股票历史数据时发生错误: {e}")
                # 回退到简单展示
                if '最高' in stock_history.columns:
                    prompt += f"近期最高价：{stock_history['最高'].max():.2f} 元\n"
                if '最低' in stock_history.columns:
                    prompt += f"近期最低价：{stock_history['最低'].min():.2f} 元\n"
                if '成交量' in stock_history.columns:
                    prompt += f"近期平均成交量：{stock_history['成交量'].mean():.2f} 手\n"
                if '成交额' in stock_history.columns:
                    prompt += f"近期平均成交额：{stock_history['成交额'].mean():.2f} 元\n"
        
        # 行业数据对比
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业对比\n"
            top_sectors = industry_data.head(3)
            for i, (_, row) in enumerate(top_sectors.iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 技术指标分析\n"
        prompt += "   - MACD指标分析\n"
        prompt += "   - KDJ指标分析\n"
        prompt += "   - 均线系统分析（5/10/20/60日均线）\n"
        prompt += "   - RSI指标分析\n"
        prompt += "   - 布林带分析\n"
        prompt += "2. 成交量分析\n"
        prompt += "   - 成交量变化趋势\n"
        prompt += "   - 量价关系分析\n"
        prompt += "   - 换手率分析\n"
        prompt += "3. 支撑位和压力位分析\n"
        prompt += "   - 关键支撑位\n"
        prompt += "   - 关键压力位\n"
        prompt += "   - 突破可能性分析\n"
        prompt += "4. 技术形态分析\n"
        prompt += "   - K线形态识别\n"
        prompt += "   - 趋势线分析\n"
        prompt += "   - 形态突破分析\n"
        prompt += "5. 短期走势预测\n"
        prompt += "   - 1-3天走势预测\n"
        prompt += "   - 1-2周走势预测\n"
        prompt += "   - 关键技术点位\n"
        prompt += "6. 操作策略建议\n"
        prompt += "   - 买入/卖出时机建议\n"
        prompt += "   - 止损/止盈设置\n"
        prompt += "   - 仓位管理建议\n"
        prompt += "   - 风险控制措施\n"
        
        return prompt
    
    def _build_stock_valuation_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票估值分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票估值分析\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        
        prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        
        # 股票报价 - 更详细展示
        stock_quote = retrieved_info.get("data", {}).get("stock_quote", {})
        if stock_quote and stock_code in stock_quote:
            df = stock_quote[stock_code]
            if not df.empty and len(df) > 0:
                try:
                    latest = df.iloc[0]
                    prompt += "\n## 最新行情\n"
                    prompt += f"当前报价：{latest.get('最新价', '未知')} 元\n"
                    prompt += f"涨跌幅：{latest.get('涨跌幅', '未知')}%\n"
                    prompt += f"开盘价：{latest.get('今开', '未知')} 元\n"
                    prompt += f"最高价：{latest.get('最高', '未知')} 元\n"
                    prompt += f"最低价：{latest.get('最低', '未知')} 元\n"
                    prompt += f"成交量：{latest.get('成交量', '未知')} 手\n"
                    prompt += f"成交额：{latest.get('成交额', '未知')} 元\n"
                    # 添加更多数据字段
                    prompt += f"昨收价：{latest.get('昨收', '未知')} 元\n"
                except IndexError:
                    logger.warning(f"获取股票 {stock_code} 报价数据时发生索引错误")
        
        # 股票新闻 - 更结构化展示
        stock_news = retrieved_info.get("data", {}).get("stock_news")
        if stock_news is not None and not stock_news.empty:
            prompt += "\n## 相关新闻\n"
            for i, (_, row) in enumerate(stock_news.head(3).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                summary = row.get('摘要', '') or row.get('summary', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 行业数据对比
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业对比\n"
            # 显示行业整体表现
            top_sectors = industry_data.head(5)
            for i, (_, row) in enumerate(top_sectors.iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 估值方法分析\n"
        prompt += "   - 市盈率（P/E）分析\n"
        prompt += "   - 市净率（P/B）分析\n"
        prompt += "   - 市销率（P/S）分析\n"
        prompt += "   - 股息率分析\n"
        prompt += "   - 自由现金流贴现模型\n"
        prompt += "2. 相对估值分析\n"
        prompt += "   - 同行业公司对比\n"
        prompt += "   - 历史估值水平对比\n"
        prompt += "   - 市场平均估值对比\n"
        prompt += "3. 绝对估值分析\n"
        prompt += "   - 内在价值计算\n"
        prompt += "   - 估值区间确定\n"
        prompt += "   - 安全边际分析\n"
        prompt += "4. 估值影响因素\n"
        prompt += "   - 业绩增长预期\n"
        prompt += "   - 行业发展前景\n"
        prompt += "   - 宏观经济环境\n"
        prompt += "   - 政策因素影响\n"
        prompt += "5. 估值结论\n"
        prompt += "   - 当前估值状态（低估/合理/高估）\n"
        prompt += "   - 目标价位区间\n"
        prompt += "   - 投资建议\n"
        
        return prompt
    
    def _build_stock_performance_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票表现分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票表现分析\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        
        prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        
        # 股票报价 - 更详细展示
        stock_quote = retrieved_info.get("data", {}).get("stock_quote", {})
        if stock_quote and stock_code in stock_quote:
            df = stock_quote[stock_code]
            if not df.empty and len(df) > 0:
                try:
                    latest = df.iloc[0]
                    prompt += "\n## 最新行情\n"
                    prompt += f"当前报价：{latest.get('最新价', '未知')} 元\n"
                    prompt += f"涨跌幅：{latest.get('涨跌幅', '未知')}%\n"
                    prompt += f"开盘价：{latest.get('今开', '未知')} 元\n"
                    prompt += f"最高价：{latest.get('最高', '未知')} 元\n"
                    prompt += f"最低价：{latest.get('最低', '未知')} 元\n"
                    prompt += f"成交量：{latest.get('成交量', '未知')} 手\n"
                    prompt += f"成交额：{latest.get('成交额', '未知')} 元\n"
                    # 添加更多数据字段
                    prompt += f"昨收价：{latest.get('昨收', '未知')} 元\n"
                except IndexError:
                    logger.warning(f"获取股票 {stock_code} 报价数据时发生索引错误")
        
        # 股票历史数据 - 更详细分析
        stock_history = retrieved_info.get("data", {}).get("stock_history")
        if stock_history is not None and not stock_history.empty:
            prompt += "\n## 近期走势数据\n"
            # 计算不同周期的表现
            try:
                # 最近5天
                if len(stock_history) >= 5:
                    last_5_days = stock_history.tail(5)
                    if '收盘' in last_5_days.columns and '开盘' in last_5_days.columns:
                        change_5d = ((last_5_days['收盘'].iloc[-1] - last_5_days['开盘'].iloc[0]) / last_5_days['开盘'].iloc[0]) * 100
                        prompt += f"5日涨跌幅：{change_5d:.2f}%\n"
                # 最近10天
                if len(stock_history) >= 10:
                    last_10_days = stock_history.tail(10)
                    if '收盘' in last_10_days.columns and '开盘' in last_10_days.columns:
                        change_10d = ((last_10_days['收盘'].iloc[-1] - last_10_days['开盘'].iloc[0]) / last_10_days['开盘'].iloc[0]) * 100
                        prompt += f"10日涨跌幅：{change_10d:.2f}%\n"
                # 最近20天
                if len(stock_history) >= 20:
                    last_20_days = stock_history.tail(20)
                    if '收盘' in last_20_days.columns and '开盘' in last_20_days.columns:
                        change_20d = ((last_20_days['收盘'].iloc[-1] - last_20_days['开盘'].iloc[0]) / last_20_days['开盘'].iloc[0]) * 100
                        prompt += f"20日涨跌幅：{change_20d:.2f}%\n"
                # 计算最高价、最低价和平均成交量
                if '最高' in stock_history.columns:
                    prompt += f"近期最高价：{stock_history['最高'].max():.2f} 元\n"
                if '最低' in stock_history.columns:
                    prompt += f"近期最低价：{stock_history['最低'].min():.2f} 元\n"
                if '成交量' in stock_history.columns:
                    prompt += f"近期平均成交量：{stock_history['成交量'].mean():.2f} 手\n"
                if '成交额' in stock_history.columns:
                    prompt += f"近期平均成交额：{stock_history['成交额'].mean():.2f} 元\n"
            except Exception as e:
                logger.warning(f"计算股票历史数据时发生错误: {e}")
                # 回退到简单展示
                if '最高' in stock_history.columns:
                    prompt += f"近期最高价：{stock_history['最高'].max():.2f} 元\n"
                if '最低' in stock_history.columns:
                    prompt += f"近期最低价：{stock_history['最低'].min():.2f} 元\n"
                if '成交量' in stock_history.columns:
                    prompt += f"近期平均成交量：{stock_history['成交量'].mean():.2f} 手\n"
                if '成交额' in stock_history.columns:
                    prompt += f"近期平均成交额：{stock_history['成交额'].mean():.2f} 元\n"
        
        # 行业数据对比
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业对比\n"
            # 显示行业整体表现
            top_sectors = industry_data.head(5)
            for i, (_, row) in enumerate(top_sectors.iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 短期表现分析\n"
        prompt += "   - 1周表现\n"
        prompt += "   - 1月表现\n"
        prompt += "   - 3月表现\n"
        prompt += "2. 中期表现分析\n"
        prompt += "   - 6月表现\n"
        prompt += "   - 1年表现\n"
        prompt += "   - 2年表现\n"
        prompt += "3. 长期表现分析\n"
        prompt += "   - 3年表现\n"
        prompt += "   - 5年表现\n"
        prompt += "   - 10年表现\n"
        prompt += "4. 相对表现分析\n"
        prompt += "   - 与大盘指数对比\n"
        prompt += "   - 与行业平均对比\n"
        prompt += "   - 与同类公司对比\n"
        prompt += "5. 波动性分析\n"
        prompt += "   - 历史波动率\n"
        prompt += "   - 贝塔系数\n"
        prompt += "   - 最大回撤\n"
        prompt += "6. 表现驱动因素\n"
        prompt += "   - 业绩增长\n"
        prompt += "   - 估值变化\n"
        prompt += "   - 行业因素\n"
        prompt += "   - 宏观因素\n"
        prompt += "7. 未来表现预期\n"
        prompt += "   - 短期预期\n"
        prompt += "   - 中期预期\n"
        prompt += "   - 长期预期\n"
        prompt += "8. 投资建议\n"
        prompt += "   - 投资时机\n"
        prompt += "   - 仓位建议\n"
        prompt += "   - 风险控制\n"
        
        return prompt
    
    def _build_sector_fundamental_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建板块基本面分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n板块基本面分析\n\n"
        
        sector = entities.get("sector")
        
        if sector:
            prompt += f"板块名称：{sector}\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业整体表现\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 板块相关股票
        sector_stocks = retrieved_info.get("data", {}).get("sector_stocks")
        if sector_stocks is not None and not sector_stocks.empty:
            prompt += "\n## 板块成分股表现\n"
            for i, (_, row) in enumerate(sector_stocks.head(10).iterrows()):
                name = row.get('名称', '')
                code = row.get('代码', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}({code}): {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 板块基本情况\n"
        prompt += "   - 板块定义与范围\n"
        prompt += "   - 板块历史演变\n"
        prompt += "   - 板块地位与重要性\n"
        prompt += "2. 行业发展分析\n"
        prompt += "   - 行业发展阶段\n"
        prompt += "   - 行业增长趋势\n"
        prompt += "   - 行业生命周期\n"
        prompt += "3. 竞争格局分析\n"
        prompt += "   - 市场集中度\n"
        prompt += "   - 主要参与者\n"
        prompt += "   - 竞争优势分析\n"
        prompt += "4. 政策环境分析\n"
        prompt += "   - 行业政策\n"
        prompt += "   - 监管环境\n"
        prompt += "   - 政策影响评估\n"
        prompt += "5. 技术发展分析\n"
        prompt += "   - 技术创新趋势\n"
        prompt += "   - 技术对行业的影响\n"
        prompt += "   - 技术壁垒分析\n"
        prompt += "6. 板块内公司分析\n"
        prompt += "   - 龙头企业分析\n"
        prompt += "   - 成长型企业分析\n"
        prompt += "   - 价值型企业分析\n"
        prompt += "7. 投资机会分析\n"
        prompt += "   - 板块投资逻辑\n"
        prompt += "   - 潜在投资机会\n"
        prompt += "   - 风险因素分析\n"
        prompt += "8. 投资建议\n"
        prompt += "   - 板块配置建议\n"
        prompt += "   - 个股选择策略\n"
        prompt += "   - 投资时机建议\n"
        
        return prompt
    
    def _build_sector_technical_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建板块技术面分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n板块技术面分析\n\n"
        
        sector = entities.get("sector")
        
        if sector:
            prompt += f"板块名称：{sector}\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业整体表现\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 板块相关股票
        sector_stocks = retrieved_info.get("data", {}).get("sector_stocks")
        if sector_stocks is not None and not sector_stocks.empty:
            prompt += "\n## 板块成分股表现\n"
            for i, (_, row) in enumerate(sector_stocks.head(10).iterrows()):
                name = row.get('名称', '')
                code = row.get('代码', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}({code}): {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 板块整体技术走势\n"
        prompt += "   - 短期走势分析\n"
        prompt += "   - 中期走势分析\n"
        prompt += "   - 长期走势分析\n"
        prompt += "2. 板块量能分析\n"
        prompt += "   - 成交量变化趋势\n"
        prompt += "   - 量价关系分析\n"
        prompt += "   - 资金流向分析\n"
        prompt += "3. 板块技术指标分析\n"
        prompt += "   - 板块指数技术指标\n"
        prompt += "   - 板块内个股技术指标分布\n"
        prompt += "   - 技术指标背离分析\n"
        prompt += "4. 板块间技术对比\n"
        prompt += "   - 与其他板块对比\n"
        prompt += "   - 与大盘对比\n"
        prompt += "   - 相对强弱分析\n"
        prompt += "5. 板块轮动分析\n"
        prompt += "   - 板块轮动规律\n"
        prompt += "   - 当前轮动阶段\n"
        prompt += "   - 未来轮动预期\n"
        prompt += "6. 板块支撑位和压力位\n"
        prompt += "   - 关键支撑位\n"
        prompt += "   - 关键压力位\n"
        prompt += "   - 突破可能性分析\n"
        prompt += "7. 技术形态分析\n"
        prompt += "   - 板块指数形态\n"
        prompt += "   - 板块内个股形态分布\n"
        prompt += "   - 形态突破分析\n"
        prompt += "8. 投资策略建议\n"
        prompt += "   - 板块配置时机\n"
        prompt += "   - 板块内个股选择\n"
        prompt += "   - 风险控制措施\n"
        
        return prompt
    
    def _build_sector_performance_analysis_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建板块表现分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n板块表现分析\n\n"
        
        sector = entities.get("sector")
        
        if sector:
            prompt += f"板块名称：{sector}\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业整体表现\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 板块相关股票
        sector_stocks = retrieved_info.get("data", {}).get("sector_stocks")
        if sector_stocks is not None and not sector_stocks.empty:
            prompt += "\n## 板块成分股表现\n"
            for i, (_, row) in enumerate(sector_stocks.head(10).iterrows()):
                name = row.get('名称', '')
                code = row.get('代码', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}({code}): {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 板块短期表现\n"
        prompt += "   - 1周表现\n"
        prompt += "   - 1月表现\n"
        prompt += "   - 3月表现\n"
        prompt += "2. 板块中期表现\n"
        prompt += "   - 6月表现\n"
        prompt += "   - 1年表现\n"
        prompt += "   - 2年表现\n"
        prompt += "3. 板块长期表现\n"
        prompt += "   - 3年表现\n"
        prompt += "   - 5年表现\n"
        prompt += "   - 10年表现\n"
        prompt += "4. 板块相对表现\n"
        prompt += "   - 与大盘指数对比\n"
        prompt += "   - 与其他板块对比\n"
        prompt += "   - 相对强弱变化趋势\n"
        prompt += "5. 板块内个股表现\n"
        prompt += "   - 龙头股表现\n"
        prompt += "   - 成长股表现\n"
        prompt += "   - 价值股表现\n"
        prompt += "6. 板块表现驱动因素\n"
        prompt += "   - 行业基本面因素\n"
        prompt += "   - 资金流向因素\n"
        prompt += "   - 政策因素\n"
        prompt += "   - 事件驱动因素\n"
        prompt += "7. 板块表现周期\n"
        prompt += "   - 板块表现周期规律\n"
        prompt += "   - 当前周期阶段\n"
        prompt += "   - 未来周期预期\n"
        prompt += "8. 投资策略建议\n"
        prompt += "   - 板块配置建议\n"
        prompt += "   - 个股选择策略\n"
        prompt += "   - 投资时机建议\n"
        
        return prompt
    
    def _build_market_fundamental_analysis_prompt(self, retrieved_info: Dict, base_prompt: str) -> str:
        """构建大盘基本面分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n大盘基本面分析\n\n"
        
        # 市场概览 - 更详细展示
        market_overview = retrieved_info.get("data", {}).get("market_overview", {})
        if market_overview:
            prompt += "\n## 市场概览\n"
            prompt += f"总股票数：{market_overview.get('total_stocks', '未知')}\n"
            prompt += f"上涨股票数：{market_overview.get('up_stocks', '未知')}\n"
            prompt += f"下跌股票数：{market_overview.get('down_stocks', '未知')}\n"
            prompt += f"平盘股票数：{market_overview.get('flat_stocks', '未知')}\n"
            prompt += f"上涨比例：{market_overview.get('up_ratio', '未知')}%\n"
            prompt += f"下跌比例：{market_overview.get('down_ratio', '未知')}%\n"
            prompt += f"平均涨跌幅：{market_overview.get('avg_change', '未知')}%\n"
        
        # 指数数据 - 更详细展示
        index_data = retrieved_info.get("data", {}).get("index_data", {})
        if index_data:
            prompt += "\n## 主要指数表现\n"
            for index_name, df in index_data.items():
                if not df.empty:
                    latest = df.iloc[0]
                    name = latest.get('名称', index_name)
                    price = latest.get('最新价', '未知')
                    change = latest.get('涨跌幅', '未知')
                    prompt += f"- {name}: {price} ({change}%)\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现（前5名）\n"
            for i, (_, row) in enumerate(industry_data.head(5).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 市场新闻 - 更结构化展示
        market_news = retrieved_info.get("data", {}).get("market_news")
        if market_news is not None and not market_news.empty:
            prompt += "\n## 市场新闻\n"
            for i, (_, row) in enumerate(market_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 市场整体状况\n"
        prompt += "   - 市场规模与结构\n"
        prompt += "   - 市场流动性分析\n"
        prompt += "   - 市场估值水平\n"
        prompt += "2. 宏观经济分析\n"
        prompt += "   - GDP增长情况\n"
        prompt += "   - 通货膨胀水平\n"
        prompt += "   - 货币政策分析\n"
        prompt += "   - 财政政策分析\n"
        prompt += "3. 行业板块分析\n"
        prompt += "   - 行业板块表现\n"
        prompt += "   - 行业轮动分析\n"
        prompt += "   - 新兴行业分析\n"
        prompt += "4. 市场资金分析\n"
        prompt += "   - 资金流向分析\n"
        prompt += "   - 机构资金动向\n"
        prompt += "   - 外资流入流出\n"
        prompt += "5. 市场情绪分析\n"
        prompt += "   - 投资者情绪\n"
        prompt += "   - 市场风险偏好\n"
        prompt += "   - 技术指标情绪\n"
        prompt += "6. 基本面综合评估\n"
        prompt += "   - 市场估值水平\n"
        prompt += "   - 市场结构分析\n"
        prompt += "   - 潜在风险因素\n"
        prompt += "7. 投资策略建议\n"
        prompt += "   - 资产配置建议\n"
        prompt += "   - 行业配置建议\n"
        prompt += "   - 投资时机建议\n"
        prompt += "   - 风险控制措施\n"
        
        return prompt
    
    def _build_market_technical_analysis_prompt(self, retrieved_info: Dict, base_prompt: str) -> str:
        """构建大盘技术面分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n大盘技术面分析\n\n"
        
        # 指数数据 - 更详细展示
        index_data = retrieved_info.get("data", {}).get("index_data", {})
        if index_data:
            prompt += "\n## 主要指数表现\n"
            for index_name, df in index_data.items():
                if not df.empty:
                    latest = df.iloc[0]
                    name = latest.get('名称', index_name)
                    price = latest.get('最新价', '未知')
                    change = latest.get('涨跌幅', '未知')
                    open_price = latest.get('今开', '未知')
                    high_price = latest.get('最高', '未知')
                    low_price = latest.get('最低', '未知')
                    prompt += f"- {name}: {price} ({change}%)\n"
                    prompt += f"  开盘：{open_price}, 最高：{high_price}, 最低：{low_price}\n"
        
        # 市场概览
        market_overview = retrieved_info.get("data", {}).get("market_overview", {})
        if market_overview:
            prompt += "\n## 市场概览\n"
            prompt += f"上涨股票数：{market_overview.get('up_stocks', '未知')}\n"
            prompt += f"下跌股票数：{market_overview.get('down_stocks', '未知')}\n"
            prompt += f"上涨比例：{market_overview.get('up_ratio', '未知')}%\n"
            prompt += f"下跌比例：{market_overview.get('down_ratio', '未知')}%\n"
            prompt += f"平均涨跌幅：{market_overview.get('avg_change', '未知')}%\n"
        
        # 行业数据
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现（前5名）\n"
            for i, (_, row) in enumerate(industry_data.head(5).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 主要指数技术分析\n"
        prompt += "   - 上证指数技术分析\n"
        prompt += "   - 深证成指技术分析\n"
        prompt += "   - 创业板指技术分析\n"
        prompt += "2. 市场量能分析\n"
        prompt += "   - 成交量变化趋势\n"
        prompt += "   - 成交额分析\n"
        prompt += "   - 量价关系分析\n"
        prompt += "3. 技术形态分析\n"
        prompt += "   - 大盘指数形态\n"
        prompt += "   - 支撑位和压力位分析\n"
        prompt += "   - 技术突破分析\n"
        prompt += "4. 重要技术指标分析\n"
        prompt += "   - MACD指标分析\n"
        prompt += "   - KDJ指标分析\n"
        prompt += "   - RSI指标分析\n"
        prompt += "   - 布林带分析\n"
        prompt += "5. 短期技术走势预测\n"
        prompt += "   - 1-3天走势预测\n"
        prompt += "   - 1-2周走势预测\n"
        prompt += "   - 关键技术点位\n"
        prompt += "6. 操作策略建议\n"
        prompt += "   - 仓位管理建议\n"
        prompt += "   - 行业配置建议\n"
        prompt += "   - 个股选择策略\n"
        prompt += "   - 风险控制措施\n"
        
        return prompt
    
    def _build_market_performance_analysis_prompt(self, retrieved_info: Dict, base_prompt: str) -> str:
        """构建大盘表现分析提示词"""
        prompt = base_prompt + "\n# 分析类型\n大盘表现分析\n\n"
        
        # 市场概览 - 更详细展示
        market_overview = retrieved_info.get("data", {}).get("market_overview", {})
        if market_overview:
            prompt += "\n## 市场概览\n"
            prompt += f"总股票数：{market_overview.get('total_stocks', '未知')}\n"
            prompt += f"上涨股票数：{market_overview.get('up_stocks', '未知')}\n"
            prompt += f"下跌股票数：{market_overview.get('down_stocks', '未知')}\n"
            prompt += f"平盘股票数：{market_overview.get('flat_stocks', '未知')}\n"
            prompt += f"上涨比例：{market_overview.get('up_ratio', '未知')}%\n"
            prompt += f"下跌比例：{market_overview.get('down_ratio', '未知')}%\n"
            prompt += f"平均涨跌幅：{market_overview.get('avg_change', '未知')}%\n"
        
        # 指数数据 - 更详细展示
        index_data = retrieved_info.get("data", {}).get("index_data", {})
        if index_data:
            prompt += "\n## 主要指数表现\n"
            for index_name, df in index_data.items():
                if not df.empty:
                    latest = df.iloc[0]
                    name = latest.get('名称', index_name)
                    price = latest.get('最新价', '未知')
                    change = latest.get('涨跌幅', '未知')
                    prompt += f"- {name}: {price} ({change}%)\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现（前5名）\n"
            for i, (_, row) in enumerate(industry_data.head(5).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 大盘短期表现\n"
        prompt += "   - 1周表现\n"
        prompt += "   - 1月表现\n"
        prompt += "   - 3月表现\n"
        prompt += "2. 大盘中期表现\n"
        prompt += "   - 6月表现\n"
        prompt += "   - 1年表现\n"
        prompt += "   - 2年表现\n"
        prompt += "3. 大盘长期表现\n"
        prompt += "   - 3年表现\n"
        prompt += "   - 5年表现\n"
        prompt += "   - 10年表现\n"
        prompt += "4. 大盘相对表现\n"
        prompt += "   - 与国际市场对比\n"
        prompt += "   - 与历史表现对比\n"
        prompt += "   - 相对强弱分析\n"
        prompt += "5. 市场内部表现\n"
        prompt += "   - 行业板块表现\n"
        prompt += "   - 个股表现分布\n"
        prompt += "   - 市场广度分析\n"
        prompt += "6. 市场表现驱动因素\n"
        prompt += "   - 宏观经济因素\n"
        prompt += "   - 政策因素\n"
        prompt += "   - 资金流向因素\n"
        prompt += "   - 事件驱动因素\n"
        prompt += "7. 市场表现周期\n"
        prompt += "   - 市场周期分析\n"
        prompt += "   - 当前周期阶段\n"
        prompt += "   - 未来周期预期\n"
        prompt += "8. 投资策略建议\n"
        prompt += "   - 资产配置建议\n"
        prompt += "   - 行业配置建议\n"
        prompt += "   - 投资时机建议\n"
        prompt += "   - 风险控制措施\n"
        
        return prompt
    
    def _build_news_query_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建新闻查询提示词"""
        prompt = base_prompt + "\n# 分析类型\n新闻查询\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        sector = entities.get("sector")
        query = entities.get("query", "")
        
        if stock_code:
            prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        if sector:
            prompt += f"板块名称：{sector}\n"
        if query:
            prompt += f"查询关键词：{query}\n"
        
        # 股票新闻 - 更结构化展示
        stock_news = retrieved_info.get("data", {}).get("stock_news")
        if stock_news is not None and not stock_news.empty:
            prompt += "\n## 相关股票新闻\n"
            for i, (_, row) in enumerate(stock_news.head(10).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                summary = row.get('摘要', '') or row.get('summary', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 市场新闻 - 更结构化展示
        market_news = retrieved_info.get("data", {}).get("market_news")
        if market_news is not None and not market_news.empty:
            prompt += "\n## 市场新闻\n"
            for i, (_, row) in enumerate(market_news.head(10).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 行业新闻 - 更结构化展示
        industry_news = retrieved_info.get("data", {}).get("industry_news")
        if industry_news is not None and not industry_news.empty:
            prompt += "\n## 行业新闻\n"
            for i, (_, row) in enumerate(industry_news.head(10).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 互联网新闻
        internet_news = []
        if retrieved_info.get("data", {}).get("internet_stock_news"):
            internet_news.extend(retrieved_info["data"]["internet_stock_news"])
        if retrieved_info.get("data", {}).get("internet_sector_news"):
            internet_news.extend(retrieved_info["data"]["internet_sector_news"])
        if retrieved_info.get("data", {}).get("internet_market_news"):
            internet_news.extend(retrieved_info["data"]["internet_market_news"])
        if retrieved_info.get("data", {}).get("internet_industry_news"):
            internet_news.extend(retrieved_info["data"]["internet_industry_news"])
        if retrieved_info.get("data", {}).get("internet_finance_news"):
            internet_news.extend(retrieved_info["data"]["internet_finance_news"])
        
        if internet_news:
            prompt += "\n## 互联网相关新闻\n"
            for i, news in enumerate(internet_news[:5], 1):
                title = news.get('title', '')
                source = news.get('source', '')
                summary = news.get('summary', '')
                prompt += f"{i}. {title}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 新闻摘要\n"
        prompt += "   - 相关新闻概览\n"
        prompt += "   - 重要新闻亮点\n"
        prompt += "2. 详细新闻信息\n"
        prompt += "   - 股票相关新闻\n"
        prompt += "   - 市场相关新闻\n"
        prompt += "   - 行业相关新闻\n"
        prompt += "3. 新闻影响分析\n"
        prompt += "   - 对相关股票的影响\n"
        prompt += "   - 对市场的影响\n"
        prompt += "   - 对行业的影响\n"
        prompt += "4. 投资建议\n"
        prompt += "   - 基于新闻的投资机会\n"
        prompt += "   - 风险提示\n"
        prompt += "   - 操作建议\n"
        
        return prompt
    
    def _build_stock_info_query_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票信息查询提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票信息查询\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        
        prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        
        # 股票报价 - 更详细展示
        stock_quote = retrieved_info.get("data", {}).get("stock_quote", {})
        if stock_quote and stock_code in stock_quote:
            df = stock_quote[stock_code]
            if not df.empty and len(df) > 0:
                try:
                    latest = df.iloc[0]
                    prompt += "\n## 最新行情\n"
                    prompt += f"当前报价：{latest.get('最新价', '未知')} 元\n"
                    prompt += f"涨跌幅：{latest.get('涨跌幅', '未知')}%\n"
                    prompt += f"开盘价：{latest.get('今开', '未知')} 元\n"
                    prompt += f"最高价：{latest.get('最高', '未知')} 元\n"
                    prompt += f"最低价：{latest.get('最低', '未知')} 元\n"
                    prompt += f"成交量：{latest.get('成交量', '未知')} 手\n"
                    prompt += f"成交额：{latest.get('成交额', '未知')} 元\n"
                except IndexError:
                    logger.warning(f"获取股票 {stock_code} 报价数据时发生索引错误")
        
        # 股票新闻 - 更结构化展示
        stock_news = retrieved_info.get("data", {}).get("stock_news")
        if stock_news is not None and not stock_news.empty:
            prompt += "\n## 相关新闻\n"
            for i, (_, row) in enumerate(stock_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                summary = row.get('摘要', '') or row.get('summary', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 行业数据对比
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业对比\n"
            # 显示行业整体表现
            top_sectors = industry_data.head(5)
            for i, (_, row) in enumerate(top_sectors.iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 股票基本信息\n"
        prompt += "   - 股票代码与名称\n"
        prompt += "   - 所属行业与板块\n"
        prompt += "   - 公司简介\n"
        prompt += "2. 最新行情\n"
        prompt += "   - 当前价格与涨跌幅\n"
        prompt += "   - 今日走势\n"
        prompt += "   - 成交量与成交额\n"
        prompt += "3. 近期表现\n"
        prompt += "   - 短期表现\n"
        prompt += "   - 中期表现\n"
        prompt += "   - 长期表现\n"
        prompt += "4. 行业对比\n"
        prompt += "   - 与行业平均对比\n"
        prompt += "   - 与板块对比\n"
        prompt += "5. 相关新闻\n"
        prompt += "   - 最新新闻\n"
        prompt += "   - 新闻影响分析\n"
        prompt += "6. 投资建议\n"
        prompt += "   - 投资价值评估\n"
        prompt += "   - 风险提示\n"
        prompt += "   - 操作建议\n"
        
        return prompt
    
    def _build_sector_info_query_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建板块信息查询提示词"""
        prompt = base_prompt + "\n# 分析类型\n板块信息查询\n\n"
        
        sector = entities.get("sector")
        
        if sector:
            prompt += f"板块名称：{sector}\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业整体表现\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 板块相关股票
        sector_stocks = retrieved_info.get("data", {}).get("sector_stocks")
        if sector_stocks is not None and not sector_stocks.empty:
            prompt += "\n## 板块成分股表现\n"
            for i, (_, row) in enumerate(sector_stocks.head(10).iterrows()):
                name = row.get('名称', '')
                code = row.get('代码', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}({code}): {change}%\n"
        
        # 行业新闻 - 更结构化展示
        industry_news = retrieved_info.get("data", {}).get("industry_news")
        if industry_news is not None and not industry_news.empty:
            prompt += "\n## 行业新闻\n"
            for i, (_, row) in enumerate(industry_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 板块基本信息\n"
        prompt += "   - 板块定义与范围\n"
        prompt += "   - 板块历史演变\n"
        prompt += "   - 板块地位与重要性\n"
        prompt += "2. 板块表现\n"
        prompt += "   - 最新表现\n"
        prompt += "   - 短期表现\n"
        prompt += "   - 中期表现\n"
        prompt += "   - 长期表现\n"
        prompt += "3. 板块成分股\n"
        prompt += "   - 主要成分股\n"
        prompt += "   - 成分股表现\n"
        prompt += "   - 龙头股分析\n"
        prompt += "4. 行业对比\n"
        prompt += "   - 与其他行业对比\n"
        prompt += "   - 行业地位分析\n"
        prompt += "5. 相关新闻\n"
        prompt += "   - 行业最新新闻\n"
        prompt += "   - 新闻影响分析\n"
        prompt += "6. 投资建议\n"
        prompt += "   - 板块投资价值\n"
        prompt += "   - 风险提示\n"
        prompt += "   - 操作建议\n"
        
        return prompt
    
    def _build_market_info_query_prompt(self, retrieved_info: Dict, base_prompt: str) -> str:
        """构建市场信息查询提示词"""
        prompt = base_prompt + "\n# 分析类型\n市场信息查询\n\n"
        
        # 市场概览 - 更详细展示
        market_overview = retrieved_info.get("data", {}).get("market_overview", {})
        if market_overview:
            prompt += "\n## 市场概览\n"
            prompt += f"总股票数：{market_overview.get('total_stocks', '未知')}\n"
            prompt += f"上涨股票数：{market_overview.get('up_stocks', '未知')}\n"
            prompt += f"下跌股票数：{market_overview.get('down_stocks', '未知')}\n"
            prompt += f"平盘股票数：{market_overview.get('flat_stocks', '未知')}\n"
            prompt += f"上涨比例：{market_overview.get('up_ratio', '未知')}%\n"
            prompt += f"下跌比例：{market_overview.get('down_ratio', '未知')}%\n"
            prompt += f"平均涨跌幅：{market_overview.get('avg_change', '未知')}%\n"
        
        # 指数数据 - 更详细展示
        index_data = retrieved_info.get("data", {}).get("index_data", {})
        if index_data:
            prompt += "\n## 主要指数表现\n"
            for index_name, df in index_data.items():
                if not df.empty:
                    latest = df.iloc[0]
                    name = latest.get('名称', index_name)
                    price = latest.get('最新价', '未知')
                    change = latest.get('涨跌幅', '未知')
                    prompt += f"- {name}: {price} ({change}%)\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现（前5名）\n"
            for i, (_, row) in enumerate(industry_data.head(5).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 市场新闻 - 更结构化展示
        market_news = retrieved_info.get("data", {}).get("market_news")
        if market_news is not None and not market_news.empty:
            prompt += "\n## 市场新闻\n"
            for i, (_, row) in enumerate(market_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 市场概览\n"
        prompt += "   - 市场整体状况\n"
        prompt += "   - 涨跌分布\n"
        prompt += "   - 市场活跃度\n"
        prompt += "2. 主要指数表现\n"
        prompt += "   - 上证指数表现\n"
        prompt += "   - 深证成指表现\n"
        prompt += "   - 创业板指表现\n"
        prompt += "3. 行业板块表现\n"
        prompt += "   - 领涨行业\n"
        prompt += "   - 领跌行业\n"
        prompt += "   - 行业轮动分析\n"
        prompt += "4. 市场热点\n"
        prompt += "   - 热门板块\n"
        prompt += "   - 热门个股\n"
        prompt += "   - 资金流向\n"
        prompt += "5. 市场新闻\n"
        prompt += "   - 重要市场新闻\n"
        prompt += "   - 新闻影响分析\n"
        prompt += "6. 投资建议\n"
        prompt += "   - 市场趋势判断\n"
        prompt += "   - 投资机会分析\n"
        prompt += "   - 风险提示\n"
        prompt += "   - 操作建议\n"
        
        return prompt
    
    def _build_stock_list_prompt(self, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票列表提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票列表查询\n\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 板块相关股票
        sector_stocks = retrieved_info.get("data", {}).get("sector_stocks")
        if sector_stocks is not None and not sector_stocks.empty:
            prompt += "\n## 板块成分股\n"
            for i, (_, row) in enumerate(sector_stocks.head(20).iterrows()):
                name = row.get('名称', '')
                code = row.get('代码', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}({code}): {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 股票列表概览\n"
        prompt += "   - 行业分布\n"
        prompt += "   - 涨跌幅分布\n"
        prompt += "2. 详细股票列表\n"
        prompt += "   - 股票代码与名称\n"
        prompt += "   - 最新价格\n"
        prompt += "   - 涨跌幅\n"
        prompt += "   - 所属行业\n"
        prompt += "3. 投资机会分析\n"
        prompt += "   - 潜力股分析\n"
        prompt += "   - 风险提示\n"
        prompt += "4. 操作建议\n"
        prompt += "   - 选股策略\n"
        prompt += "   - 投资时机\n"
        
        return prompt
    
    def _build_stock_recommendation_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建股票推荐提示词"""
        prompt = base_prompt + "\n# 分析类型\n股票推荐\n\n"
        
        sector = entities.get("sector")
        query = entities.get("query", "")
        
        if sector:
            prompt += f"推荐板块：{sector}\n"
        if query:
            prompt += f"推荐关键词：{query}\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 板块相关股票
        sector_stocks = retrieved_info.get("data", {}).get("sector_stocks")
        if sector_stocks is not None and not sector_stocks.empty:
            prompt += "\n## 板块成分股\n"
            for i, (_, row) in enumerate(sector_stocks.head(20).iterrows()):
                name = row.get('名称', '')
                code = row.get('代码', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}({code}): {change}%\n"
        
        # 市场新闻 - 更结构化展示
        market_news = retrieved_info.get("data", {}).get("market_news")
        if market_news is not None and not market_news.empty:
            prompt += "\n## 市场新闻\n"
            for i, (_, row) in enumerate(market_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 推荐理由\n"
        prompt += "   - 市场环境分析\n"
        prompt += "   - 行业发展前景\n"
        prompt += "   - 选股逻辑\n"
        prompt += "2. 推荐股票列表\n"
        prompt += "   - 股票代码与名称\n"
        prompt += "   - 推荐理由\n"
        prompt += "   - 目标价位\n"
        prompt += "   - 风险等级\n"
        prompt += "3. 投资策略\n"
        prompt += "   - 买入时机\n"
        prompt += "   - 仓位建议\n"
        prompt += "   - 持有周期\n"
        prompt += "4. 风险提示\n"
        prompt += "   - 市场风险\n"
        prompt += "   - 行业风险\n"
        prompt += "   - 个股风险\n"
        prompt += "5. 操作建议\n"
        prompt += "   - 止损止盈设置\n"
        prompt += "   - 调仓策略\n"
        prompt += "   - 监控指标\n"
        
        return prompt
    
    def _build_sector_recommendation_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建板块推荐提示词"""
        prompt = base_prompt + "\n# 分析类型\n板块推荐\n\n"
        
        query = entities.get("query", "")
        
        if query:
            prompt += f"推荐关键词：{query}\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现\n"
            for i, (_, row) in enumerate(industry_data.head(20).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 市场新闻 - 更结构化展示
        market_news = retrieved_info.get("data", {}).get("market_news")
        if market_news is not None and not market_news.empty:
            prompt += "\n## 市场新闻\n"
            for i, (_, row) in enumerate(market_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 推荐理由\n"
        prompt += "   - 市场环境分析\n"
        prompt += "   - 板块轮动分析\n"
        prompt += "   - 推荐逻辑\n"
        prompt += "2. 推荐板块列表\n"
        prompt += "   - 板块名称\n"
        prompt += "   - 推荐理由\n"
        prompt += "   - 预期表现\n"
        prompt += "   - 风险等级\n"
        prompt += "3. 投资策略\n"
        prompt += "   - 配置比例\n"
        prompt += "   - 买入时机\n"
        prompt += "   - 持有周期\n"
        prompt += "4. 风险提示\n"
        prompt += "   - 市场风险\n"
        prompt += "   - 行业风险\n"
        prompt += "   - 政策风险\n"
        prompt += "5. 操作建议\n"
        prompt += "   - 板块内个股选择\n"
        prompt += "   - 调仓策略\n"
        prompt += "   - 监控指标\n"
        
        return prompt
    
    def _build_portfolio_recommendation_prompt(self, retrieved_info: Dict, base_prompt: str) -> str:
        """构建投资组合推荐提示词"""
        prompt = base_prompt + "\n# 分析类型\n投资组合推荐\n\n"
        
        # 市场概览 - 更详细展示
        market_overview = retrieved_info.get("data", {}).get("market_overview", {})
        if market_overview:
            prompt += "\n## 市场概览\n"
            prompt += f"总股票数：{market_overview.get('total_stocks', '未知')}\n"
            prompt += f"上涨股票数：{market_overview.get('up_stocks', '未知')}\n"
            prompt += f"下跌股票数：{market_overview.get('down_stocks', '未知')}\n"
            prompt += f"上涨比例：{market_overview.get('up_ratio', '未知')}%\n"
            prompt += f"下跌比例：{market_overview.get('down_ratio', '未知')}%\n"
            prompt += f"平均涨跌幅：{market_overview.get('avg_change', '未知')}%\n"
        
        # 指数数据 - 更详细展示
        index_data = retrieved_info.get("data", {}).get("index_data", {})
        if index_data:
            prompt += "\n## 主要指数表现\n"
            for index_name, df in index_data.items():
                if not df.empty:
                    latest = df.iloc[0]
                    name = latest.get('名称', index_name)
                    price = latest.get('最新价', '未知')
                    change = latest.get('涨跌幅', '未知')
                    prompt += f"- {name}: {price} ({change}%)\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现（前10名）\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 市场环境分析\n"
        prompt += "   - 市场整体状况\n"
        prompt += "   - 行业板块表现\n"
        prompt += "   - 市场趋势判断\n"
        prompt += "2. 投资组合构建\n"
        prompt += "   - 资产配置比例\n"
        prompt += "   - 行业配置策略\n"
        prompt += "   - 个股选择逻辑\n"
        prompt += "3. 推荐投资组合\n"
        prompt += "   - 核心配置\n"
        prompt += "   - 卫星配置\n"
        prompt += "   - 防御性配置\n"
        prompt += "4. 投资策略\n"
        prompt += "   - 买入时机\n"
        prompt += "   - 持有周期\n"
        prompt += "   - 调仓策略\n"
        prompt += "5. 风险控制\n"
        prompt += "   - 风险评估\n"
        prompt += "   - 止损策略\n"
        prompt += "   - 风险对冲措施\n"
        prompt += "6. 预期收益\n"
        prompt += "   - 短期预期\n"
        prompt += "   - 中期预期\n"
        prompt += "   - 长期预期\n"
        
        return prompt
    
    def _build_report_generation_prompt(self, entities: Dict, retrieved_info: Dict, base_prompt: str) -> str:
        """构建报告生成提示词"""
        prompt = base_prompt + "\n# 分析类型\n报告生成\n\n"
        
        stock_code = entities.get("stock_code")
        stock_name = entities.get("stock_name")
        sector = entities.get("sector")
        query = entities.get("query", "")
        
        if stock_code:
            prompt += f"股票代码：{stock_code}\n"
        if stock_name:
            prompt += f"股票名称：{stock_name}\n"
        if sector:
            prompt += f"板块名称：{sector}\n"
        if query:
            prompt += f"报告关键词：{query}\n"
        
        # 股票报价 - 更详细展示
        if stock_code:
            stock_quote = retrieved_info.get("data", {}).get("stock_quote", {})
            if stock_quote and stock_code in stock_quote:
                df = stock_quote[stock_code]
                if not df.empty and len(df) > 0:
                    try:
                        latest = df.iloc[0]
                        prompt += "\n## 最新行情\n"
                        prompt += f"当前报价：{latest.get('最新价', '未知')} 元\n"
                        prompt += f"涨跌幅：{latest.get('涨跌幅', '未知')}%\n"
                        prompt += f"开盘价：{latest.get('今开', '未知')} 元\n"
                        prompt += f"最高价：{latest.get('最高', '未知')} 元\n"
                        prompt += f"最低价：{latest.get('最低', '未知')} 元\n"
                        prompt += f"成交量：{latest.get('成交量', '未知')} 手\n"
                        prompt += f"成交额：{latest.get('成交额', '未知')} 元\n"
                    except IndexError:
                        logger.warning(f"获取股票 {stock_code} 报价数据时发生索引错误")
        
        # 市场概览 - 更详细展示
        market_overview = retrieved_info.get("data", {}).get("market_overview", {})
        if market_overview:
            prompt += "\n## 市场概览\n"
            prompt += f"总股票数：{market_overview.get('total_stocks', '未知')}\n"
            prompt += f"上涨股票数：{market_overview.get('up_stocks', '未知')}\n"
            prompt += f"下跌股票数：{market_overview.get('down_stocks', '未知')}\n"
            prompt += f"上涨比例：{market_overview.get('up_ratio', '未知')}%\n"
            prompt += f"下跌比例：{market_overview.get('down_ratio', '未知')}%\n"
            prompt += f"平均涨跌幅：{market_overview.get('avg_change', '未知')}%\n"
        
        # 指数数据 - 更详细展示
        index_data = retrieved_info.get("data", {}).get("index_data", {})
        if index_data:
            prompt += "\n## 主要指数表现\n"
            for index_name, df in index_data.items():
                if not df.empty:
                    latest = df.iloc[0]
                    name = latest.get('名称', index_name)
                    price = latest.get('最新价', '未知')
                    change = latest.get('涨跌幅', '未知')
                    prompt += f"- {name}: {price} ({change}%)\n"
        
        # 行业数据 - 更详细展示
        industry_data = retrieved_info.get("data", {}).get("industry_data")
        if industry_data is not None and not industry_data.empty:
            prompt += "\n## 行业表现（前10名）\n"
            for i, (_, row) in enumerate(industry_data.head(10).iterrows()):
                name = row.get('名称', '')
                change = row.get('涨跌幅', '未知')
                prompt += f"{i+1}. {name}: {change}%\n"
        
        # 股票新闻 - 更结构化展示
        stock_news = retrieved_info.get("data", {}).get("stock_news")
        if stock_news is not None and not stock_news.empty:
            prompt += "\n## 相关新闻\n"
            for i, (_, row) in enumerate(stock_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                summary = row.get('摘要', '') or row.get('summary', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
                if summary:
                    prompt += f"   摘要：{summary}\n"
        
        # 市场新闻 - 更结构化展示
        market_news = retrieved_info.get("data", {}).get("market_news")
        if market_news is not None and not market_news.empty:
            prompt += "\n## 市场新闻\n"
            for i, (_, row) in enumerate(market_news.head(5).iterrows()):
                title = row.get('title', '')
                time = row.get('发布时间', '') or row.get('publish_time', '')
                source = row.get('来源', '') or row.get('source', '')
                prompt += f"{i+1}. {title}\n"
                prompt += f"   时间：{time}\n"
                if source:
                    prompt += f"   来源：{source}\n"
        
        # 更详细的输出结构要求
        prompt += "\n\n# 输出结构要求\n请按照以下结构组织回答：\n"
        prompt += "1. 报告摘要\n"
        prompt += "   - 核心观点\n"
        prompt += "   - 关键发现\n"
        prompt += "   - 投资建议\n"
        prompt += "2. 市场环境分析\n"
        prompt += "   - 宏观经济状况\n"
        prompt += "   - 市场整体表现\n"
        prompt += "   - 行业板块表现\n"
        prompt += "3. 详细分析\n"
        prompt += "   - 股票/板块分析\n"
        prompt += "   - 基本面分析\n"
        prompt += "   - 技术面分析\n"
        prompt += "   - 估值分析\n"
        prompt += "4. 风险分析\n"
        prompt += "   - 市场风险\n"
        prompt += "   - 行业风险\n"
        prompt += "   - 个股风险\n"
        prompt += "5. 投资策略\n"
        prompt += "   - 资产配置建议\n"
        prompt += "   - 行业配置建议\n"
        prompt += "   - 个股选择策略\n"
        prompt += "   - 买入时机建议\n"
        prompt += "6. 结论与建议\n"
        prompt += "   - 综合结论\n"
        prompt += "   - 具体建议\n"
        prompt += "   - 后续跟踪要点\n"
        
        return prompt


information_retriever = InformationRetriever()