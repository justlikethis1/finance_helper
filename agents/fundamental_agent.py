import pandas as pd
import numpy as np
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from agents.data_agent import data_agent
from backend.model import model_manager

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FundamentalAnalysisAgent:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = 3600  # 缓存1小时
    
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
    
    def _cache_result(self, key: str, result: Dict) -> None:
        """缓存结果"""
        self.cache[key] = (time.time(), result)
    
    async def analyze_company(self, symbol: str) -> Dict:
        """分析公司基本面"""
        try:
            cache_key = self._get_cache_key("analyze_company", symbol)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # 1. 获取公司基本信息
            company_info = data_agent.get_company_info(symbol)
            if not company_info:
                logger.error(f"获取公司基本信息失败: {symbol}")
                return {
                    "valuation": "",
                    "growth": "",
                    "financial_health": "",
                    "summary": ""
                }
            
            # 2. 获取公司所属指数
            company_indices = data_agent.get_company_indices(symbol)
            
            # 3. 获取近一年各个季度的利润数据
            quarterly_profit = data_agent.get_quarterly_profit(symbol)
            
            # 4. 获取资产负债表数据
            # 设置时间范围：过去1年
            end_time = datetime.now().strftime("%Y%m%d")
            start_time = (datetime.now() - timedelta(days=365)).strftime("%Y%m%d")
            balance_sheet = data_agent.get_balance_sheet(symbol, start_time, end_time)
            
            # 5. 获取公司主要指标
            company_metrics = data_agent.get_company_metrics(symbol, start_time, end_time)
            
            # 6. 获取股票实时行情数据
            real_time_transaction = data_agent.get_real_time_transaction(symbol)
            
            # 7. 获取股票历史数据
            stock_history = data_agent.get_stock_history(symbol, days=365)
            
            # 8. 获取详细财务指标数据
            financial_indicators = data_agent.get_financial_indicators(symbol)
            
            # 9. 计算PE和PB
            pe_pb_data = data_agent.calculate_pe_pb(symbol)
            
            # 10. 获取行业财务数据（用于财务造假风险检测）
            industry_financial_data = None
            if company_info and '概念及板块' in company_info:
                industry_info_str = company_info['概念及板块']
                industry_name = industry_info_str.split(' ')[0] if industry_info_str else '未知行业'
                industry_financial_data = data_agent.get_industry_financial_data(industry_name)
            
            # 9. 构建分析提示词
            prompt = self._build_analysis_prompt(symbol, company_info, company_indices, quarterly_profit, 
                                                balance_sheet, company_metrics, real_time_transaction, stock_history, financial_indicators, pe_pb_data)
            
            # 10. 调用模型生成分析
            analysis = await model_manager.async_generate_response(prompt)
            
            # 11. 解析分析结果
            result = self._parse_analysis(analysis)
            
            # 添加股票名称到结果中
            if company_info and '公司名称' in company_info:
                result['stock_name'] = company_info['公司名称']
            
            # 添加PE和PB数据到结果中
            if pe_pb_data:
                result['pe_ttm'] = pe_pb_data.get('pe_ttm', '无法计算')
                result['pb'] = pe_pb_data.get('pb', '无法计算')
                result['valuation_status'] = pe_pb_data.get('valuation_status', '无法判断')
                result['valuation_detail'] = pe_pb_data.get('valuation_detail', '无')
            
            # 12. 检测财务造假风险
            financial_risks = self._detect_financial_fraud_risks(balance_sheet, company_metrics, industry_financial_data)
            if financial_risks:
                result['financial_risk'] = '; '.join(financial_risks)
                # 在综合评价中添加风险提示
                if 'summary' in result:
                    result['summary'] = f"【财务风险提示】{'; '.join(financial_risks)}。\n" + result['summary']
            
            # 11. 缓存结果
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"基本面分析失败: {e}")
            return {
                "valuation": "",
                "growth": "",
                "financial_health": "",
                "summary": ""
            }
    
    def _build_analysis_prompt(self, symbol: str, company_info: Dict, company_indices: List[Dict], quarterly_profit: List[Dict], 
                              balance_sheet: List[Dict], company_metrics: List[Dict], real_time_transaction: Optional[Dict], 
                              stock_history: pd.DataFrame, financial_indicators: List[Dict] = None, pe_pb_data: Dict = None) -> str:
        """构建基本面分析提示词"""
        # 构建公司基本信息描述
        financial_desc = f"股票代码: {symbol}\n"
        
        # 公司基本信息
        if company_info:
            financial_desc += "\n公司基本信息：\n"
            if '公司名称' in company_info:
                financial_desc += f"公司名称: {company_info['公司名称']}\n"
            if '上市市场' in company_info:
                financial_desc += f"上市市场: {company_info['上市市场']}\n"
            if '概念及板块' in company_info:
                financial_desc += f"概念及板块: {company_info['概念及板块']}\n"
            if '成立日期' in company_info:
                financial_desc += f"成立日期: {company_info['成立日期']}\n"
            if '上市日期' in company_info:
                financial_desc += f"上市日期: {company_info['上市日期']}\n"
            if '公司简介' in company_info:
                financial_desc += f"公司简介: {company_info['公司简介'][:100]}...\n"
        
        # 公司所属指数
        if company_indices:
            financial_desc += "\n公司所属指数：\n"
            for idx, index_info in enumerate(company_indices[:3]):  # 只显示前3个
                if '指数名称' in index_info:
                    financial_desc += f"- {index_info['指数名称']}\n"
        
        # 近一年季度利润数据
        if quarterly_profit:
            financial_desc += "\n近一年季度利润数据：\n"
            for quarter in quarterly_profit[:4]:  # 只显示最近4个季度
                if '截止日期' in quarter:
                    financial_desc += f"{quarter['截止日期']}:\n"
                    if '营业收入' in quarter:
                        try:
                            income = float(quarter['营业收入']) / 10000  # 转换为亿元
                            financial_desc += f"  营业收入: {income:.2f}亿元\n"
                        except (ValueError, TypeError):
                            financial_desc += f"  营业收入: {quarter['营业收入']}万元\n"
                    if '净利润' in quarter:
                        try:
                            profit = float(quarter['净利润']) / 10000  # 转换为亿元
                            financial_desc += f"  净利润: {profit:.2f}亿元\n"
                        except (ValueError, TypeError):
                            financial_desc += f"  净利润: {quarter['净利润']}万元\n"
                    if '基本每股收益' in quarter:
                        financial_desc += f"  基本每股收益: {quarter['基本每股收益']}元/股\n"
        
        # 资产负债表数据
        if balance_sheet:
            financial_desc += "\n资产负债表数据（最近一期）：\n"
            latest_balance = balance_sheet[0]  # 获取最近一期数据
            if '截止日期' in latest_balance:
                financial_desc += f"截止日期: {latest_balance['截止日期']}\n"
            if '资产总计' in latest_balance:
                try:
                    total_assets = float(latest_balance['资产总计']) / 10000  # 转换为亿元
                    financial_desc += f"资产总计: {total_assets:.2f}亿元\n"
                except (ValueError, TypeError):
                    financial_desc += f"资产总计: {latest_balance['资产总计']}万元\n"
            if '负债合计' in latest_balance:
                try:
                    total_liabilities = float(latest_balance['负债合计']) / 10000  # 转换为亿元
                    financial_desc += f"负债合计: {total_liabilities:.2f}亿元\n"
                except (ValueError, TypeError):
                    financial_desc += f"负债合计: {latest_balance['负债合计']}万元\n"
            if '所有者权益合计' in latest_balance:
                try:
                    equity = float(latest_balance['所有者权益合计']) / 10000  # 转换为亿元
                    financial_desc += f"所有者权益合计: {equity:.2f}亿元\n"
                except (ValueError, TypeError):
                    financial_desc += f"所有者权益合计: {latest_balance['所有者权益合计']}万元\n"
        
        # 公司主要指标
        if company_metrics:
            financial_desc += "\n公司主要指标（最近一期）：\n"
            latest_metrics = company_metrics[0]  # 获取最近一期数据
            if '基本每股收益' in latest_metrics:
                financial_desc += f"基本每股收益: {latest_metrics['基本每股收益']}元/股\n"
            if '每股净资产' in latest_metrics:
                financial_desc += f"每股净资产: {latest_metrics['每股净资产']}元/股\n"
            if '净资产收益率' in latest_metrics:
                financial_desc += f"净资产收益率: {latest_metrics['净资产收益率']}%\n"
            if '销售毛利率' in latest_metrics:
                financial_desc += f"销售毛利率: {latest_metrics['销售毛利率']}%\n"
            if '资产负债比率' in latest_metrics:
                financial_desc += f"资产负债比率: {latest_metrics['资产负债比率']}%\n"
            if '主营收入同比增长' in latest_metrics:
                financial_desc += f"主营收入同比增长: {latest_metrics['主营收入同比增长']}%\n"
            if '净利润同比增长' in latest_metrics:
                financial_desc += f"净利润同比增长: {latest_metrics['净利润同比增长']}%\n"
        
        # 行业对比数据
        if company_info and '概念及板块' in company_info:
            # 尝试从概念及板块中提取行业信息
            industry_info = company_info['概念及板块']
            # 简单提取第一个行业作为所属行业
            industry_name = industry_info.split(' ')[0] if industry_info else '未知行业'
            
            # 获取行业财务数据
            industry_financial_data = data_agent.get_industry_financial_data(industry_name)
            if industry_financial_data:
                financial_desc += "\n行业对比数据：\n"
                financial_desc += f"所属行业: {industry_name}\n"
                financial_desc += f"行业平均PE: {industry_financial_data.get('平均PE', 'N/A')}\n"
                financial_desc += f"行业平均PB: {industry_financial_data.get('平均PB', 'N/A')}\n"
                financial_desc += f"行业平均PS: {industry_financial_data.get('平均PS', 'N/A')}\n"
                financial_desc += f"行业平均ROE: {industry_financial_data.get('平均ROE', 'N/A')}%\n"
                financial_desc += f"行业平均毛利率: {industry_financial_data.get('平均毛利率', 'N/A')}%\n"
                financial_desc += f"行业平均净利率: {industry_financial_data.get('平均净利率', 'N/A')}%\n"
        
        # 前瞻性指标
        # 业绩预告
        forecast_data = data_agent.get_company_forecast(symbol)
        if forecast_data:
            financial_desc += "\n业绩预告：\n"
            financial_desc += f"业绩预告类型: {forecast_data.get('业绩预告类型', 'N/A')}\n"
            financial_desc += f"预计净利润: {forecast_data.get('预计净利润', 'N/A')}\n"
            financial_desc += f"同比增长: {forecast_data.get('同比增长', 'N/A')}\n"
            financial_desc += f"发布日期: {forecast_data.get('发布日期', 'N/A')}\n"
        
        # 机构一致预期
        estimates_data = data_agent.get_institution_estimates(symbol)
        if estimates_data:
            financial_desc += "\n机构一致预期：\n"
            financial_desc += f"机构数量: {estimates_data.get('机构数量', 'N/A')}\n"
            financial_desc += f"一致预期 EPS: {estimates_data.get('一致预期 EPS', 'N/A')}\n"
            financial_desc += f"一致预期 PE: {estimates_data.get('一致预期 PE', 'N/A')}\n"
            financial_desc += f"最近3个月调整方向: {estimates_data.get('最近3个月调整方向', 'N/A')}\n"
            financial_desc += f"目标价: {estimates_data.get('目标价', 'N/A')}\n"
        
        # 股东户数变化
        shareholder_data = data_agent.get_shareholder_number(symbol)
        if shareholder_data:
            financial_desc += "\n股东户数变化：\n"
            financial_desc += f"最新股东户数: {shareholder_data.get('最新股东户数', 'N/A')}\n"
            financial_desc += f"上期股东户数: {shareholder_data.get('上期股东户数', 'N/A')}\n"
            financial_desc += f"变化率: {shareholder_data.get('变化率', 'N/A')}\n"
            financial_desc += f"数据日期: {shareholder_data.get('数据日期', 'N/A')}\n"
        
        # 实时交易数据
        if real_time_transaction:
            financial_desc += "\n实时交易数据：\n"
            if 'zdf' in real_time_transaction:
                try:
                    financial_desc += f"涨跌幅: {float(real_time_transaction['zdf']):.2f}%\n"
                except (ValueError, TypeError):
                    financial_desc += f"涨跌幅: {real_time_transaction['zdf']}%\n"
            if 'cje' in real_time_transaction:
                try:
                    financial_desc += f"成交额: {float(real_time_transaction['cje']):.2f}万元\n"
                except (ValueError, TypeError):
                    financial_desc += f"成交额: {real_time_transaction['cje']}万元\n"
            if 'cjl' in real_time_transaction:
                try:
                    financial_desc += f"成交量: {float(real_time_transaction['cjl']):.0f}手\n"
                except (ValueError, TypeError):
                    financial_desc += f"成交量: {real_time_transaction['cjl']}手\n"
        
        # 详细财务指标数据
        if financial_indicators:
            financial_desc += "\n详细财务指标数据（近四个季度）：\n"
            for i, indicator in enumerate(financial_indicators):
                financial_desc += f"\n第{i+1}季度（{indicator.get('date', '未知日期')}）：\n"
                # 只添加关键财务指标
                key_indicators = [
                    ('tbmg', '摊薄每股收益(元)'),
                    ('kfmg', '扣除非经常性损益后的每股收益(元)'),
                    ('mgjz', '每股净资产_调整前(元)'),
                    ('mgjy', '每股经营性现金流(元)'),
                    ('xsml', '销售毛利率(%)'),
                    ('xsjl', '销售净利率(%)'),
                    ('jzsy', '净资产收益率(%)'),
                    ('zcfzl', '资产负债率(%)'),
                    ('ldbl', '流动比率'),
                    ('sdbl', '速动比率'),
                    ('zysr', '主营业务收入增长率(%)'),
                    ('jlzz', '净利润增长率(%)')
                ]
                for key, name in key_indicators:
                    value = indicator.get(key, '')
                    if value:
                        financial_desc += f"{name}: {value}\n"
        
        # PE和PB估值数据
        if pe_pb_data:
            financial_desc += "\n估值指标：\n"
            financial_desc += f"PE-TTM: {pe_pb_data.get('pe_ttm', '无法计算')}\n"
            financial_desc += f"PB: {pe_pb_data.get('pb', '无法计算')}\n"
            financial_desc += f"估值状态: {pe_pb_data.get('valuation_status', '无法判断')}\n"
            financial_desc += f"估值说明: {pe_pb_data.get('valuation_detail', '无')}\n"
        else:
            financial_desc += "\n估值指标：无\n"
        
        # 历史价格数据
        if not stock_history.empty and '收盘' in stock_history.columns:
            financial_desc += "\n历史价格：\n"
            try:
                start_price = stock_history['收盘'].iloc[0]
                end_price = stock_history['收盘'].iloc[-1]
                
                if start_price != 0 and not pd.isna(start_price):
                    price_change = ((end_price - start_price) / start_price * 100)
                    financial_desc += f"近一年涨跌幅: {price_change:.2f}%\n"
                else:
                    financial_desc += "近一年涨跌幅: 数据不可用\n"
            except (IndexError, ValueError) as e:
                logger.warning(f"计算历史价格涨跌幅失败: {e}")
                financial_desc += "近一年涨跌幅: 数据不可用\n"
        
        prompt = f"""作为一名专业的股票基本面分析师，请根据以下财务数据和市场信息，对该公司进行深入分析：

{financial_desc}

请按照以下格式输出分析结果：
1. 估值分析：基于基本每股收益、每股净资产、净资产收益率等指标分析公司估值水平，并与行业平均进行对比
2. 成长性分析：分析公司近一年季度营收和利润的增长趋势，结合同比增长率判断成长性
3. 财务健康度：分析公司的资产负债结构、盈利能力、运营效率等财务健康指标，包括ROE趋势分析（连续三年变化）、负债结构（流动/非流动负债比例）、盈利现金保障倍数
4. 市场表现：结合历史价格走势、实时交易数据分析市场表现
5. 综合评价：综合以上分析，给出对该公司的总体评价和投资建议，并与行业平均水平进行对比

请使用专业、客观的语言，基于提供的数据进行分析，避免主观臆断。
"""
        
        return prompt
    
    def _parse_analysis(self, analysis_text: str) -> Dict:
        """解析分析结果"""
        result = {
            "valuation": "",
            "growth": "",
            "financial_health": "",
            "summary": "",
            "financial_risk": ""
        }
        
        # 简单解析，实际应用中可能需要更复杂的解析
        sections = analysis_text.split("\n\n")
        
        for i, section in enumerate(sections):
            if "估值分析" in section:
                result["valuation"] = section.replace("1. 估值分析：", "").strip()
            elif "成长性分析" in section:
                result["growth"] = section.replace("2. 成长性分析：", "").strip()
            elif "财务健康度" in section:
                result["financial_health"] = section.replace("3. 财务健康度：", "").strip()
            elif "综合评价" in section:
                result["summary"] = section.replace("4. 综合评价：", "").strip()
            elif "财务风险" in section:
                result["financial_risk"] = section.replace("财务风险：", "").strip()
        
        return result
    
    def _detect_financial_fraud_risks(self, balance_sheet: List[Dict], company_metrics: List[Dict], industry_financial_data: Dict) -> List[str]:
        """检测财务造假风险
        
        Args:
            balance_sheet: 资产负债表数据
            company_metrics: 公司主要指标
            industry_financial_data: 行业财务数据
            
        Returns:
            List[str]: 财务风险信号列表
        """
        risk_signals = []
        
        try:
            if not balance_sheet or not company_metrics:
                return risk_signals
            
            # 获取最近一期数据
            latest_balance = balance_sheet[0]
            latest_metrics = company_metrics[0]
            
            # 1. 存贷双高检测
            # 假设货币资金和有息负债数据存在
            if '货币资金' in latest_balance and '有息负债' in latest_balance:
                try:
                    cash = float(latest_balance['货币资金'])
                    debt = float(latest_balance['有息负债'])
                    total_assets = float(latest_balance.get('资产总计', 1))
                    
                    # 货币资金和有息负债同时占总资产的比例较高
                    if cash / total_assets > 0.2 and debt / total_assets > 0.2:
                        risk_signals.append('存贷双高')
                except (ValueError, TypeError):
                    pass
            
            # 2. 毛利率异常高于同行
            if '销售毛利率' in latest_metrics and industry_financial_data:
                try:
                    company_gross_margin = float(latest_metrics['销售毛利率'])
                    industry_gross_margin = float(industry_financial_data.get('平均毛利率', 0))
                    
                    # 公司毛利率高于行业平均50%以上
                    if company_gross_margin > industry_gross_margin * 1.5:
                        risk_signals.append('毛利率异常高于同行')
                except (ValueError, TypeError):
                    pass
            
            # 3. 其他应收款占比过大
            if '其他应收款' in latest_balance:
                try:
                    other_receivables = float(latest_balance['其他应收款'])
                    total_assets = float(latest_balance.get('资产总计', 1))
                    
                    # 其他应收款占总资产比例超过10%
                    if other_receivables / total_assets > 0.1:
                        risk_signals.append('其他应收款占比过大')
                except (ValueError, TypeError):
                    pass
            
        except Exception as e:
            logger.error(f"检测财务造假风险失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
        
        return risk_signals
    
    async def get_valuation_score(self, symbol: str) -> float:
        """获取公司估值评分（0-10）"""
        try:
            # 1. 获取公司基本信息和财务数据
            company_info = data_agent.get_company_info(symbol)
            company_metrics = data_agent.get_company_metrics(symbol)
            real_time_transaction = data_agent.get_real_time_transaction(symbol)
            
            if not company_metrics or not real_time_transaction:
                logger.warning(f"获取估值评分数据失败: {symbol}")
                return 5.0
            
            # 2. 获取最近一期财务指标
            latest_metrics = company_metrics[0]
            
            # 3. 计算相对估值分位得分（0-10）
            relative_score = self._calculate_relative_valuation_score(latest_metrics)
            
            # 4. 计算绝对估值折价得分（0-10）
            absolute_score = self._calculate_absolute_valuation_score(symbol, latest_metrics, real_time_transaction)
            
            # 5. 综合估值分 = 相对分位得分×0.6 + 绝对折价得分×0.4
            composite_score = relative_score * 0.6 + absolute_score * 0.4
            
            # 确保得分在0-10范围内
            composite_score = max(0, min(10, composite_score))
            
            logger.info(f"股票 {symbol} 的动态估值评分: {composite_score:.2f}")
            
            return composite_score
            
        except Exception as e:
            logger.error(f"获取估值评分失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return 5.0
    
    async def get_market_sentiment_analysis(self) -> Dict:
        """获取市场情绪分析
        
        综合分析市场情绪指标，包括：
        1. 活筹指数
        2. 涨跌家数比
        3. 北向资金流向
        4. 海外主要指数
        
        Returns:
            Dict: 市场情绪分析结果
        """
        try:
            # 获取市场情绪指标
            sentiment_indicators = data_agent.get_market_sentiment_indicators()
            
            # 检查是否所有指标都失败
            all_failed = True
            if sentiment_indicators.get("active_market", {}).get("active_index", 0) != 0:
                all_failed = False
            if sentiment_indicators.get("adr", {}).get("ad_ratio", 0) != 0:
                all_failed = False
            if sentiment_indicators.get("north_flow", {}).get("latest_flow", 0) != 0:
                all_failed = False
            
            if all_failed:
                logger.warning("所有市场情绪指标获取失败")
                return {
                    "overall_sentiment": "未知",
                    "analysis": "获取市场情绪数据失败",
                    "indicators": {},
                    "timestamp": self._get_current_time()
                }
            
            # 分析市场情绪
            analysis = self._analyze_market_sentiment(sentiment_indicators)
            
            # 获取海外主要指数数据
            overseas_indices = await self._get_overseas_indices()
            if overseas_indices:
                analysis['overseas_indices'] = overseas_indices
            
            # 缓存结果
            cache_key = self._get_cache_key("get_market_sentiment_analysis")
            self._cache_result(cache_key, analysis)
            
            logger.info("市场情绪分析完成")
            return analysis
            
        except Exception as e:
            logger.error(f"获取市场情绪分析失败: {e}")
            return {
                "overall_sentiment": "未知",
                "analysis": "获取市场情绪数据失败",
                "indicators": {},
                "timestamp": self._get_current_time()
            }
    
    async def _get_overseas_indices(self) -> Dict:
        """获取海外主要指数数据
        
        Returns:
            Dict: 海外指数数据
        """
        try:
            # 获取海外主要指数
            index_symbols = ['hkHSI', 'usSPX', 'usDJI', 'usIXIC']
            
            # 使用异步方式获取指数数据
            loop = asyncio.get_running_loop()
            index_data = await loop.run_in_executor(
                None,  # 使用默认线程池
                data_agent.get_index_quotes,
                index_symbols
            )
            
            # 整理指数数据
            overseas_indices = {}
            for symbol, df in index_data.items():
                if not df.empty and '最新价' in df.columns and '涨跌幅' in df.columns:
                    try:
                        latest_price = float(df.iloc[0]['最新价'])
                        change_percent = float(df.iloc[0]['涨跌幅'])
                        index_name = df.iloc[0]['名称'] if '名称' in df.columns else symbol
                        
                        overseas_indices[symbol] = {
                            'name': index_name,
                            'price': latest_price,
                            'change_percent': change_percent
                        }
                    except (ValueError, IndexError) as e:
                        logger.warning(f"解析指数 {symbol} 数据失败: {e}")
            
            return overseas_indices
            
        except Exception as e:
            logger.error(f"获取海外指数数据失败: {e}")
            return {}
    
    def _analyze_market_sentiment(self, indicators: Dict) -> Dict:
        """分析市场情绪
        
        Args:
            indicators: 市场情绪指标数据
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 提取各指标数据
            active_market = indicators.get("active_market", {"active_index": 0})
            adr = indicators.get("adr", {"ad_ratio": 0, "breadth": 0, "sentiment": "情绪中性"})
            north_flow = indicators.get("north_flow", {"latest_flow": 0})
            
            # 分析活筹指数
            active_index = active_market.get("active_index", 0)
            active_analysis = "活跃资金稳定" if abs(active_index) < 100000 else ("活跃资金流入" if active_index > 0 else "活跃资金流出")
            
            # 分析涨跌家数比
            ad_ratio = adr.get("ad_ratio", 0)
            breadth = adr.get("breadth", 0)
            adr_sentiment = adr.get("sentiment", "情绪中性")
            
            # 分析北向资金
            latest_flow = north_flow.get("latest_flow", 0)
            north_analysis = "北向资金净流入" if latest_flow > 0 else ("北向资金净流出" if latest_flow < 0 else "北向资金平衡")
            
            # 综合判断市场情绪
            sentiment_score = 0
            
            # 活筹指数得分
            if active_index > 50000:
                sentiment_score += 3
            elif active_index > 0:
                sentiment_score += 1
            elif active_index < -50000:
                sentiment_score -= 3
            elif active_index < 0:
                sentiment_score -= 1
            
            # 涨跌家数比得分
            if ad_ratio > 2:
                sentiment_score += 3
            elif ad_ratio > 1.5:
                sentiment_score += 2
            elif ad_ratio > 1:
                sentiment_score += 1
            elif ad_ratio < 0.5:
                sentiment_score -= 3
            elif ad_ratio < 0.8:
                sentiment_score -= 1
            
            # 北向资金得分
            if latest_flow > 50:
                sentiment_score += 3
            elif latest_flow > 10:
                sentiment_score += 1
            elif latest_flow < -50:
                sentiment_score -= 3
            elif latest_flow < -10:
                sentiment_score -= 1
            
            # 确定整体市场情绪
            if sentiment_score >= 5:
                overall_sentiment = "极度乐观"
            elif sentiment_score >= 3:
                overall_sentiment = "乐观"
            elif sentiment_score >= 0:
                overall_sentiment = "中性"
            elif sentiment_score >= -3:
                overall_sentiment = "谨慎"
            else:
                overall_sentiment = "悲观"
            
            # 构建分析结果
            analysis = {
                "overall_sentiment": overall_sentiment,
                "analysis": f"市场情绪整体呈现{overall_sentiment}状态。{active_analysis}，{adr_sentiment}，{north_analysis}。",
                "indicators": {
                    "active_market": {
                        "value": active_index,
                        "analysis": active_analysis
                    },
                    "adr": {
                        "value": ad_ratio,
                        "breadth": breadth,
                        "sentiment": adr_sentiment
                    },
                    "north_flow": {
                        "value": latest_flow,
                        "analysis": north_analysis
                    }
                },
                "timestamp": indicators.get("timestamp", "")
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析市场情绪失败: {e}")
            return {
                "overall_sentiment": "未知",
                "analysis": "分析市场情绪失败",
                "indicators": {}
            }
    
    def _calculate_relative_valuation_score(self, metrics: Dict) -> float:
        """计算相对估值分位得分（0-10）
        
        基于PE、PB、PS 历史5年分位，赋予0-10分（分位越低分越高）
        """
        try:
            # 这里使用模拟数据，实际应用中需要获取历史5年分位数据
            # 模拟PE、PB、PS的历史分位（0-100）
            pe_percentile = 30  # 假设PE在历史30%分位
            pb_percentile = 25  # 假设PB在历史25%分位
            ps_percentile = 35  # 假设PS在历史35%分位
            
            # 计算平均分位
            avg_percentile = (pe_percentile + pb_percentile + ps_percentile) / 3
            
            # 转换为0-10分（分位越低分越高）
            score = 10 - (avg_percentile / 10)
            
            return max(0, min(10, score))
            
        except Exception as e:
            logger.error(f"计算相对估值分位得分失败: {e}")
            return 5.0
    
    def _calculate_absolute_valuation_score(self, symbol: str, metrics: Dict, real_time_transaction: Dict) -> float:
        """计算绝对估值折价得分（0-10）
        
        使用DCF（贴现现金流）简化模型，计算内在价值与当前股价的折溢价率，转换为分数
        """
        try:
            # 获取当前股价
            current_price = float(real_time_transaction.get('jz', 0))
            if current_price <= 0:
                return 5.0
            
            # 简化的DCF模型计算内在价值
            # 假设自由现金流增长率为5%，贴现率为10%
            # 这里使用模拟数据，实际应用中需要获取真实的自由现金流数据
            free_cash_flow = 1.0  # 假设自由现金流为1元/股
            growth_rate = 0.05  # 5%的增长率
            discount_rate = 0.10  # 10%的贴现率
            
            # 计算内在价值
            intrinsic_value = free_cash_flow * (1 + growth_rate) / (discount_rate - growth_rate)
            
            # 计算折溢价率
            premium_rate = (current_price - intrinsic_value) / intrinsic_value * 100
            
            # 转换为0-10分（折价越多分数越高，溢价越多分数越低）
            if premium_rate < -20:  # 折价20%以上
                score = 10
            elif premium_rate < -10:  # 折价10-20%
                score = 8
            elif premium_rate < 0:  # 折价0-10%
                score = 6
            elif premium_rate < 10:  # 溢价0-10%
                score = 4
            elif premium_rate < 20:  # 溢价10-20%
                score = 2
            else:  # 溢价20%以上
                score = 0
            
            return score
            
        except Exception as e:
            logger.error(f"计算绝对估值折价得分失败: {e}")
            return 5.0
    
    def _get_current_time(self) -> str:
        """获取当前时间字符串
        
        Returns:
            str: 当前时间字符串，格式为"YYYY-MM-DD HH:MM:SS"
        """
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# 全局基本面分析Agent实例
fundamental_agent = FundamentalAnalysisAgent()
