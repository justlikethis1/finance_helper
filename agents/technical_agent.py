import pandas as pd
import numpy as np
import asyncio
import time
import os
from typing import Dict, List, Optional
import logging

from agents.data_agent import data_agent
from backend.model import model_manager

# 导入自定义日志配置
from backend.logging_config import get_logger

# 获取日志记录器
logger = get_logger("agents.technical")

class TechnicalAnalysisAgent:
    def __init__(self):
        self.cache = {}
        self.cache_expiry = 1800  # 缓存30分钟
        
        # 技术指标权重配置
        self.indicator_weights = {
            'macd': 0.3,
            'ma': 0.3,
            'boll': 0.2,
            'kdj': 0.2
        }
        
        # 评分配置 - 按照新的权重
        self.score_config = {
            'trend': 0.4,      # 趋势得分(40%)
            'momentum': 0.4,    # 动量得分(40%)
            'volatility': 0.2,  # 波动率得分(20%)
            'sentiment': 0.0    # 不再使用情绪得分
        }
    
    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        """生成缓存键"""
        return f"{func_name}:{args}:{kwargs}"
    
    def _get_cached_result(self, key: str, current_price: float = None) -> Optional[Dict]:
        """获取缓存结果
        
        Args:
            key: 缓存键
            current_price: 当前价格，用于判断价格波动是否超过阈值
            
        Returns:
            Optional[Dict]: 缓存结果
        """
        if key in self.cache:
            cached_time, result = self.cache[key]
            if time.time() - cached_time < self.cache_expiry:
                # 检查价格波动
                if current_price is not None and 'price' in result:
                    price_diff = abs(current_price - result['price']) / result['price'] * 100
                    if price_diff < 0.5:
                        logger.info(f"从缓存获取结果: {key} (价格波动<0.5%)")
                        return result
                else:
                    logger.info(f"从缓存获取结果: {key}")
                    return result
        return None
    
    def _cache_result(self, key: str, result: Dict, current_price: float = None) -> None:
        """缓存结果
        
        Args:
            key: 缓存键
            result: 结果
            current_price: 当前价格，用于后续判断价格波动
        """
        if current_price is not None:
            result['price'] = current_price
        self.cache[key] = (time.time(), result)
    
    async def analyze_stock(self, symbol: str, period: str = "daily") -> Dict:
        """分析股票技术面"""
        try:
            # 获取当前价格
            current_price = None
            stock_data = data_agent.get_stock_history(symbol, days=1)
            if not stock_data.empty:
                current_price = stock_data.iloc[-1]['收盘']
            
            cache_key = self._get_cache_key("analyze_stock", symbol, period)
            cached_result = self._get_cached_result(cache_key, current_price)
            if cached_result:
                return cached_result
            
            # 转换周期参数，将daily转换为d
            period_map = {
                "daily": "d",
                "weekly": "w",
                "monthly": "m",
                "yearly": "y"
            }
            biying_period = period_map.get(period, period)
            
            # 1. 获取技术指标数据
            macd_data = data_agent.get_macd_data(symbol, period=biying_period, limit=50)
            ma_data = data_agent.get_ma_data(symbol, period=biying_period, limit=50)
            boll_data = data_agent.get_boll_data(symbol, period=biying_period, limit=50)
            kdj_data = data_agent.get_kdj_data(symbol, period=biying_period, limit=50)
            rsi_data = data_agent.get_rsi_data(symbol, period=biying_period, limit=50)
            atr_data = data_agent.get_atr_data(symbol, period=biying_period, limit=50)
            obv_data = data_agent.get_obv_data(symbol, period=biying_period, limit=50)
            vwap_data = data_agent.get_vwap_data(symbol, period=biying_period, limit=50)
            
            # 2. 整合技术指标，生成技术分析信号
            technical_signals = self._integrate_technical_signals(macd_data, ma_data, boll_data, kdj_data, rsi_data, atr_data, obv_data, vwap_data)
            
            # 3. 构建技术分析提示词
            prompt = self._build_technical_prompt(symbol, macd_data, ma_data, boll_data, kdj_data, rsi_data, atr_data, obv_data, vwap_data, period, technical_signals)
            
            # 3. 调用模型生成技术分析
            analysis = await model_manager.async_generate_response(prompt)
            
            # 4. 解析分析结果
            result = self._parse_analysis(analysis)
            
            # 5. 计算技术面评分
            result['symbol'] = symbol  # 添加股票代码，用于机器学习预测
            score = self._calculate_technical_score(result, technical_signals, ma_data, macd_data, boll_data)
            result["score"] = score
            
            # 6. 缓存结果
            self._cache_result(cache_key, result, current_price)
            
            return result
            
        except Exception as e:
            logger.error(f"技术分析失败: {e}")
            return {
                "trend_analysis": "",
                "momentum_analysis": "",
                "support_resistance": "",
                "short_term_prediction": "",
                "score": 5.0
            }
    
    def _integrate_technical_signals(self, macd_data: pd.DataFrame, ma_data: pd.DataFrame, boll_data: pd.DataFrame, kdj_data: pd.DataFrame, rsi_data: pd.DataFrame = None, atr_data: pd.DataFrame = None, obv_data: pd.DataFrame = None, vwap_data: pd.DataFrame = None) -> Dict:
        """整合技术指标，生成技术分析信号"""
        signals = {
            'trend': 'neutral',  # bullish, bearish, neutral
            'momentum': 'neutral',  # strong, weak, neutral
            'volatility': 'neutral',  # high, low, neutral
            'sentiment': 'neutral',  # positive, negative, neutral
            'buy_signals': [],
            'sell_signals': [],
            'warning_signals': []
        }
        
        # 1. 分析MA指标
        if not ma_data.empty:
            latest_ma = ma_data.iloc[-1]
            # 检查均线排列
            ma_values = []
            for ma in ['ma5', 'ma10', 'ma20', 'ma30', 'ma60']:
                if ma in latest_ma and not pd.isna(latest_ma[ma]):
                    ma_values.append(latest_ma[ma])
            
            if len(ma_values) >= 3:
                # 多头排列
                if all(ma_values[i] > ma_values[i+1] for i in range(len(ma_values)-1)):
                    signals['trend'] = 'bullish'
                    signals['buy_signals'].append('多头排列')
                # 空头排列
                elif all(ma_values[i] < ma_values[i+1] for i in range(len(ma_values)-1)):
                    signals['trend'] = 'bearish'
                    signals['sell_signals'].append('空头排列')
        
        # 2. 分析MACD指标
        if not macd_data.empty:
            latest_macd = macd_data.iloc[-1]
            
            if 'diff' in latest_macd and 'dea' in latest_macd and not pd.isna(latest_macd['diff']) and not pd.isna(latest_macd['dea']):
                # MACD金叉
                if latest_macd['diff'] > latest_macd['dea']:
                    signals['momentum'] = 'strong'
                    signals['buy_signals'].append('MACD金叉')
                # MACD死叉
                elif latest_macd['diff'] < latest_macd['dea']:
                    signals['momentum'] = 'weak'
                    signals['sell_signals'].append('MACD死叉')
            
            if 'macd' in latest_macd and not pd.isna(latest_macd['macd']):
                # MACD柱状图分析
                if latest_macd['macd'] > 0:
                    signals['momentum'] = 'strong'
                else:
                    signals['momentum'] = 'weak'
        
        # 3. 分析KDJ指标
        if not kdj_data.empty:
            latest_kdj = kdj_data.iloc[-1]
            
            if 'k' in latest_kdj and 'd' in latest_kdj and 'j' in latest_kdj:
                k, d, j = latest_kdj['k'], latest_kdj['d'], latest_kdj['j']
                
                if not pd.isna(k) and not pd.isna(d) and not pd.isna(j):
                    # KDJ超买超卖
                    if j > 80:
                        signals['sentiment'] = 'negative'
                        signals['sell_signals'].append('KDJ超买')
                    elif j < 20:
                        signals['sentiment'] = 'positive'
                        signals['buy_signals'].append('KDJ超卖')
                    
                    # KDJ金叉死叉
                    if k > d:
                        signals['buy_signals'].append('KDJ金叉')
                    else:
                        signals['sell_signals'].append('KDJ死叉')
        
        # 4. 分析BOLL指标
        if not boll_data.empty:
            latest_boll = boll_data.iloc[-1]
            
            if 'u' in latest_boll and 'm' in latest_boll and 'd' in latest_boll:
                u, m, d = latest_boll['u'], latest_boll['m'], latest_boll['d']
                
                if not pd.isna(u) and not pd.isna(m) and not pd.isna(d):
                    # 价格位置分析（假设使用MA5作为价格参考）
                    if not ma_data.empty and 'ma5' in ma_data.iloc[-1]:
                        price = ma_data.iloc[-1]['ma5']
                        if price > u:
                            signals['warning_signals'].append('价格突破上轨')
                        elif price < d:
                            signals['warning_signals'].append('价格突破下轨')
                    
                    # 布林带宽度分析
                    boll_width = u - d
                    if len(boll_data) >= 10:
                        avg_width = (boll_data['u'] - boll_data['d']).mean()
                        if boll_width > avg_width * 1.5:
                            signals['volatility'] = 'high'
                            signals['warning_signals'].append('布林带开口扩大')
                        elif boll_width < avg_width * 0.5:
                            signals['volatility'] = 'low'
                            signals['warning_signals'].append('布林带开口缩小')
        
        # 5. 分析RSI指标
        if rsi_data is not None and not rsi_data.empty:
            if len(rsi_data) >= 2:
                latest_rsi = rsi_data.iloc[-1]['rsi']
                previous_rsi = rsi_data.iloc[-2]['rsi']
                
                # RSI超买超卖
                if latest_rsi > 70:
                    signals['sentiment'] = 'negative'
                    signals['sell_signals'].append('RSI超买')
                elif latest_rsi < 30:
                    signals['sentiment'] = 'positive'
                    signals['buy_signals'].append('RSI超卖')
                
                # RSI背离识别
                if not ma_data.empty and 'ma5' in ma_data.columns:
                    if len(ma_data) >= 2:
                        latest_price = ma_data.iloc[-1]['ma5']
                        previous_price = ma_data.iloc[-2]['ma5']
                        
                        # 价格新高但RSI未新高（顶背离）
                        if latest_price > previous_price and latest_rsi < previous_rsi:
                            signals['warning_signals'].append('RSI顶背离')
                            signals['sell_signals'].append('RSI顶背离')
                        # 价格新低但RSI未新低（底背离）
                        elif latest_price < previous_price and latest_rsi > previous_rsi:
                            signals['buy_signals'].append('RSI底背离')
        
        # 6. 分析ATR指标
        if atr_data is not None and not atr_data.empty:
            latest_atr = atr_data.iloc[-1]['atr']
            if len(atr_data) >= 20:
                avg_atr = atr_data['atr'].tail(20).mean()
                if latest_atr > avg_atr * 1.5:
                    signals['volatility'] = 'high'
                    signals['warning_signals'].append('ATR大幅增加')
                elif latest_atr < avg_atr * 0.5:
                    signals['volatility'] = 'low'
                    signals['warning_signals'].append('ATR大幅减少')
        
        # 7. 分析OBV指标
        if obv_data is not None and not obv_data.empty:
            if len(obv_data) >= 2:
                latest_obv = obv_data.iloc[-1]['obv']
                previous_obv = obv_data.iloc[-2]['obv']
                
                # OBV与价格的关系
                if not ma_data.empty and 'ma5' in ma_data.columns:
                    if len(ma_data) >= 2:
                        latest_price = ma_data.iloc[-1]['ma5']
                        previous_price = ma_data.iloc[-2]['ma5']
                        
                        # 价格上涨，OBV也上涨，确认趋势
                        if latest_price > previous_price and latest_obv > previous_obv:
                            signals['buy_signals'].append('OBV与价格同步上涨')
                        # 价格下跌，OBV也下跌，确认趋势
                        elif latest_price < previous_price and latest_obv < previous_obv:
                            signals['sell_signals'].append('OBV与价格同步下跌')
                        # 价格上涨但OBV下跌，可能反转
                        elif latest_price > previous_price and latest_obv < previous_obv:
                            signals['warning_signals'].append('OBV与价格背离')
                        # 价格下跌但OBV上涨，可能反转
                        elif latest_price < previous_price and latest_obv > previous_obv:
                            signals['warning_signals'].append('OBV与价格背离')
        
        # 8. 分析VWAP指标
        if vwap_data is not None and not vwap_data.empty:
            latest_vwap = vwap_data.iloc[-1]['vwap']
            if not ma_data.empty and 'ma5' in ma_data.columns:
                latest_price = ma_data.iloc[-1]['ma5']
                if latest_price > latest_vwap:
                    signals['buy_signals'].append('价格在VWAP上方')
                else:
                    signals['sell_signals'].append('价格在VWAP下方')
        
        # 9. 风险信号量化
        self._quantify_risk_signals(signals, ma_data, rsi_data, macd_data, obv_data)
        
        return signals
    
    def _quantify_risk_signals(self, signals: Dict, ma_data: pd.DataFrame, rsi_data: pd.DataFrame, macd_data: pd.DataFrame, obv_data: pd.DataFrame):
        """量化风险信号
        
        Args:
            signals: 技术信号字典
            ma_data: 均线数据
            rsi_data: RSI数据
            macd_data: MACD数据
            obv_data: OBV数据
        """
        # 1. 顶/底背离量化
        if not ma_data.empty and 'ma5' in ma_data.columns and rsi_data is not None and not rsi_data.empty:
            if len(ma_data) >= 3 and len(rsi_data) >= 3:
                # 价格与RSI背离程度
                price_change = ma_data['ma5'].iloc[-1] - ma_data['ma5'].iloc[-3]
                rsi_change = rsi_data['rsi'].iloc[-1] - rsi_data['rsi'].iloc[-3]
                
                # 顶背离：价格新高，RSI未新高
                if price_change > 0 and rsi_change < 0:
                    背离程度 = abs(rsi_change) / (abs(price_change) / ma_data['ma5'].iloc[-3] * 100)
                    if 背离程度 > 2:
                        signals['warning_signals'].append('RSI顶背离(强烈)')
                    elif 背离程度 > 1:
                        signals['warning_signals'].append('RSI顶背离(中等)')
                    else:
                        signals['warning_signals'].append('RSI顶背离(轻微)')
                # 底背离：价格新低，RSI未新低
                elif price_change < 0 and rsi_change > 0:
                    背离程度 = abs(rsi_change) / (abs(price_change) / ma_data['ma5'].iloc[-3] * 100)
                    if 背离程度 > 2:
                        signals['buy_signals'].append('RSI底背离(强烈)')
                    elif 背离程度 > 1:
                        signals['buy_signals'].append('RSI底背离(中等)')
                    else:
                        signals['buy_signals'].append('RSI底背离(轻微)')
        
        # 2. 成交量异常
        if not ma_data.empty and 'ma5' in ma_data.columns and obv_data is not None and not obv_data.empty:
            if len(ma_data) >= 2 and len(obv_data) >= 2:
                # 价格微涨
                price_change = (ma_data['ma5'].iloc[-1] - ma_data['ma5'].iloc[-2]) / ma_data['ma5'].iloc[-2] * 100
                # OBV变化（作为成交量变化的代理）
                obv_change = (obv_data['obv'].iloc[-1] - obv_data['obv'].iloc[-2]) / obv_data['obv'].iloc[-2] * 100
                
                # 放量滞涨：价格微涨（<1%）但成交量放大2倍以上
                if 0 < price_change < 1 and obv_change > 100:
                    signals['warning_signals'].append('放量滞涨')
        
        # 3. 支撑/阻力强度
        if not ma_data.empty and 'ma5' in ma_data.columns:
            # 基于历史高低点计算支撑/阻力强度
            if len(ma_data) >= 20:
                # 计算最近20天的高低点
                recent_highs = ma_data['ma5'].tail(20).rolling(window=5).max()
                recent_lows = ma_data['ma5'].tail(20).rolling(window=5).min()
                
                # 找到关键价位
                key_levels = []
                for i in range(5, len(recent_highs)):
                    if i+1 < len(recent_highs):
                        if recent_highs.iloc[i] > recent_highs.iloc[i-1] and recent_highs.iloc[i] > recent_highs.iloc[i+1]:
                            key_levels.append(('resistance', recent_highs.iloc[i]))
                        if recent_lows.iloc[i] < recent_lows.iloc[i-1] and recent_lows.iloc[i] < recent_lows.iloc[i+1]:
                            key_levels.append(('support', recent_lows.iloc[i]))
                    else:
                        if recent_highs.iloc[i] > recent_highs.iloc[i-1]:
                            key_levels.append(('resistance', recent_highs.iloc[i]))
                        if recent_lows.iloc[i] < recent_lows.iloc[i-1]:
                            key_levels.append(('support', recent_lows.iloc[i]))
                
                # 计算触及次数
                current_price = ma_data['ma5'].iloc[-1]
                for level_type, level in key_levels:
                    # 计算触及次数
                    触及次数 = 0
                    for price in ma_data['ma5'].tail(20):
                        if abs(price - level) / level < 0.01:  # 1%以内视为触及
                            触及次数 += 1
                    
                    # 基于触及次数判断强度
                    if 触及次数 >= 5:
                        if level_type == 'support':
                            signals['buy_signals'].append('强支撑位')
                        else:
                            signals['sell_signals'].append('强阻力位')
                    elif 触及次数 >= 3:
                        if level_type == 'support':
                            signals['buy_signals'].append('中等支撑位')
                        else:
                            signals['sell_signals'].append('中等阻力位')

    def _build_technical_prompt(self, symbol: str, macd_data: pd.DataFrame, ma_data: pd.DataFrame, boll_data: pd.DataFrame, kdj_data: pd.DataFrame, rsi_data: pd.DataFrame = None, atr_data: pd.DataFrame = None, obv_data: pd.DataFrame = None, vwap_data: pd.DataFrame = None, period: str = "daily", technical_signals: Dict = None) -> str:
        """构建技术分析提示词"""
        # 检查是否所有技术数据都为空
        if macd_data.empty and ma_data.empty and boll_data.empty and kdj_data.empty and (rsi_data is None or rsi_data.empty) and (atr_data is None or atr_data.empty) and (obv_data is None or obv_data.empty) and (vwap_data is None or vwap_data.empty):
            return f"股票 {symbol} 的技术数据不可用。"
        
        # 构建技术指标描述
        tech_desc = f"股票代码: {symbol}\n周期: {period}\n\n"
        
        # 添加技术信号汇总（表格形式）
        if technical_signals:
            tech_desc += "## 技术信号汇总\n"
            tech_desc += "| 信号类型 | 状态 |\n"
            tech_desc += "|---------|------|\n"
            tech_desc += f"| 趋势判断 | {technical_signals['trend']} |\n"
            tech_desc += f"| 动量状态 | {technical_signals['momentum']} |\n"
            tech_desc += f"| 波动性 | {technical_signals['volatility']} |\n"
            tech_desc += f"| 市场情绪 | {technical_signals['sentiment']} |\n"
            
            if technical_signals['buy_signals']:
                tech_desc += f"| 买入信号 | {', '.join(technical_signals['buy_signals'])} |\n"
            if technical_signals['sell_signals']:
                tech_desc += f"| 卖出信号 | {', '.join(technical_signals['sell_signals'])} |\n"
            if technical_signals['warning_signals']:
                tech_desc += f"| 警告信号 | {', '.join(technical_signals['warning_signals'])} |\n"
            tech_desc += "\n"
        
        # 技术指标表格
        tech_desc += "## 技术指标\n"
        tech_desc += "| 指标 | 数值 |\n"
        tech_desc += "|------|------|\n"
        
        # 移动平均线
        if not ma_data.empty:
            latest_ma = ma_data.iloc[-1]
            if 'ma5' in latest_ma and not pd.isna(latest_ma['ma5']):
                tech_desc += f"| MA5 | {latest_ma['ma5']:.2f} |\n"
            if 'ma10' in latest_ma and not pd.isna(latest_ma['ma10']):
                tech_desc += f"| MA10 | {latest_ma['ma10']:.2f} |\n"
            if 'ma20' in latest_ma and not pd.isna(latest_ma['ma20']):
                tech_desc += f"| MA20 | {latest_ma['ma20']:.2f} |\n"
            if 'ma30' in latest_ma and not pd.isna(latest_ma['ma30']):
                tech_desc += f"| MA30 | {latest_ma['ma30']:.2f} |\n"
            if 'ma60' in latest_ma and not pd.isna(latest_ma['ma60']):
                tech_desc += f"| MA60 | {latest_ma['ma60']:.2f} |\n"
        
        # MACD指标
        if not macd_data.empty:
            latest_macd = macd_data.iloc[-1]
            if 'diff' in latest_macd and not pd.isna(latest_macd['diff']):
                tech_desc += f"| MACD_DIF | {latest_macd['diff']:.4f} |\n"
            if 'dea' in latest_macd and not pd.isna(latest_macd['dea']):
                tech_desc += f"| MACD_DEA | {latest_macd['dea']:.4f} |\n"
            if 'macd' in latest_macd and not pd.isna(latest_macd['macd']):
                tech_desc += f"| MACD_HIST | {latest_macd['macd']:.4f} |\n"
        
        # BOLL指标
        if not boll_data.empty:
            latest_boll = boll_data.iloc[-1]
            if 'u' in latest_boll and not pd.isna(latest_boll['u']):
                tech_desc += f"| BOLL_UPPER | {latest_boll['u']:.2f} |\n"
            if 'm' in latest_boll and not pd.isna(latest_boll['m']):
                tech_desc += f"| BOLL_MID | {latest_boll['m']:.2f} |\n"
            if 'd' in latest_boll and not pd.isna(latest_boll['d']):
                tech_desc += f"| BOLL_LOWER | {latest_boll['d']:.2f} |\n"
        
        # KDJ指标
        if not kdj_data.empty:
            latest_kdj = kdj_data.iloc[-1]
            if 'k' in latest_kdj and not pd.isna(latest_kdj['k']):
                tech_desc += f"| KDJ_K | {latest_kdj['k']:.2f} |\n"
            if 'd' in latest_kdj and not pd.isna(latest_kdj['d']):
                tech_desc += f"| KDJ_D | {latest_kdj['d']:.2f} |\n"
            if 'j' in latest_kdj and not pd.isna(latest_kdj['j']):
                tech_desc += f"| KDJ_J | {latest_kdj['j']:.2f} |\n"
        
        # RSI指标
        if rsi_data is not None and not rsi_data.empty:
            latest_rsi = rsi_data.iloc[-1]['rsi']
            tech_desc += f"| RSI | {latest_rsi:.2f} |\n"
        
        # ATR指标
        if atr_data is not None and not atr_data.empty:
            latest_atr = atr_data.iloc[-1]['atr']
            tech_desc += f"| ATR | {latest_atr:.2f} |\n"
        
        # OBV指标
        if obv_data is not None and not obv_data.empty:
            latest_obv = obv_data.iloc[-1]['obv']
            tech_desc += f"| OBV | {latest_obv:.2f} |\n"
        
        # VWAP指标
        if vwap_data is not None and not vwap_data.empty:
            latest_vwap = vwap_data.iloc[-1]['vwap']
            tech_desc += f"| VWAP | {latest_vwap:.2f} |\n"
        
        # 构建提示词
        prompt = f"""作为一名专业的股票技术分析师，请根据以下技术指标和价格数据，对该股票进行技术分析：

{tech_desc}

请按照以下格式输出分析结果：
1. 趋势分析：分析股票的长期、中期和短期趋势，包括均线排列、趋势线等
2. 动量分析：基于MACD、RSI等指标，分析股票的动量状态
3. 支撑阻力：识别关键的支撑位和阻力位
4. 短期预测：基于技术面分析，给出未来3-5个交易日的短期走势预测

请使用专业、客观的语言，基于提供的数据进行分析，避免主观臆断。
"""
        
        return prompt
    
    def _parse_analysis(self, analysis_text: str) -> Dict:
        """解析技术分析结果"""
        result = {
            "trend_analysis": "",
            "momentum_analysis": "",
            "support_resistance": "",
            "short_term_prediction": ""
        }
        
        # 简单解析，实际应用中可能需要更复杂的解析
        sections = analysis_text.split("\n\n")
        
        for i, section in enumerate(sections):
            if "趋势分析" in section:
                result["trend_analysis"] = section.replace("1. 趋势分析：", "").strip()
            elif "动量分析" in section:
                result["momentum_analysis"] = section.replace("2. 动量分析：", "").strip()
            elif "支撑阻力" in section:
                result["support_resistance"] = section.replace("3. 支撑阻力：", "").strip()
            elif "短期预测" in section:
                result["short_term_prediction"] = section.replace("4. 短期预测：", "").strip()
        
        return result
    
    def _calculate_technical_score(self, analysis_result: Dict, technical_signals: Dict = None, ma_data: pd.DataFrame = None, macd_data: pd.DataFrame = None, boll_data: pd.DataFrame = None) -> float:
        """计算技术面评分（0-10分）"""
        try:
            # 获取市场状态，动态调整权重
            market_state = self._get_market_state()
            dynamic_weights = self._get_dynamic_weights(market_state)
            
            # 1. 趋势得分计算 (0-10分)
            trend_score = 5.0  # 默认中性
            if ma_data is not None and not ma_data.empty:
                latest_ma = ma_data.iloc[-1]
                # 检查是否有MA5, MA10, MA20, MA30
                if all(ma in latest_ma for ma in ['ma5', 'ma10', 'ma20', 'ma30']) and \
                   not any(pd.isna(latest_ma[ma]) for ma in ['ma5', 'ma10', 'ma20', 'ma30']):
                    # 检查多头排列
                    if latest_ma['ma5'] > latest_ma['ma10'] > latest_ma['ma20'] > latest_ma['ma30']:
                        # 检查均线斜率
                        if len(ma_data) >= 5:
                            ma5_slope = (latest_ma['ma5'] - ma_data.iloc[-5]['ma5']) / ma_data.iloc[-5]['ma5'] * 100
                            ma10_slope = (latest_ma['ma10'] - ma_data.iloc[-5]['ma10']) / ma_data.iloc[-5]['ma10'] * 100
                            ma20_slope = (latest_ma['ma20'] - ma_data.iloc[-10]['ma20']) / ma_data.iloc[-10]['ma20'] * 100
                            ma30_slope = (latest_ma['ma30'] - ma_data.iloc[-15]['ma30']) / ma_data.iloc[-15]['ma30'] * 100
                            
                            if all(slope > 0.1 for slope in [ma5_slope, ma10_slope, ma20_slope, ma30_slope]):
                                trend_score = 10.0
                            else:
                                # 部分多头排列，线性插值
                                slope_sum = ma5_slope + ma10_slope + ma20_slope + ma30_slope
                                trend_score = min(10.0, max(0.0, 5.0 + slope_sum * 0.5))
                        else:
                            trend_score = 8.0  # 没有足够数据，给予较高分
                    # 检查空头排列
                    elif latest_ma['ma5'] < latest_ma['ma10'] < latest_ma['ma20'] < latest_ma['ma30']:
                        # 检查均线斜率
                        if len(ma_data) >= 5:
                            ma5_slope = (latest_ma['ma5'] - ma_data.iloc[-5]['ma5']) / ma_data.iloc[-5]['ma5'] * 100
                            ma10_slope = (latest_ma['ma10'] - ma_data.iloc[-5]['ma10']) / ma_data.iloc[-5]['ma10'] * 100
                            ma20_slope = (latest_ma['ma20'] - ma_data.iloc[-10]['ma20']) / ma_data.iloc[-10]['ma20'] * 100
                            ma30_slope = (latest_ma['ma30'] - ma_data.iloc[-15]['ma30']) / ma_data.iloc[-15]['ma30'] * 100
                            
                            if all(slope < -0.1 for slope in [ma5_slope, ma10_slope, ma20_slope, ma30_slope]):
                                trend_score = 0.0
                            else:
                                # 部分空头排列，线性插值
                                slope_sum = ma5_slope + ma10_slope + ma20_slope + ma30_slope
                                trend_score = min(10.0, max(0.0, 5.0 + slope_sum * 0.5))
                        else:
                            trend_score = 2.0  # 没有足够数据，给予较低分
                    else:
                        # 其他情况，根据均线排列情况线性插值
                        ma_values = [latest_ma['ma5'], latest_ma['ma10'], latest_ma['ma20'], latest_ma['ma30']]
                        increasing_count = sum(1 for i in range(len(ma_values)-1) if ma_values[i] > ma_values[i+1])
                        trend_score = (increasing_count / 3) * 10.0  # 0-3个递增，对应0-10分
            
            # 2. 动量得分计算 (0-10分)
            momentum_score = 5.0  # 默认中性
            if macd_data is not None and not macd_data.empty:
                latest_macd = macd_data.iloc[-1]
                if all(col in latest_macd for col in ['diff', 'dea', 'macd']) and \
                   not any(pd.isna(latest_macd[col]) for col in ['diff', 'dea', 'macd']):
                    # MACD金叉且柱状线翻红
                    if latest_macd['diff'] > latest_macd['dea'] and latest_macd['macd'] > 0:
                        momentum_score = 10.0
                    # MACD死叉且柱状线翻绿
                    elif latest_macd['diff'] < latest_macd['dea'] and latest_macd['macd'] < 0:
                        momentum_score = 0.0
                    else:
                        # 其他情况，按DIFF与DEA距离归一化
                        diff_dea_diff = abs(latest_macd['diff'] - latest_macd['dea'])
                        # 计算最近20天的平均差异作为归一化基准
                        if len(macd_data) >= 20:
                            avg_diff = macd_data[['diff', 'dea']].apply(lambda x: abs(x['diff'] - x['dea']), axis=1).mean()
                            if avg_diff > 0:
                                normalized_diff = min(1.0, diff_dea_diff / avg_diff)
                                if latest_macd['diff'] > latest_macd['dea']:
                                    # 金叉趋势，得分从5-10
                                    momentum_score = 5.0 + normalized_diff * 5.0
                                else:
                                    # 死叉趋势，得分从0-5
                                    momentum_score = 5.0 - normalized_diff * 5.0
            
            # 3. 波动率得分计算 (0-10分)
            volatility_score = 5.0  # 默认中性
            if boll_data is not None and not boll_data.empty:
                # 基于布林带宽度计算波动率
                if len(boll_data) >= 20:
                    # 计算最近20天的布林带宽度
                    boll_data['width'] = boll_data['u'] - boll_data['d']
                    current_width = boll_data.iloc[-1]['width']
                    avg_width = boll_data['width'].tail(20).mean()
                    
                    if avg_width > 0:
                        # 波动率低（稳定）给高分，高波动给低分
                        volatility_ratio = current_width / avg_width
                        # 波动率比率越低，得分越高
                        volatility_score = max(0.0, min(10.0, 10.0 - (volatility_ratio - 0.5) * 10.0))
            
            # 计算加权总分
            total_score = (trend_score * dynamic_weights['trend'] +
                          momentum_score * dynamic_weights['momentum'] +
                          volatility_score * dynamic_weights['volatility'])
            
            # 引入机器学习辅助信号
            if 'symbol' in analysis_result:
                symbol = analysis_result['symbol']
                up_probability = ml_model.predict(symbol)
                # 上涨概率>60%则评分+0.5
                if up_probability > 0.6:
                    total_score += 0.5
                # 上涨概率<40%则评分-0.5
                elif up_probability < 0.4:
                    total_score -= 0.5
            
            # 确保分数在0-10之间
            final_score = max(0.0, min(10.0, total_score))
            return final_score
            
        except Exception as e:
            logger.error(f"计算技术评分失败: {e}")
            return 5.0
    
    def _get_market_state(self) -> str:
        """获取当前市场状态
        
        Returns:
            str: 市场状态，可选值：'trend_up'（趋势上升）、'trend_down'（趋势下降）、'range'（震荡）
        """
        try:
            # 获取上证指数数据
            index_data = data_agent.get_index_quotes(['sh000001'])
            
            if not index_data.empty:
                # 获取最近的指数数据
                latest_data = index_data.iloc[-1]
                
                # 计算20日均线和斜率
                if len(index_data) >= 20:
                    index_data['ma20'] = index_data['最新价'].rolling(window=20).mean()
                    latest_ma20 = index_data['ma20'].iloc[-1]
                    previous_ma20 = index_data['ma20'].iloc[-2]
                    ma20_slope = (latest_ma20 - previous_ma20) / previous_ma20 * 100
                    
                    # 判断市场状态
                    if latest_data['最新价'] > latest_ma20 and ma20_slope > 0:
                        return 'trend_up'  # 趋势上升
                    elif latest_data['最新价'] < latest_ma20 and ma20_slope < 0:
                        return 'trend_down'  # 趋势下降
                    else:
                        return 'range'  # 震荡
        except Exception as e:
            logger.error(f"获取市场状态失败: {e}")
        
        # 默认返回震荡
        return 'range'
    
    def _get_dynamic_weights(self, market_state: str) -> Dict:
        """根据市场状态获取动态权重
        
        Args:
            market_state: 市场状态
            
        Returns:
            Dict: 动态权重配置
        """
        weights = {
            'trend': 0.4,
            'momentum': 0.4,
            'volatility': 0.2
        }
        
        if market_state == 'trend_up':
            # 趋势市：提升趋势权重，降低动量权重
            weights = {
                'trend': 0.5,
                'momentum': 0.3,
                'volatility': 0.2
            }
        elif market_state == 'range':
            # 震荡市：提升波动率权重，调整动量权重
            weights = {
                'trend': 0.3,
                'momentum': 0.4,
                'volatility': 0.3
            }
        elif market_state == 'trend_down':
            # 下跌市：降低趋势权重，提升波动率权重
            weights = {
                'trend': 0.2,
                'momentum': 0.3,
                'volatility': 0.5
            }
        
        return weights
    
    async def get_technical_summary(self, symbol: str) -> str:
        """获取技术分析摘要"""
        analysis = await self.analyze_stock(symbol)
        
        summary = f"技术面评分: {analysis['score']:.1f}/10分\n"
        summary += f"趋势分析: {analysis['trend_analysis'][:100]}...\n"
        summary += f"短期预测: {analysis['short_term_prediction'][:100]}..."
        
        return summary
    
    async def analyze_multiple_periods(self, symbol: str) -> Dict:
        """多周期分析
        
        分析日线、周线、60分钟线的技术指标，并输出多周期共振信号
        
        Args:
            symbol: 股票代码
            
        Returns:
            Dict: 多周期分析结果
        """
        try:
            cache_key = self._get_cache_key("analyze_multiple_periods", symbol)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # 分析不同周期
            periods = {
                "daily": "d",      # 日线
                "weekly": "w",     # 周线
                "60min": "60"      # 60分钟线
            }
            
            period_analyses = {}
            for period_name, period_code in periods.items():
                # 获取技术指标数据
                macd_data = data_agent.get_macd_data(symbol, period=period_code, limit=50)
                ma_data = data_agent.get_ma_data(symbol, period=period_code, limit=50)
                boll_data = data_agent.get_boll_data(symbol, period=period_code, limit=50)
                kdj_data = data_agent.get_kdj_data(symbol, period=period_code, limit=50)
                rsi_data = data_agent.get_rsi_data(symbol, period=period_code, limit=50)
                atr_data = data_agent.get_atr_data(symbol, period=period_code, limit=50)
                obv_data = data_agent.get_obv_data(symbol, period=period_code, limit=50)
                vwap_data = data_agent.get_vwap_data(symbol, period=period_code, limit=50)
                
                # 整合技术信号
                technical_signals = self._integrate_technical_signals(
                    macd_data, ma_data, boll_data, kdj_data, 
                    rsi_data, atr_data, obv_data, vwap_data
                )
                
                # 计算技术评分
                score = self._calculate_technical_score(
                    {}, technical_signals, ma_data, macd_data, boll_data
                )
                
                period_analyses[period_name] = {
                    "signals": technical_signals,
                    "score": score
                }
            
            # 生成多周期共振信号
            resonance_signals = self._generate_resonance_signals(period_analyses)
            
            result = {
                "period_analyses": period_analyses,
                "resonance_signals": resonance_signals
            }
            
            # 缓存结果
            self._cache_result(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"多周期分析失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                "period_analyses": {},
                "resonance_signals": []
            }
    
    def _generate_resonance_signals(self, period_analyses: Dict) -> List[str]:
        """生成多周期共振信号
        
        Args:
            period_analyses: 不同周期的分析结果
            
        Returns:
            List[str]: 共振信号列表
        """
        signals = []
        
        # 检查是否所有周期都看涨
        if all(
            analysis["signals"]["trend"] == "bullish" 
            for analysis in period_analyses.values()
        ):
            signals.append("多周期共振看涨")
        
        # 检查是否所有周期都看跌
        if all(
            analysis["signals"]["trend"] == "bearish" 
            for analysis in period_analyses.values()
        ):
            signals.append("多周期共振看跌")
        
        # 检查日线和周线是否都看涨
        if "daily" in period_analyses and "weekly" in period_analyses:
            if period_analyses["daily"]["signals"]["trend"] == "bullish" and \
               period_analyses["weekly"]["signals"]["trend"] == "bullish":
                signals.append("日周线共振看涨")
            elif period_analyses["daily"]["signals"]["trend"] == "bearish" and \
                 period_analyses["weekly"]["signals"]["trend"] == "bearish":
                signals.append("日周线共振看跌")
        
        # 检查短期和长期周期的背离
        if "60min" in period_analyses and "weekly" in period_analyses:
            if period_analyses["60min"]["signals"]["trend"] == "bullish" and \
               period_analyses["weekly"]["signals"]["trend"] == "bearish":
                signals.append("短期与长期趋势背离")
            elif period_analyses["60min"]["signals"]["trend"] == "bearish" and \
                 period_analyses["weekly"]["signals"]["trend"] == "bullish":
                signals.append("短期与长期趋势背离")
        
        # 检查动量共振
        if all(
            analysis["signals"]["momentum"] == "strong" 
            for analysis in period_analyses.values()
        ):
            signals.append("多周期动量共振走强")
        
        if all(
            analysis["signals"]["momentum"] == "weak" 
            for analysis in period_analyses.values()
        ):
            signals.append("多周期动量共振走弱")
        
        return signals

class MachineLearningModel:
    """机器学习模型类
    
    使用XGBoost训练二分类模型，预测次日涨跌幅
    """
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_path = os.path.join(os.path.dirname(__file__), "ml_model.json")
        self.scaler_path = os.path.join(os.path.dirname(__file__), "scaler.json")
        self._load_model()
    
    def _load_model(self):
        """加载模型和缩放器"""
        try:
            if os.path.exists(self.model_path):
                import joblib
                self.model = joblib.load(self.model_path)
                logger.info("成功加载机器学习模型")
            
            if os.path.exists(self.scaler_path):
                import joblib
                self.scaler = joblib.load(self.scaler_path)
                logger.info("成功加载特征缩放器")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
    
    def train_model(self, symbol: str, days: int = 365):
        """训练模型
        
        Args:
            symbol: 股票代码
            days: 训练数据天数
        """
        try:
            # 获取历史数据
            stock_data = data_agent.get_stock_history(symbol, days=days)
            
            if stock_data.empty or len(stock_data) < 30:
                logger.error("训练数据不足")
                return False
            
            # 计算技术指标
            stock_data['price_change'] = stock_data['收盘'].pct_change() * 100
            stock_data['rsi'] = self._calculate_rsi(stock_data['收盘'], 14)
            stock_data['macd'], stock_data['macd_signal'], stock_data['macd_hist'] = self._calculate_macd(stock_data['收盘'])
            stock_data['boll_width'] = self._calculate_boll_width(stock_data['收盘'])
            stock_data['volume_change'] = stock_data['成交量'].pct_change() * 100
            
            # 标记标签：次日涨为1，跌为0
            stock_data['label'] = (stock_data['price_change'].shift(-1) > 0).astype(int)
            
            # 移除NaN值
            stock_data = stock_data.dropna()
            
            if len(stock_data) < 20:
                logger.error("有效训练数据不足")
                return False
            
            # 提取特征和标签
            features = stock_data[['rsi', 'macd', 'macd_signal', 'macd_hist', 'boll_width', 'volume_change']]
            labels = stock_data['label']
            
            # 特征缩放
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
            
            # 训练模型
            from xgboost import XGBClassifier
            self.model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
            self.model.fit(features_scaled, labels)
            
            # 保存模型和缩放器
            import joblib
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            logger.info(f"成功训练模型，准确率: {self.model.score(features_scaled, labels):.2f}")
            return True
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return False
    
    def predict(self, symbol: str) -> float:
        """预测次日上涨概率
        
        Args:
            symbol: 股票代码
            
        Returns:
            float: 上涨概率
        """
        try:
            if self.model is None or self.scaler is None:
                # 如果模型不存在，尝试训练
                if not self.train_model(symbol):
                    return 0.5  # 默认中性
            
            # 获取最新数据
            stock_data = data_agent.get_stock_history(symbol, days=30)
            
            if stock_data.empty or len(stock_data) < 20:
                return 0.5
            
            # 计算技术指标
            stock_data['rsi'] = self._calculate_rsi(stock_data['收盘'], 14)
            stock_data['macd'], stock_data['macd_signal'], stock_data['macd_hist'] = self._calculate_macd(stock_data['收盘'])
            stock_data['boll_width'] = self._calculate_boll_width(stock_data['收盘'])
            stock_data['volume_change'] = stock_data['成交量'].pct_change() * 100
            
            # 获取最新特征
            latest_data = stock_data.iloc[-1]
            features = [[latest_data['rsi'], latest_data['macd'], latest_data['macd_signal'], 
                        latest_data['macd_hist'], latest_data['boll_width'], latest_data['volume_change']]]
            
            # 特征缩放
            features_scaled = self.scaler.transform(features)
            
            # 预测概率
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return probability
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return 0.5
    
    def _calculate_rsi(self, prices, period=14):
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """计算MACD"""
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist
    
    def _calculate_boll_width(self, prices, period=20):
        """计算布林带宽度"""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        width = (upper - lower) / ma
        return width

# 全局机器学习模型实例
ml_model = MachineLearningModel()

# 全局技术分析Agent实例
technical_agent = TechnicalAnalysisAgent()
