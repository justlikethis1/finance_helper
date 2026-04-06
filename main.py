import sys
import os
# 将当前目录添加到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
import uuid
import time
from typing import List, Dict, Optional
import logging

# 导入自定义日志配置
from backend.logging_config import get_logger

# 获取日志记录器
logger = get_logger("app.main")

# 导入自定义模块
from database.database import db_manager
from agents.market_analysis_agent import market_analysis_agent
from agents.summary_agent import summary_agent
from agents.data_agent import data_agent
from agents.intent_recognizer import intent_recognizer, IntentType
from agents.information_retriever import information_retriever
from agents.output_processor import output_processor
from agents.nlp_agent import nlp_agent
from backend.model import initialize_model, model_manager


async def _generate_stock_analysis(stock_code: str, stock_name: str, user_message: str, request_report: bool):
    """生成股票分析报告或快速分析
    
    Args:
        stock_code: 股票代码
        stock_name: 股票名称
        user_message: 用户输入消息
        request_report: 是否需要生成完整报告
    
    Returns:
        Tuple[bool, str, str]: (是否生成了报告, 分析结果, 报告ID)
    """
    try:
        if request_report:
            # 用户明确要求生成报告
            report_result = await summary_agent.generate_investment_report(stock_code, user_message)
            ai_response = f"已为您生成{stock_name}({stock_code})的投资报告！\n\n报告摘要：\n{report_result['summary']}\n\n"
            ai_response += f"综合评级：{report_result['rating']}\n目标价格：{report_result['target_price']}\n风险等级：{report_result['risk_level']}\n"
            report_id = report_result.get("report_id", "")
            return True, ai_response, report_id
        else:
            # 用户只是要求分析，生成快速分析
            quick_analysis = await summary_agent.generate_quick_analysis(stock_code, user_message)
            ai_response = f"已为您分析{stock_name}({stock_code})的投资情况：\n\n{quick_analysis}\n\n"
            ai_response += "如果需要更详细的投资报告，请输入'生成投资报告'或'下载报告'。"
            return False, ai_response, ""
    except Exception as e:
        logger.error(f"生成股票分析失败: {e}", exc_info=True)
        ai_response = f"生成{stock_name}({stock_code})的分析失败，请稍后重试。"
        return False, ai_response, ""

# 使用lifespan事件处理程序代替on_event
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("应用正在启动...")
    
    # 初始化并验证模型加载
    print("正在加载模型...")
    try:
        # 使用返回值检查模型是否加载成功
        is_model_loaded = initialize_model()
        
        # 验证模型加载状态
        from backend.model import model_manager
        if is_model_loaded:
            print("✅ 模型加载成功！")
            
            # 进行简单的测试，确保模型能正常工作
            test_prompt = "你好"
            try:
                # 使用最小参数进行快速验证
                test_response = model_manager.generate_response(
                    test_prompt, 
                    max_new_tokens=10,  # 减少生成长度
                    temperature=0.1     # 使用低温度加速生成
                )
                if test_response and "模型未加载" not in test_response:
                    print("✅ 模型验证通过，可以正常生成响应")
                else:
                    print("❌ 模型验证失败，无法正常生成响应")
            except Exception as e:
                print(f"❌ 模型验证过程中发生错误: {e}")
                import traceback
                print(f"  详细错误信息: {traceback.format_exc()}")
        else:
            print("❌ 模型加载失败！")
            if not is_model_loaded:
                print("  原因：模型加载函数返回失败")
    except Exception as e:
        print(f"❌ 模型初始化过程中发生错误: {e}")
        import traceback
        print(f"  详细错误信息: {traceback.format_exc()}")
    
    print("请在浏览器中访问: http://localhost:8000/")
    yield
    # 清理代码可以放在这里（如果需要）

# 初始化FastAPI应用
app = FastAPI(
    title="AI智能股票投资助手",
    description="基于大语言模型的智能股票投资推荐系统",
    version="1.0.0",
    lifespan=lifespan
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """记录所有HTTP请求的中间件"""
    start_time = time.time()
    
    # 记录请求信息
    client_ip = request.client.host
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"请求开始: {request.method} {request.url}，客户端IP: {client_ip}，User-Agent: {user_agent}")
    
    try:
        # 处理请求
        response = await call_next(request)
        
        # 记录响应信息
        process_time = time.time() - start_time
        logger.info(f"请求完成: {request.method} {request.url}，状态码: {response.status_code}，处理时间: {process_time:.2f} 秒")
        
        return response
    except Exception as e:
        # 记录错误信息
        process_time = time.time() - start_time
        logger.error(f"请求出错: {request.method} {request.url}，错误: {e}，处理时间: {process_time:.2f} 秒")
        raise

# 请求模型
class ChatRequest(BaseModel):
    session_id: str
    message: str

class NewSessionRequest(BaseModel):
    title: Optional[str] = "新会话"

# API路由
@app.get("/api/market/overview")
async def get_market_overview():
    """获取市场概览数据"""
    logger.info("接收到市场概览请求")
    
    try:
        # 直接调用market_analysis_agent的方法获取真实数据
        market_data = await market_analysis_agent.get_market_overview()
        return JSONResponse(content=market_data)
    except Exception as e:
        logger.error(f"获取市场概览数据失败: {e}")
        # 返回一个包含错误信息的响应，而不是模拟数据
        return JSONResponse(
            content={
                "error": True,
                "message": "获取市场数据失败，请稍后重试。",
                "details": str(e)[:100]  # 限制错误信息长度
            },
            status_code=503  # 使用服务不可用状态码
        )
    


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """处理聊天请求"""
    try:
        session_id = request.session_id
        user_message = request.message
        
        # 生成消息ID
        user_message_id = str(uuid.uuid4())
        
        # 保存用户消息
        db_manager.add_message(
            message_id=user_message_id,
            session_id=session_id,
            role="user",
            content=user_message
        )
        
        # 获取对话历史（最近7天）
        message_history = db_manager.get_session_messages(session_id, days_limit=7)
        
        # 构建对话历史字符串
        history_str = ""
        for msg in message_history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_str += f"{role}: {msg['content']}\n"
        
        # 使用增强NLP分析用户需求
        logger.info(f"开始分析用户输入: {user_message}")
        intent, entities = nlp_agent.process_input(user_message, session_id)
        # 增强意图识别结果的日志记录，添加类型检查
        logger.info(f"意图识别结果: 意图={intent.value if hasattr(intent, 'value') else str(intent)}, 实体={entities}")
        
        # 检查用户是否要求生成报告
        request_report = False
        report_keywords = ['生成报告', '下载报告', '出具报告', '提供报告', '发送报告']
        for keyword in report_keywords:
            if keyword in user_message:
                request_report = True
                break
        logger.info(f"用户是否要求生成报告: {request_report}")
        
        # 初始化变量
        ai_response = ""
        report_id = ""
        report_generated = False
        
        # 验证意图类型是否有效
        if not hasattr(intent, 'value'):
            logger.error(f"意图类型无效: {intent}")
            ai_response = "系统无法识别您的请求意图，请稍后重试。"
        else:
            requirements = intent_recognizer.analyze_requirements(intent, entities, user_message)
            logger.info(f"需求分析结果: {requirements}")
            
            try:
                # 股票分析特殊处理 - 直接使用summary_agent生成详细报告
                if intent == IntentType.STOCK_ANALYSIS:
                    logger.info("开始处理股票分析意图")
                    stock_code = entities.get("stock_code")
                    stock_name = entities.get("stock_name")
                    
                    logger.info(f"股票分析参数: 代码={stock_code}, 名称={stock_name}")
                    
                    # 如果没有股票代码但有股票名称，尝试动态获取代码
                    if not stock_code and stock_name:
                        try:
                            logger.info(f"尝试根据股票名称 '{stock_name}' 动态获取代码...")
                            stock_code = data_agent.get_stock_code_by_name(stock_name)
                            if stock_code:
                                entities["stock_code"] = stock_code
                                logger.info(f"成功获取到股票 '{stock_name}' 的代码: {stock_code}")
                        except Exception as e:
                            logger.error(f"动态获取股票代码失败: {e}", exc_info=True)
                    
                    if stock_code:
                        try:
                            if request_report:
                                # 用户明确要求生成报告
                                logger.info(f"开始为股票代码 {stock_code} 生成投资报告")
                                report_result = await summary_agent.generate_investment_report(stock_code, user_message)
                                logger.info(f"投资报告生成成功")
                                ai_response = f"已为您生成{report_result.get('stock_name', stock_code)}的投资报告！\n\n报告摘要：\n{report_result['summary']}\n\n"
                                ai_response += f"综合评级：{report_result['rating']}\n目标价格：{report_result['target_price']}\n风险等级：{report_result['risk_level']}\n"
                                ai_response += f"您可以下载完整报告：[报告下载链接](http://localhost:8000/api/reports/{report_result.get('report_id', '')})\n"
                                report_id = report_result.get("report_id", "")
                                report_generated = True
                            else:
                                # 用户只是要求分析，生成快速分析
                                logger.info(f"开始为股票代码 {stock_code} 生成快速投资分析")
                                quick_analysis = await summary_agent.generate_quick_analysis(stock_code, user_message)
                                logger.info(f"快速投资分析生成成功")
                                ai_response = f"已为您分析{stock_code}的投资情况：\n\n{quick_analysis}\n\n"
                                ai_response += "如果需要更详细的投资报告，请输入'生成投资报告'或'下载报告'。"
                                report_generated = False
                        except Exception as e:
                            logger.error(f"处理股票分析请求失败: {e}", exc_info=True)
                    
                    if not report_generated and not (entities.get('stock_code') and entities.get('stock_name')):
                        # 只有在NLP未能识别股票信息的情况下，才尝试模糊匹配
                        import difflib
                        import re
                        
                        logger.info("NLP未能识别股票信息，尝试使用模糊匹配从用户输入中查找股票名称")
                        
                        # 获取所有股票名称列表
                        stock_names = list(data_agent.stock_name_code_map.keys())
                        
                        # 提取用户输入中的所有中文词语作为候选
                        # 1. 首先尝试分割整个输入为单个中文字符串
                        chinese_words = re.findall(r'[\u4e00-\u9fa5]+', user_message)
                        
                        # 2. 然后尝试提取可能的股票名称（包含常见后缀的）
                        stock_name_candidates = re.findall(r'[\u4e00-\u9fa5]{2,8}(?:股份|集团|控股|科技|银行|证券|保险|医药|能源|材料|化工|电子|机械|汽车|地产|建筑|交通|通信|农业|食品|饮料|纺织|服装|轻工|家电|零售|服务|文化|旅游|传媒|教育|体育|娱乐|环保|节能|新能源|互联网|软件|硬件|半导体|芯片|集成电路|人工智能|大数据|云计算|区块链|5G|物联网|生物|医疗|健康|养老)', user_message)
                        
                        # 3. 合并所有候选
                        all_candidates = chinese_words + stock_name_candidates
                        # 去重
                        all_candidates = list(set(all_candidates))
                        
                        logger.info(f"提取到的股票名称候选: {all_candidates}")
                        
                        # 尝试模糊匹配找到最接近的股票名称
                        best_match = None
                        best_ratio = 0.0
                        
                        for candidate in all_candidates:
                            if len(candidate) < 2:  # 忽略太短的词语
                                continue
                            
                            # 使用difflib查找最接近的匹配
                            matches = difflib.get_close_matches(candidate, stock_names, n=1, cutoff=0.6)
                            if matches:
                                match = matches[0]
                                ratio = difflib.SequenceMatcher(None, candidate, match).ratio()
                                logger.debug(f"候选 '{candidate}' 匹配到 '{match}'，相似度: {ratio:.2f}")
                                if ratio > best_ratio:
                                    best_ratio = ratio
                                    best_match = match
                        
                        logger.info(f"最佳匹配结果: 名称='{best_match}', 相似度={best_ratio:.2f}")
                        
                        # 如果没有找到匹配，尝试直接将完整输入作为候选
                        if not best_match:
                            # 直接使用完整输入作为候选
                            full_input_match = difflib.get_close_matches(user_message, stock_names, n=1, cutoff=0.4)
                            if full_input_match:
                                best_match = full_input_match[0]
                                best_ratio = difflib.SequenceMatcher(None, user_message, best_match).ratio()
                                logger.info(f"完整输入匹配结果: 名称='{best_match}', 相似度={best_ratio:.2f}")
                        
                        # 如果找到匹配的股票名称，尝试获取代码
                        if best_match and best_ratio >= 0.6:
                            try:
                                logger.info(f"通过模糊匹配找到股票名称 '{best_match}'，相似度: {best_ratio:.2f}")
                                # 首先从静态映射表获取代码
                                if best_match in data_agent.stock_name_code_map:
                                    stock_code = data_agent.stock_name_code_map[best_match]
                                    entities["stock_name"] = best_match
                                    entities["stock_code"] = stock_code
                                    logger.info(f"从静态映射表获取到股票 '{best_match}' 的代码: {stock_code}")
                                    
                                    # 处理报告生成或快速分析
                                    report_generated, ai_response, report_id = await _generate_stock_analysis(stock_code, best_match, user_message, request_report)
                                elif not report_generated:
                                    # 如果静态映射表中没有，尝试动态获取
                                    stock_code = data_agent.get_stock_code_by_name(best_match)
                                    if stock_code:
                                        entities["stock_name"] = best_match
                                        entities["stock_code"] = stock_code
                                        logger.info(f"动态获取到股票 '{best_match}' 的代码: {stock_code}")
                                        
                                        # 处理报告生成或快速分析
                                        report_generated, ai_response, report_id = await _generate_stock_analysis(stock_code, best_match, user_message, request_report)
                            except Exception as e:
                                logger.error(f"基于模糊匹配生成报告失败: {e}", exc_info=True)
                        
                        # 如果还是没有找到，尝试直接使用用户输入中可能的股票名称部分
                        # 提取可能的股票名称（2-8个中文字符）
                        if not report_generated:
                            logger.info("尝试使用用户输入中可能的股票名称部分")
                            # 先提取包含常见股票后缀的名称
                            stock_names_with_suffix = re.findall(r'[\u4e00-\u9fa5]{2,8}(?:股份|集团|控股|科技|银行|证券|保险|医药|能源|材料|化工|电子|机械|汽车|地产|建筑|交通|通信|农业|食品|饮料|纺织|服装|轻工|家电|零售|服务|文化|旅游|传媒|教育|体育|娱乐|环保|节能|新能源|互联网|软件|硬件|半导体|芯片|集成电路|人工智能|大数据|云计算|区块链|5G|物联网|生物|医疗|健康|养老)', user_message)
                            # 再提取普通的2-4个中文字符（大多数股票名称在2-4个字符之间）
                            general_stock_names = re.findall(r'[\u4e00-\u9fa5]{2,4}', user_message)
                            
                            # 合并所有候选，并去重
                            possible_names = list(set(stock_names_with_suffix + general_stock_names))
                            logger.info(f"提取到的可能股票名称: {possible_names}")
                            
                            # 常见的非股票词汇列表
                            common_non_stock = ['帮我', '分析', '一下', '看看', '了解', '查询', '的股票', '的股', '走势', '如何', '什么', '请帮', '我看', '股票', '估值', '基本面', '技术面', '资金面', '分析一', '分析下', '请分析', '帮看看', '我想', '了解下', '帮分析', '请帮我', '看一下', '了解一', '帮我看', '请看看', '一下', '分析']
                            
                            # 优先尝试包含后缀的股票名称
                            for possible_name in stock_names_with_suffix:
                                if possible_name not in common_non_stock:
                                    try:
                                        logger.info(f"尝试使用可能的股票名称 '{possible_name}'")
                                        stock_code = data_agent.get_stock_code_by_name(possible_name)
                                        if stock_code:
                                            entities["stock_name"] = possible_name
                                            entities["stock_code"] = stock_code
                                            logger.info(f"成功获取到股票 '{possible_name}' 的代码: {stock_code}")
                                            # 生成投资报告
                                            report_generated, ai_response, report_id = await _generate_stock_analysis(stock_code, possible_name, user_message, request_report)
                                            break
                                    except Exception as e:
                                        logger.error(f"获取股票代码失败: {e}", exc_info=True)
                            
                            # 如果没有找到包含后缀的股票名称，再尝试普通的股票名称
                            if not report_generated:
                                logger.info("尝试使用普通的股票名称")
                                for possible_name in general_stock_names:
                                    if possible_name not in common_non_stock:
                                        try:
                                            logger.info(f"尝试使用可能的股票名称 '{possible_name}'")
                                            stock_code = data_agent.get_stock_code_by_name(possible_name)
                                            if stock_code:
                                                entities["stock_name"] = possible_name
                                                entities["stock_code"] = stock_code
                                                logger.info(f"成功获取到股票 '{possible_name}' 的代码: {stock_code}")
                                                # 生成投资报告
                                                report_generated, ai_response, report_id = await _generate_stock_analysis(stock_code, possible_name, user_message, request_report)
                                                break
                                        except Exception as e:
                                            logger.error(f"获取股票代码失败: {e}", exc_info=True)
                        
                        if not report_generated:
                            logger.warning("无法生成投资报告，没有找到有效的股票代码")
                            ai_response = "请提供股票代码或准确的股票名称，我可以为您生成详细的投资报告。"
                
                # 其他意图使用智能信息检索和模型生成
                else:
                    try:
                        # 1. 检索相关信息
                        logger.info(f"开始检索信息，意图: {intent.value}")
                        retrieved_info = await information_retriever.retrieve_information(intent, entities, requirements, user_message)
                        logger.info(f"信息检索完成，检索到的数据类型: {list(retrieved_info.get('data', {}).keys())}")
                        
                        # 2. 生成提示词
                        logger.info(f"开始生成提示词")
                        prompt = information_retriever.generate_prompt(intent, entities, retrieved_info, user_message)
                        logger.info(f"提示词生成完成，长度: {len(prompt)} 字符")
                        
                        # 3. 调用模型生成响应
                        logger.info(f"开始调用模型生成响应")
                        model_output = await model_manager.async_generate_response(prompt)
                        logger.info(f"模型响应生成完成，长度: {len(model_output)} 字符")
                        
                        # 4. 处理模型输出
                        logger.info(f"开始处理模型输出")
                        ai_response = output_processor.process_output(model_output, intent, entities)
                        logger.info(f"模型输出处理完成，长度: {len(ai_response)} 字符")
                    except Exception as e:
                        logger.error(f"智能信息检索和模型生成失败: {e}", exc_info=True)
                        # 根据意图返回合适的错误提示
                        if intent == IntentType.SECTOR_ANALYSIS:
                            ai_response = "获取板块分析时出错，请稍后重试。"
                        elif intent == IntentType.MARKET_ANALYSIS:
                            ai_response = "获取大盘分析时出错，请稍后重试。"
                        elif intent == IntentType.QUERY_NEWS:
                            ai_response = "查询新闻时出错，请稍后重试。"
                        elif intent == IntentType.QUERY_STOCK_LIST:
                            ai_response = "获取股票列表时出错，请稍后重试。"
                        elif intent == IntentType.QUERY_STOCK_INFO:
                            ai_response = "查询股票信息时出错，请稍后重试。"
                        elif intent == IntentType.QUERY_MARKET_INFO:
                            ai_response = "查询市场指数时出错，请稍后重试。"
                        elif intent == IntentType.QUERY_SECTOR_INFO:
                            ai_response = "查询板块信息时出错，请稍后重试。"
                        elif intent == IntentType.RECOMMEND_STOCK:
                            ai_response = "生成股票推荐时出错，请稍后重试。"
                        elif intent == IntentType.RECOMMEND_SECTOR:
                            ai_response = "生成板块推荐时出错，请稍后重试。"
                        elif intent == IntentType.RECOMMEND_PORTFOLIO:
                            ai_response = "生成投资组合推荐时出错，请稍后重试。"
                        else:
                            ai_response = "处理您的请求时出错，请稍后重试。"
            except Exception as e:
                logger.error(f"处理聊天请求时发生错误: {e}", exc_info=True)
                
                # 根据意图返回合适的错误提示
                if hasattr(intent, 'value'):
                    if intent == IntentType.STOCK_ANALYSIS:
                        ai_response = "生成股票分析报告时出错，请稍后重试。"
                    elif intent == IntentType.SECTOR_ANALYSIS:
                        ai_response = "获取板块分析时出错，请稍后重试。"
                    elif intent == IntentType.MARKET_ANALYSIS:
                        ai_response = "获取大盘分析时出错，请稍后重试。"
                    elif intent == IntentType.QUERY_NEWS:
                        ai_response = "查询新闻时出错，请稍后重试。"
                    elif intent == IntentType.QUERY_STOCK_LIST:
                        ai_response = "获取股票列表时出错，请稍后重试。"
                    else:
                        ai_response = "处理您的请求时出错，请稍后重试。"
                else:
                    ai_response = "处理您的请求时出错，请稍后重试。"
        
        # 生成AI消息ID
        ai_message_id = str(uuid.uuid4())
        
        # 保存AI消息
        db_manager.add_message(
            message_id=ai_message_id,
            session_id=session_id,
            role="assistant",
            content=ai_response
        )
        
        # 返回响应
        response_data = {
            "message": ai_response,
            "report_id": report_id
        }
        
        return JSONResponse(content=response_data)
        
    except Exception as e:
        logger.error(f"处理聊天请求失败: {e}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"处理聊天请求失败: {str(e)}")

@app.get("/api/sessions")
async def get_sessions(limit: int = 15):
    """获取会话列表"""
    try:
        sessions = db_manager.get_recent_sessions(limit=limit)
        return JSONResponse(content=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取会话列表失败: {str(e)}")

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, days_limit: int = 7):
    """获取会话历史消息"""
    try:
        messages = db_manager.get_session_messages(session_id, days_limit=days_limit)
        return JSONResponse(content=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取历史消息失败: {str(e)}")

@app.post("/api/sessions/new")
async def create_new_session(request: NewSessionRequest):
    """创建新会话"""
    try:
        session_id = str(uuid.uuid4())
        title = request.title or f"会话_{time.strftime('%Y%m%d_%H%M%S')}"
        
        db_manager.create_session(session_id=session_id, title=title)
        
        return JSONResponse(content={"session_id": session_id, "title": title})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建新会话失败: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    try:
        # 检查会话是否存在
        session = db_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
        
        # 删除会话
        db_manager.delete_session(session_id)
        
        return JSONResponse(content={"message": f"会话 {session_id} 已成功删除"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除会话失败: {str(e)}")

@app.get("/api/download/{report_id}")
async def download_report(report_id: str):
    """下载投资报告"""
    try:
        report_path = summary_agent.get_report_path(report_id, format="docx")
        
        if not report_path or not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="报告不存在")
        
        return FileResponse(
            path=report_path,
            filename=f"投资报告_{report_id}.docx",
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"下载报告失败: {str(e)}")

# 定时任务：清理旧数据
@app.get("/api/admin/clean_data")
async def clean_old_data():
    """清理旧数据（内部使用）"""
    try:
        # 清理7天前的对话数据
        db_manager.clean_old_data(days_limit=7)
        
        # 清理爬取的旧数据（大前天的数据）
        # 这里可以根据实际情况扩展
        
        return JSONResponse(content={"message": "数据清理完成"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清理数据失败: {str(e)}")

# 市场数据更新检查API
@app.get("/api/market/check-update")
async def check_market_update(last_timestamp: Optional[float] = 0.0):
    """检查市场数据是否有更新
    
    Args:
        last_timestamp: 客户端最后一次更新的时间戳
    
    Returns:
        包含更新状态的JSON响应
    """
    try:
        result = market_analysis_agent.check_market_data_update(last_timestamp)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"检查市场数据更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查市场数据更新失败: {str(e)}")

# 获取市场分析数据API
@app.get("/api/market/analysis")
async def get_market_analysis(check_validity: bool = True):
    """获取市场分析数据
    
    Args:
        check_validity: 是否检查数据有效性
    
    Returns:
        包含市场分析结果的JSON响应
    """
    try:
        result = await market_analysis_agent.analyze_market(check_validity=check_validity)
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"获取市场分析数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取市场分析数据失败: {str(e)}")

# 挂载前端静态文件（注意：必须在所有API路由定义之后挂载）
# 使用绝对路径确保无论从哪个目录运行都能找到frontend目录
frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

# 主函数
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="warning"  # 设置日志级别为warning，减少不必要的输出
    )
