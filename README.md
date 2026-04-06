# AI智能股票投资助手

一个基于大语言模型的智能股票投资推荐系统，采用多Agent架构，提供实时市场分析和多轮对话功能。

## 项目概述

本系统是一个完整的AI股票投资助手，用户通过网页界面可以：
- 获取今日大盘分析和板块轮动信息
- 与系统进行多轮对话，获取股票分析建议
- 查看历史会话记录

系统采用多Agent架构，包含数据获取、市场分析、基本面分析、技术分析和汇总决策等核心Agent，结合4-bit量化的Finance-Llama-8B模型提供专业的投资分析。

## 系统架构

系统采用分层架构设计，主要包括以下层次：

1. **前端层**：基于HTML/CSS/JavaScript的原生前端，提供用户交互界面
2. **API层**：FastAPI实现的RESTful API，处理前端请求
3. **Agent层**：多个专业Agent协同工作，处理不同类型的分析任务
4. **数据层**：多数据源集成，提供股票、财务和新闻数据
5. **模型层**：基于DeepSeek API的大语言模型，提供智能分析和生成能力
6. **存储层**：SQLite数据库存储会话和消息，文件系统存储报告和缓存

## 核心功能模块

### 1. 数据获取模块
- 多数据源集成：akshare、baostock、新浪财经、东方财富等
- 实时行情数据获取和处理
- 历史数据查询和分析
- 技术指标计算
- 智能缓存系统

### 2. 市场分析模块
- 大盘指数分析
- 板块资金流向分析
- 市场情绪分析
- 板块轮动预测
- 热点板块识别

### 3. 股票分析模块
- 基本面分析：财务指标、估值分析
- 技术面分析：趋势、动量、支撑阻力
- 消息面分析：相关新闻和公告
- 综合评分系统

### 4. 对话系统模块
- 多轮对话支持
- 意图识别和理解
- 上下文记忆
- 个性化回答

### 5. 模型集成模块
- 本地Llama模型支持
- DeepSeek API集成（可选）
- 模型推理优化

## 技术特色

1. **多Agent协作**：不同专业Agent协同工作，提供全面的分析
2. **多数据源融合**：从多个数据源获取数据，提高数据可靠性
3. **智能缓存系统**：多级缓存机制，提高响应速度
4. **并发处理**：异步请求和批次处理，提高系统效率
5. **安全配置**：API密钥环境变量管理，提高安全性
6. **可扩展性**：模块化设计，易于添加新功能和数据源

## 技术栈

### 后端
- **Python 3.10+**：开发语言
- **FastAPI**：Web框架
- **Hugging Face Transformers**：模型加载和推理
- **bitsandbytes**：4-bit模型量化
- **SQLite**：会话和消息存储
- **akshare**：股票和财务数据获取
- **baostock**：股票历史数据获取
- **tickflow**：备用数据源

### 模型
- **本地Llama模型**：默认使用4-bit量化的Finance-Llama-8B模型
- **DeepSeek API**：可选，提供在线模型推理

### 前端
- **HTML/CSS/JavaScript**：原生前端开发

### 其他依赖

- **NewsAPI/Serper.dev**：新闻搜索（可配置）
- **asyncio**：并发支持
- **cachetools**：内存缓存
- **diskcache**：磁盘缓存
- **aiohttp**：异步HTTP请求

## 功能特点

### 1. 模型部署
- 直接使用Transformers加载4-bit量化的Finance-Llama-8B模型
- 启动时完成模型加载和预热，确保模型常驻显存
- 推理请求排队处理，避免并发冲突

### 2. 多Agent架构
- **数据获取Agent**：通过多个数据源获取行情、财务、新闻数据，计算技术指标
- **市场分析Agent**：分析大盘指数、板块资金流向、涨跌幅，预测明日板块走势
- **基本面分析Agent**：基于财务数据进行估值、成长性和财务健康度分析
- **技术分析Agent**：分析趋势、动量、支撑阻力，给出技术面评分
- **汇总决策Agent**：整合分析结果，提供投资建议

### 3. 缓存系统
- 多级缓存：内存缓存 + 磁盘缓存
- 智能缓存策略：为不同数据类型设置合理的过期时间
- 缓存预热：系统启动时预先加载常用数据
- 哈希缓存键：使用MD5哈希算法生成紧凑的缓存键

### 4. 多数据源支持
- **akshare**：主要数据源
- **baostock**：备用数据源
- **tickflow**：备用数据源
- **新浪财经**：备用数据源
- **东方财富**：备用数据源
- **必盈API**：备用数据源

### 5. 并发处理
- 异步请求：使用asyncio和aiohttp实现并发请求
- 批次处理：限制并发数量，避免触发API速率限制
- 线程池：用于异步磁盘缓存操作

### 6. 对话记忆
- SQLite存储会话和消息，支持最近7天的对话记忆
- 前端显示最近15条会话，自动清除7天前的对话数据

### 7. 网络搜索集成
- 用户问题涉及新闻时触发搜索API
- 市场分析定期获取重要财经新闻

### 8. 前端界面
- 左侧历史会话列表
- 中间上方大盘概览卡片，显示实时市场信息
- 下方聊天区域，支持发送消息和查看历史对话
- 每条AI回复含报告时显示下载按钮

## 安装步骤

### 1. 克隆项目
```bash
git clone <项目地址>
cd finance_helper
```

### 2. 创建虚拟环境（推荐）
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境变量（可选）
创建 `.env` 文件，添加以下配置：
```
# DeepSeek API配置（必填）
DEEPSEEK_API_KEY=your_deepseek_api_key

# 新闻搜索API（可选）
SERPER_API_KEY=your_serper_api_key

# 聚合数据API（可选）
JUHE_API_KEY=your_juhe_api_key
JUHE_NEWS_API_KEY=your_juhe_news_api_key

# 天API（可选）
TIANAPI_KEY=your_tianapi_key

# TickFlow API（可选）
TICKFLOW_API_KEY=your_tickflow_api_key
```

## 使用说明

### 1. 启动服务
```bash
cd backend
python main.py
```

服务将在 `http://localhost:8000` 启动。

### 2. 访问界面
在浏览器中打开 `http://localhost:8000`，即可使用AI股票投资助手。

### 3. 使用功能

#### 查看市场概览
打开页面后，中间上方的大盘概览卡片会自动加载最新的市场数据，包括：
- 上证指数和深证成指的实时行情
- 今日大盘综述
- 强势板块和板块轮动分析
- 明日板块预测

#### 进行股票分析
在聊天输入框中输入您的问题，例如：
- "请分析股票 600519"
- "贵州茅台的投资价值如何？"

系统会自动识别股票代码，生成详细的投资分析。

#### 管理会话
- 点击左侧的"新会话"按钮创建新会话
- 左侧会话列表显示最近15条会话，点击即可切换
- 系统会自动清除7天前的对话数据

## 项目结构

```
finance_helper/
├── backend/          # 后端代码
│   ├── main.py               # FastAPI主应用
│   ├── model.py              # 模型加载和推理模块
│   ├── logging_config.py     # 日志配置
│   └── db/                   # 数据库文件
│       └── finance_assistant.db  # SQLite数据库文件
├── frontend/         # 前端代码
│   └── index.html    # 主页面
├── agents/           # Agent实现
│   ├── data_agent.py           # 数据获取Agent
│   ├── market_analysis_agent.py # 市场分析Agent
│   ├── fundamental_agent.py    # 基本面分析Agent
│   ├── technical_agent.py      # 技术分析Agent
│   ├── summary_agent.py        # 汇总决策Agent
│   ├── intent_recognizer.py    # 意图识别
│   ├── data_processor.py       # 数据处理器
│   ├── information_retriever.py # 信息检索
│   ├── internet_info_agent.py  # 互联网信息Agent
│   ├── nlp_agent.py            # NLP处理Agent
│   ├── output_processor.py     # 输出处理器
│   └── intent_type.py          # 意图类型定义
├── data/             # 数据文件目录
│   ├── common_indices.json     # 常用指数列表
│   ├── common_sectors.json     # 常用板块列表
│   ├── stock_aliases.json      # 股票别名映射
│   └── stock_name_code_map.json # 股票名称代码映射
├── database/         # 数据库相关
│   ├── database.py             # 数据库管理
│   └── finance_assistant.db    # SQLite数据库文件
├── db/               # 备用数据库目录
│   └── finance_assistant.db    # SQLite数据库文件
├── disk_cache/       # 磁盘缓存目录
├── finance_model_finetuning/  # 模型微调相关
│   ├── configs/               # 配置文件
│   ├── data/                  # 训练数据
│   ├── models/                # 微调模型
│   └── scripts/               # 训练脚本
├── logs/             # 日志目录
├── model/            # 模型文件
│   ├── finetuned/             # 微调后的模型
│   └── mengzi-bert-base-fin/  # 金融领域BERT模型
├── reports/          # 报告存储目录
├── .env              # 环境变量配置
├── main.py           # 主入口文件
├── requirements.txt  # 项目依赖
└── README.md         # 项目说明
```

## API接口

### 市场概览
- **GET /api/market/overview**：获取市场概览数据

### 聊天功能
- **POST /api/chat**：发送聊天消息，返回AI回复
- **GET /api/sessions**：获取会话列表
- **GET /api/sessions/{id}/messages**：获取会话历史消息
- **POST /api/sessions/new**：创建新会话

## 注意事项

1. **模型加载**：首次启动会加载模型，可能需要几分钟时间，请耐心等待

2. **硬件要求**：
   - 建议使用GPU（至少8GB显存）运行
   - CPU也可运行，但推理速度较慢

3. **数据更新**：
   - 市场数据每5分钟自动更新
   - 每日爬取的信息会在3天后自动清除

4. **搜索配置**：
   - 默认使用模拟新闻数据
   - 如需真实新闻搜索，请配置Serper.dev API密钥

5. **数据源配置**：
   - 系统会自动尝试多个数据源，确保数据获取的可靠性
   - 部分数据源可能需要API密钥，请根据需要配置

## 免责声明

本系统提供的投资建议仅供参考，不构成任何投资建议或决策依据。投资有风险，入市需谨慎。用户应根据自身风险承受能力和投资目标，独立做出投资决策。

## 更新日志

- **v1.0.0**：初始版本发布，包含完整的多Agent架构和核心功能
- **v1.1.0**：
  - 优化缓存系统，实现多级缓存和智能缓存策略
  - 添加多数据源支持，提高数据获取的可靠性
  - 优化并发处理，提高系统响应速度
  - 修复bug，提高系统稳定性

## 联系方式

如有问题或建议，请联系开发团队。
