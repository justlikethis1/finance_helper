import pandas as pd
import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Callable

logger = logging.getLogger(__name__)


def convert_numpy_types(obj: Any) -> Any:
    """
    将numpy类型转换为原生Python类型
    
    Args:
        obj: 需要转换的对象
        
    Returns:
        转换后的对象
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


class DataProcessor:
    """
    通用数据处理器，用于处理不同API返回的数据字段映射、验证和异常处理
    """
    
    def __init__(self):
        # 预定义的字段映射，支持多语言和不同API的字段名称
        self.field_mappings = {
            '板块名称': ['板块', 'name', '板块名称', '名称', 'industry_name', 'industry', 'sector', '板块编号', '行业名称'],
            '涨跌幅': ['涨跌幅', 'change_pct', '涨跌幅%', '涨跌', 'change', 'pct_change', 'percent_change', 'return'],
            '成交量': ['总成交量', 'volume', '成交量', '成交额', 'vol', 'amount', 'turnover'],
            '指数名称': ['index_name', '名称', '指数', '指数代码', 'code', 'symbol'],
            '开盘价': ['open', '开盘', '开盘价'],
            '收盘价': ['close', '收盘', '收盘价'],
            '最高价': ['high', '最高', '最高价'],
            '最低价': ['low', '最低', '最低价'],
        }
        
        # 字段默认值配置
        self.default_values = {
            '板块名称': '未知板块',
            '涨跌幅': 0.0,
            '成交量': 0,
            '指数名称': '未知指数',
            '开盘价': 0.0,
            '收盘价': 0.0,
            '最高价': 0.0,
            '最低价': 0.0,
        }
        
        # 预设的板块列表，用于数据严重异常时的回退
        self.preset_sectors = [
            '电力', '通信服务', '通信设备', '旅游及酒店', '贵金属',
            '教育', '计算机设备', '医药生物', '电子', '机械设备'
        ]
    
    def map_fields(self, df: pd.DataFrame, required_fields: List[str]) -> pd.DataFrame:
        """
        根据预定义的映射规则处理DataFrame字段
        
        Args:
            df: 输入的DataFrame
            required_fields: 需要确保存在的字段列表
            
        Returns:
            处理后的DataFrame
        """
        if df.empty:
            logger.warning("输入DataFrame为空，返回空DataFrame")
            return df
        
        # 直接修改原始DataFrame，避免不必要的copy操作
        # 只在需要时才copy
        if set(required_fields).issubset(df.columns):
            # 所有字段都存在，检查数据有效性
            valid = True
            for field in required_fields:
                if field in ['涨跌幅', '成交量', '开盘价', '收盘价', '最高价', '最低价']:
                    try:
                        numeric_data = pd.to_numeric(df[field], errors='coerce')
                        if numeric_data.isnull().all() or (numeric_data.abs() < 0.001).all():
                            valid = False
                            break
                    except Exception:
                        valid = False
                        break
            if valid:
                return df
        
        # 需要处理字段映射，创建copy
        df_processed = df.copy()
        
        # 预处理：创建列名到字段的映射，减少重复查找
        column_to_field = {}
        for field in required_fields:
            if field in df_processed.columns:
                column_to_field[field] = field
            elif field in self.field_mappings:
                for alt_field in self.field_mappings[field]:
                    if alt_field in df_processed.columns:
                        column_to_field[alt_field] = field
                        break
        
        # 处理每个需要的字段
        for field in required_fields:
            if field in df_processed.columns:
                # 检查字段是否已有有效数据
                if field in ['涨跌幅', '成交量', '开盘价', '收盘价', '最高价', '最低价']:
                    try:
                        numeric_data = pd.to_numeric(df_processed[field], errors='coerce')
                        if not numeric_data.isnull().all() and not (numeric_data.abs() < 0.001).all():
                            continue
                    except Exception:
                        pass
            
            # 查找替代字段
            mapped = False
            if field in self.field_mappings:
                for alt_field in self.field_mappings[field]:
                    if alt_field in df_processed.columns:
                        # 对不同类型的字段使用不同的处理逻辑
                        if field in ['涨跌幅', '成交量', '开盘价', '收盘价', '最高价', '最低价']:
                            # 数值字段，尝试转换为数值
                            try:
                                numeric_data = pd.to_numeric(df_processed[alt_field], errors='coerce')
                                if not numeric_data.isnull().all() and not (numeric_data.abs() < 0.001).all():
                                    df_processed[field] = numeric_data
                                    if logger.isEnabledFor(logging.INFO):
                                        logger.info(f"将 {alt_field} 转换并命名为 {field}")
                                    mapped = True
                                    break
                            except Exception:
                                continue
                        else:
                            # 非数值字段，直接映射
                            df_processed[field] = df_processed[alt_field]
                            if logger.isEnabledFor(logging.INFO):
                                logger.info(f"将 {alt_field} 重命名为 {field}")
                            mapped = True
                            break
            
            # 特殊处理涨跌幅字段
            if field == '涨跌幅' and not mapped:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"数据缺少涨跌幅字段的直接映射，尝试从原始数据中提取")
                
                # 尝试查找包含涨跌幅信息的字段
                for col in df_processed.columns:
                    if any(keyword in col.lower() for keyword in ['change', '涨', '跌', 'pct', 'percent']):
                        try:
                            numeric_data = pd.to_numeric(df_processed[col], errors='coerce')
                            if not numeric_data.isnull().all() and not (numeric_data.abs() < 0.001).all():
                                df_processed[field] = numeric_data
                                if logger.isEnabledFor(logging.INFO):
                                    logger.info(f"从 {col} 提取涨跌幅信息")
                                mapped = True
                                break
                        except Exception:
                            continue
            
            # 添加默认值
            if not mapped:
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(f"数据缺少字段: {field}，将添加默认值")
                df_processed[field] = self.default_values.get(field, None)
        
        return df_processed
    
    def validate_data(self, df: pd.DataFrame, required_fields: List[str]) -> bool:
        """
        验证数据是否包含必要的字段和有效数据
        
        Args:
            df: 输入的DataFrame
            required_fields: 需要验证的字段列表
            
        Returns:
            数据是否有效的布尔值
        """
        if df.empty:
            logger.error("数据为空")
            return False
            
        # 检查必要字段是否存在
        for field in required_fields:
            if field not in df.columns:
                logger.error(f"缺少必要字段: {field}")
                return False
        
        # 检查数据是否有有效内容
        for field in required_fields:
            if df[field].isnull().all() or (df[field].astype(str) == '').all():
                logger.error(f"字段 {field} 没有有效数据")
                return False
        
        return True
    
    def fix_missing_sectors(self, df: pd.DataFrame, sector_field: str = '板块名称') -> pd.DataFrame:
        """
        修复缺少板块名称的数据
        
        Args:
            df: 输入的DataFrame
            sector_field: 板块名称字段
            
        Returns:
            修复后的DataFrame
        """
        if df.empty:
            return df
            
        df_fixed = df.copy()
        
        # 检查是否有有效板块名称
        has_valid_sectors = not (df_fixed[sector_field].isnull().all() or 
                                (df_fixed[sector_field].astype(str) == '').all() or
                                (df_fixed[sector_field].astype(str) == self.default_values[sector_field]).all())
        
        if has_valid_sectors:
            return df_fixed
        
        # 如果没有有效板块名称，使用预设的板块列表
        logger.warning("数据中没有有效板块名称，使用预设板块列表")
        
        # 确保预设列表足够长
        num_sectors = len(df_fixed)
        sectors_to_use = self.preset_sectors[:num_sectors] if num_sectors <= len(self.preset_sectors) else self.preset_sectors
        
        # 如果预设列表不够，复制补充
        if len(sectors_to_use) < num_sectors:
            repeated = (num_sectors // len(sectors_to_use)) + 1
            sectors_to_use = (sectors_to_use * repeated)[:num_sectors]
        
        df_fixed[sector_field] = sectors_to_use
        return df_fixed
    
    def get_top_items(self, df: pd.DataFrame, field: str, sort_field: str, ascending: bool = False, top_n: int = 5) -> List[Any]:
        """
        获取按指定字段排序的前N个项目
        
        Args:
            df: 输入的DataFrame
            field: 要获取的字段名
            sort_field: 用于排序的字段名
            ascending: 是否升序排序
            top_n: 要获取的项目数量
            
        Returns:
            排序后的项目列表
        """
        if df.empty:
            logger.warning("输入DataFrame为空，返回空列表")
            return []
        
        if sort_field not in df.columns:
            logger.error(f"排序字段 {sort_field} 不存在")
            return []
            
        if field not in df.columns:
            logger.error(f"获取字段 {field} 不存在")
            return []
        
        try:
            # 按指定字段排序
            sorted_df = df.sort_values(by=sort_field, ascending=ascending)
            
            # 获取前N个项目
            top_items = sorted_df[field].head(top_n).tolist()
            
            return top_items
        except Exception as e:
            logger.error(f"获取前N个项目失败: {e}")
            return []
    
    def add_data_quality_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        添加数据质量信息
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            数据质量信息字典
        """
        if df.empty:
            return {
                'total_rows': 0,
                'total_columns': 0,
                'columns': [],
                'missing_values': {},
                'data_types': {},
                'quality_score': 0.0,
                'is_valid': False,
                'warning_messages': ['DataFrame is empty']
            }
            
        quality_info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': {},
            'data_types': {},
            'unique_values': {},
            'quality_score': 0.0,
            'is_valid': True,
            'warning_messages': []
        }
        
        # 计算基本统计信息
        for col in df.columns:
            quality_info['missing_values'][col] = df[col].isnull().sum()
            quality_info['data_types'][col] = str(df[col].dtype)
            quality_info['unique_values'][col] = df[col].nunique()
            
            # 检查缺失值比例
            missing_ratio = df[col].isnull().sum() / len(df)
            if missing_ratio > 0.8:
                quality_info['warning_messages'].append(f"Column '{col}' has high missing value ratio: {missing_ratio:.2%}")
            elif missing_ratio > 0.5:
                quality_info['warning_messages'].append(f"Column '{col}' has moderate missing value ratio: {missing_ratio:.2%}")
        
        # 检查是否有重复行
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            quality_info['warning_messages'].append(f"Found {duplicate_rows} duplicate rows")
        
        # 计算数据质量评分 (0-100)
        try:
            # 基础分数（非空）
            base_score = 80 if len(df) > 0 and len(df.columns) > 0 else 0
            
            # 缺失值扣分（最多扣30分）
            total_cells = len(df) * len(df.columns)
            total_missing = sum(quality_info['missing_values'].values())
            missing_penalty = min(30, (total_missing / total_cells) * 100) if total_cells > 0 else 0
            
            # 重复行扣分（最多扣10分）
            duplicate_penalty = min(10, (duplicate_rows / len(df)) * 100) if len(df) > 0 else 0
            
            quality_info['quality_score'] = max(0, base_score - missing_penalty - duplicate_penalty)
            
            # 如果质量评分过低，标记为无效
            if quality_info['quality_score'] < 30:
                quality_info['is_valid'] = False
                quality_info['warning_messages'].append(f"Data quality score is too low: {quality_info['quality_score']:.1f}")
        except Exception as e:
            logger.error(f"Error calculating data quality score: {e}")
            quality_info['quality_score'] = 0.0
            quality_info['is_valid'] = False
            quality_info['warning_messages'].append(f"Failed to calculate data quality score: {e}")
        
        return quality_info
    
    def log_data_quality(self, df: pd.DataFrame, data_source: str = "unknown") -> None:
        """
        记录数据质量信息到日志
        
        Args:
            df: 输入的DataFrame
            data_source: 数据来源标识
        """
        quality_info = self.add_data_quality_info(df)
        quality_info['data_source'] = data_source
        
        # 转换numpy类型为原生Python类型，避免JSON序列化错误
        quality_info = convert_numpy_types(quality_info)
        
        # 根据质量评分选择日志级别
        if not quality_info['is_valid']:
            logger.error(f"Data quality report ({data_source}): {json.dumps(quality_info, ensure_ascii=False)}")
        elif quality_info['quality_score'] < 60:
            logger.warning(f"Data quality report ({data_source}): {json.dumps(quality_info, ensure_ascii=False)}")
        else:
            logger.info(f"Data quality report ({data_source}): {json.dumps(quality_info, ensure_ascii=False)}")
