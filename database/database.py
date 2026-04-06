import sqlite3
import time
import datetime
from typing import List, Dict, Optional
import os

class DatabaseManager:
    def __init__(self, db_path: str = "db/finance_assistant.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """初始化数据库表"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建会话表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at INTEGER,
                    updated_at INTEGER
                )
            ''')
            
            # 创建消息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages (created_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions (updated_at)')
            
            conn.commit()
    
    def create_session(self, session_id: str, title: str) -> None:
        """创建新会话"""
        timestamp = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, timestamp, timestamp)
            )
            conn.commit()
    
    def add_message(self, message_id: str, session_id: str, role: str, content: str) -> None:
        """添加消息到会话"""
        timestamp = int(time.time())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 添加消息
            cursor.execute(
                "INSERT INTO messages (id, session_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (message_id, session_id, role, content, timestamp)
            )
            
            # 更新会话更新时间
            cursor.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id)
            )
            
            conn.commit()
    
    def get_session_messages(self, session_id: str, days_limit: int = 7) -> List[Dict]:
        """获取会话的最近几天消息"""
        days_ago = int(time.time()) - (days_limit * 24 * 3600)
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? AND created_at >= ? ORDER BY created_at",
                (session_id, days_ago)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_recent_sessions(self, limit: int = 15) -> List[Dict]:
        """获取最近的会话列表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def clean_old_data(self, days_limit: int = 7) -> None:
        """清理指定天数前的对话数据"""
        days_ago = int(time.time()) - (days_limit * 24 * 3600)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 清理旧消息
            cursor.execute("DELETE FROM messages WHERE created_at < ?", (days_ago,))
            
            # 清理没有消息的会话
            cursor.execute(
                "DELETE FROM sessions WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)"
            )
            
            conn.commit()
    
    def clean_crawled_data(self, days_limit: int = 3) -> None:
        """清理指定天数前的爬取数据"""
        # 这里可以扩展，如果有单独的爬取数据表
        pass
    
    def update_session_title(self, session_id: str, title: str) -> None:
        """更新会话标题"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE sessions SET title = ? WHERE id = ?",
                (title, session_id)
            )
            conn.commit()
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """获取会话信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def delete_session(self, session_id: str) -> None:
        """删除会话及其所有消息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 删除会话的所有消息
            cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            
            # 删除会话本身
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            
            conn.commit()

# 全局数据库实例
db_manager = DatabaseManager()
