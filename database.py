import os
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import asyncpg
import jieba
import jieba.analyse

DATABASE_URL = os.getenv("DATABASE_URL", "")

# 召回配置
MAX_CONTEXT_MEMORIES = int(os.getenv("MAX_CONTEXT_MEMORIES", "15"))
TOP_K = 3
HIGH_IMPORTANCE_THRESHOLD = 7
HIGH_IMPORTANCE_LIMIT = 5

# 分词初始化
jieba.setLogLevel(jieba.logging.INFO)

EN_WORD_PATTERN = re.compile(r'[a-zA-Z][a-zA-Z0-9]*')
NUM_PATTERN = re.compile(r'\d{2,}')

_STOP_WORDS = frozenset({
    "的", "了", "在", "是", "我", "你", "他", "她", "它", "们",
    "这", "那", "有", "和", "与", "也", "都", "又", "就", "但",
    "而", "或", "到", "被", "把", "让", "从", "对", "为", "以",
    "及", "等", "个", "不", "没", "很", "太", "吗", "呢", "吧",
    "啊", "嗯", "哦", "哈", "呀", "嘛", "么", "啦", "哇", "喔",
    "会", "能", "要", "想", "去", "来", "说", "做", "看", "给",
    "上", "下", "里", "中", "大", "小", "多", "少", "好", "可以",
    "什么", "怎么", "如何", "哪里", "哪个", "为什么", "还是",
    "然后", "因为", "所以", "虽然", "但是", "可以", "已经",
    "一个", "一些", "一下", "一点", "一起", "一样",
    "比较", "应该", "可能", "如果", "这个", "那个",
    "自己", "知道", "觉得", "感觉", "时候", "现在",
})

_pool: Optional[asyncpg.Pool] = None

# ========== 连接池 ==========
async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise RuntimeError("DATABASE_URL 未设置！")
        _pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
        print("✅ 数据库连接池已创建")
    return _pool

async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        print("✅ 数据库连接池已关闭")

# ========== 表初始化（升级版，增加必要字段和索引） ==========
async def init_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
        # 对话表
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id              SERIAL PRIMARY KEY,
                session_id      TEXT NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                model           TEXT,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # 记忆表（添加 type, status, is_completed 字段）
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                type            TEXT DEFAULT 'atomic',
                status          TEXT DEFAULT 'active',
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                is_completed    BOOLEAN DEFAULT FALSE,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # 补充缺失字段（兼容旧表）
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS type TEXT DEFAULT 'atomic';")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_completed BOOLEAN DEFAULT FALSE;")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed TIMESTAMPTZ DEFAULT NOW();")
        # 索引
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_fts 
            ON memories USING gin(to_tsvector('simple', content));
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_session 
            ON conversations (session_id, created_at);
        """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_status ON memories (status);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_completed ON memories (is_completed);")
    print("✅ 数据库表结构初始化完成")

# ========== 中文分词 ==========
def extract_search_keywords(query: str) -> List[str]:
    keywords = set()
    for match in EN_WORD_PATTERN.finditer(query):
        word = match.group()
        if len(word) >= 2:
            keywords.add(word)
    for match in NUM_PATTERN.finditer(query):
        keywords.add(match.group())
    words = jieba.cut(query, cut_all=False)
    for word in words:
        word = word.strip()
        if not word:
            continue
        if EN_WORD_PATTERN.fullmatch(word) or NUM_PATTERN.fullmatch(word):
            continue
        if len(word) < 2 or word in _STOP_WORDS:
            continue
        keywords.add(word)
    return list(keywords)

# ========== 对话操作 ==========
async def save_message(session_id: str, role: str, content: str, model: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO conversations (session_id, role, content, model) VALUES ($1, $2, $3, $4)",
            session_id, role, content, model,
        )

async def get_recent_messages(session_id: str, limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content, created_at FROM conversations WHERE session_id = $1 ORDER BY created_at DESC LIMIT $2",
            session_id, limit,
        )
        return list(reversed(rows))

# ========== 记忆保存（简化版，但支持新字段） ==========
async def save_memory(content: str, importance: int = 5, source_session: str = "", 
                      mem_type: str = "atomic", is_completed: bool = False):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO memories (content, importance, source_session, type, is_completed, updated_at)
            VALUES ($1, $2, $3, $4, $5, NOW())
            """,
            content, importance, source_session, mem_type, is_completed,
        )

# ========== 核心召回逻辑 ==========
async def get_memories_for_context(session_id: str, user_query: str, limit: int = MAX_CONTEXT_MEMORIES) -> List[Dict]:
    """
    返回注入上下文的记忆列表：
    - 只取 status='active'
    - Top3 相关记忆（基于用户查询的全文搜索）
    - 高重要性记忆 (importance >= 7) 最多5条
    - 未完成的计划 (type='plan' and is_completed=false)
    - 合并去重，按重要性+新鲜度排序，截断
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        top3 = []
        if user_query:
            keywords = extract_search_keywords(user_query)
            if keywords:
                ts_query = " & ".join(keywords)
                rows = await conn.fetch(
                    """
                    SELECT id, content, importance, created_at, type, is_completed
                    FROM memories
                    WHERE status = 'active' 
                      AND to_tsvector('simple', content) @@ to_tsquery('simple', $1)
                    ORDER BY ts_rank(to_tsvector('simple', content), to_tsquery('simple', $1)) DESC,
                             importance DESC, created_at DESC
                    LIMIT $2
                    """,
                    ts_query, TOP_K
                )
                top3 = [dict(r) for r in rows]
        
        # 高重要性记忆
        high_imp = await conn.fetch(
            """
            SELECT id, content, importance, created_at, type, is_completed
            FROM memories
            WHERE status = 'active' AND importance >= $1
            ORDER BY importance DESC, created_at DESC
            LIMIT $2
            """,
            HIGH_IMPORTANCE_THRESHOLD, HIGH_IMPORTANCE_LIMIT
        )
        high_imp = [dict(r) for r in high_imp]
        
        # 未完成的计划
        plans = await conn.fetch(
            """
            SELECT id, content, importance, created_at, type, is_completed
            FROM memories
            WHERE status = 'active' AND type = 'plan' AND is_completed = false
            ORDER BY importance DESC, created_at DESC
            """
        )
        plans = [dict(r) for r in plans]
        
        # 合并去重（按 id）
        merged = {}
        for m in top3 + high_imp + plans:
            if m["id"] not in merged:
                merged[m["id"]] = m
        
        result = list(merged.values())
        # 排序：重要性优先，其次新鲜度
        result.sort(key=lambda x: (x["importance"], x["created_at"]), reverse=True)
        return result[:limit]

# 为了兼容旧代码，保留 search_memories 但改为调用新逻辑（也可返回空，但建议用上面的函数）
async def search_memories(query: str, limit: int = 10):
    """兼容旧接口，返回基于查询的相关记忆（仅 active）"""
    if not query:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        keywords = extract_search_keywords(query)
        if not keywords:
            return []
        ts_query = " & ".join(keywords)
        rows = await conn.fetch(
            """
            SELECT id, content, importance, created_at,
                   ts_rank(to_tsvector('simple', content), to_tsquery('simple', $1)) AS rank
            FROM memories
            WHERE status = 'active' 
              AND to_tsvector('simple', content) @@ to_tsquery('simple', $1)
            ORDER BY rank DESC, importance DESC, created_at DESC
            LIMIT $2
            """,
            ts_query, limit
        )
        return [dict(r) for r in rows]

# ========== 辅助函数 ==========
async def get_all_memories():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content, importance, source_session, created_at, type, status, is_completed FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]

async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]

async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content, importance, created_at, type, status, is_completed FROM memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )
        return [dict(r) for r in rows]

async def update_memory(memory_id: int, content: str = None, importance: int = None, is_completed: bool = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        updates = []
        params = []
        if content is not None:
            updates.append("content = $1")
            params.append(content)
        if importance is not None:
            updates.append("importance = $2")
            params.append(importance)
        if is_completed is not None:
            updates.append("is_completed = $3")
            params.append(is_completed)
        if not updates:
            return
        updates.append("updated_at = NOW()")
        params.append(memory_id)
        await conn.execute(
            f"UPDATE memories SET {', '.join(updates)} WHERE id = ${len(params)}",
            *params
        )

async def delete_memory(memory_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

async def delete_memories_batch(memory_ids: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = ANY($1::int[])", memory_ids)