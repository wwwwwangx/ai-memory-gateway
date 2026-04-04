import os
import asyncpg
from typing import Optional, List, Dict
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL", "")

_pool: Optional[asyncpg.Pool] = None

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

async def init_tables():
    pool = await get_pool()
    async with pool.acquire() as conn:
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
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id              SERIAL PRIMARY KEY,
                content         TEXT NOT NULL,
                type            TEXT DEFAULT 'atomic',
                status          TEXT DEFAULT 'active',
                importance      INTEGER DEFAULT 5,
                source_session  TEXT,
                event_time      TIMESTAMPTZ,
                is_completed    BOOLEAN DEFAULT FALSE,
                supersedes_id   INTEGER,
                created_at      TIMESTAMPTZ DEFAULT NOW(),
                updated_at      TIMESTAMPTZ DEFAULT NOW(),
                last_accessed   TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS summaries (
                id              SERIAL PRIMARY KEY,
                type            TEXT NOT NULL,
                content         TEXT NOT NULL,
                start_time      TIMESTAMPTZ NOT NULL,
                end_time        TIMESTAMPTZ NOT NULL,
                created_at      TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        # 兼容旧表缺少字段
        for col in ['type', 'status', 'event_time', 'is_completed', 'supersedes_id', 'updated_at']:
            await conn.execute(f"""
                ALTER TABLE memories ADD COLUMN IF NOT EXISTS {col} 
                {'TEXT DEFAULT ' + ("'atomic'" if col=='type' else "'active'") if col in ('type','status') else 
                 'BOOLEAN DEFAULT FALSE' if col=='is_completed' else
                 'INTEGER' if col=='supersedes_id' else
                 'TIMESTAMPTZ' if col in ('event_time','updated_at') else 'TEXT'}
            """)
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations (session_id, created_at);")
    print("✅ 数据库表结构初始化完成")

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

async def save_memory(content: str, importance: int = 5, source_session: str = ""):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memories (content, importance, source_session) VALUES ($1, $2, $3)",
            content, importance, source_session,
        )

async def search_memories(query: str, limit: int = 10):
    return []

async def get_all_memories():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT content, importance, source_session, created_at FROM memories ORDER BY id")
        return [dict(r) for r in rows]

# 下面是 main.py 可能需要的额外函数（避免 ImportError）
async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"] if row else 0

async def get_all_memories_detail():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, content, importance, source_session, created_at FROM memories ORDER BY id")
        return [dict(r) for r in rows]

async def update_memory(memory_id: int, content: str = None, importance: int = None):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if content is not None and importance is not None:
            await conn.execute("UPDATE memories SET content = $1, importance = $2 WHERE id = $3", content, importance, memory_id)
        elif content is not None:
            await conn.execute("UPDATE memories SET content = $1 WHERE id = $2", content, memory_id)
        elif importance is not None:
            await conn.execute("UPDATE memories SET importance = $1 WHERE id = $2", importance, memory_id)

async def delete_memory(memory_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

async def delete_memories_batch(memory_ids: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = ANY($1::int[])", memory_ids)

async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, content, importance, created_at FROM memories ORDER BY created_at DESC LIMIT $1", limit)
        return [dict(r) for r in rows]