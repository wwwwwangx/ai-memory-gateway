import os
import re
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import asyncpg
import jieba
import jieba.analyse

DATABASE_URL = os.getenv("DATABASE_URL", "")

WEIGHT_KEYWORD = float(os.getenv("WEIGHT_KEYWORD", "0.5"))
WEIGHT_IMPORTANCE = float(os.getenv("WEIGHT_IMPORTANCE", "0.3"))
WEIGHT_RECENCY = float(os.getenv("WEIGHT_RECENCY", "0.2"))
MIN_SCORE_THRESHOLD = float(os.getenv("MIN_SCORE_THRESHOLD", "0.15"))

MAX_CONTEXT_MEMORIES = int(os.getenv("MAX_CONTEXT_MEMORIES", "15"))
TOP_K = 3
HIGH_IMPORTANCE_THRESHOLD = 7
HIGH_IMPORTANCE_LIMIT = 5

DEDUP_SIMILARITY = float(os.getenv("DEDUP_SIMILARITY", "0.6"))

CORE_PROMOTION_DAYS = 7
CORE_PROMOTION_COUNT = 3

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
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS type TEXT DEFAULT 'atomic';")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active';")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS event_time TIMESTAMPTZ;")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS is_completed BOOLEAN DEFAULT FALSE;")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS supersedes_id INTEGER;")
        await conn.execute("ALTER TABLE memories ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();")
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
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_event_time ON memories (event_time);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories (supersedes_id);")
    print("✅ 数据库表结构初始化完成")

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

def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

async def save_memory(
    content: str,
    importance: int = 5,
    source_session: str = "",
    type: str = "atomic",
    event_time: Optional[datetime] = None,
    is_completed: bool = False,
) -> int:
    pool = await get_pool()
    async with pool.acquire() as conn:
        dup = await conn.fetchrow(
            "SELECT id, content, importance FROM memories WHERE status = 'active' AND content ILIKE $1 LIMIT 1",
            f"%{content[:50]}%"
        )
        if dup and text_similarity(dup["content"], content) >= DEDUP_SIMILARITY:
            old_id = dup["id"]
            new_importance = max(importance, dup["importance"])
            await conn.execute(
                "UPDATE memories SET status = 'superseded', updated_at = NOW() WHERE id = $1",
                old_id
            )
            row = await conn.fetchrow(
                """
                INSERT INTO memories (content, importance, source_session, type, event_time, is_completed, supersedes_id, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
                RETURNING id
                """,
                content, new_importance, source_session, type, event_time, is_completed, old_id
            )
            print(f"🔄 记忆去重：旧ID {old_id} 被 supersede，新ID {row['id']}")
            return row["id"]
        else:
            row = await conn.fetchrow(
                """
                INSERT INTO memories (content, importance, source_session, type, event_time, is_completed, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, NOW())
                RETURNING id
                """,
                content, importance, source_session, type, event_time, is_completed
            )
            return row["id"]

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

async def get_memories_for_context(session_id: str, user_query: str, limit: int = MAX_CONTEXT_MEMORIES) -> List[Dict]:
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
                    WHERE status = 'active' AND to_tsvector('simple', content) @@ to_tsquery('simple', $1)
                    ORDER BY ts_rank(to_tsvector('simple', content), to_tsquery('simple', $1)) DESC,
                             importance DESC, created_at DESC
                    LIMIT $2
                    """,
                    ts_query, TOP_K
                )
                top3 = [dict(r) for r in rows]
        
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
        
        plans = await conn.fetch(
            """
            SELECT id, content, importance, created_at, type, is_completed
            FROM memories
            WHERE status = 'active' AND type = 'plan' AND is_completed = false
            ORDER BY importance DESC, created_at DESC
            """
        )
        plans = [dict(r) for r in plans]
        
        merged = {}
        for m in top3 + high_imp + plans:
            if m["id"] not in merged:
                merged[m["id"]] = m
        
        result = list(merged.values())
        result.sort(key=lambda x: (x["importance"], x["created_at"]), reverse=True)
        return result[:limit]

async def search_memories(query: str, limit: int = 10):
    keywords = extract_search_keywords(query)
    if not keywords:
        return []
    pool = await get_pool()
    async with pool.acquire() as conn:
        ts_query = " & ".join(keywords)
        rows = await conn.fetch(
            """
            SELECT id, content, importance, created_at,
                   ts_rank(to_tsvector('simple', content), to_tsquery('simple', $1)) AS rank
            FROM memories
            WHERE status = 'active' AND to_tsvector('simple', content) @@ to_tsquery('simple', $1)
            ORDER BY rank DESC, importance DESC, created_at DESC
            LIMIT $2
            """,
            ts_query, limit
        )
        return [dict(r) for r in rows]

async def get_recent_memories(limit: int = 20):
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content, importance, created_at, type, status, is_completed FROM memories ORDER BY created_at DESC LIMIT $1",
            limit,
        )
        return [dict(r) for r in rows]

async def get_all_memories_count():
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT COUNT(*) as cnt FROM memories")
        return row["cnt"]

async def get_all_memories():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT content, importance, source_session, created_at, type, status, is_completed FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]

async def get_all_memories_detail():
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, content, importance, source_session, created_at, type, status, is_completed FROM memories ORDER BY id"
        )
        return [dict(r) for r in rows]

async def delete_memory(memory_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

async def delete_memories_batch(memory_ids: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM memories WHERE id = ANY($1::int[])", memory_ids)

async def user_correction(memory_id: int, action: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        if action == "lower_importance":
            await conn.execute(
                "UPDATE memories SET importance = GREATEST(1, importance - 2), updated_at = NOW() WHERE id = $1",
                memory_id
            )
        elif action == "archive":
            await conn.execute(
                "UPDATE memories SET status = 'archived', updated_at = NOW() WHERE id = $1",
                memory_id
            )
        elif action == "delete":
            await conn.execute("DELETE FROM memories WHERE id = $1", memory_id)

async def promote_atomic_to_core():
    pool = await get_pool()
    async with pool.acquire() as conn:
        cutoff = datetime.now() - timedelta(days=CORE_PROMOTION_DAYS)
        rows = await conn.fetch(
            """
            SELECT id, content, importance, source_session, created_at
            FROM memories
            WHERE type = 'atomic' AND status = 'active' AND created_at >= $1
            ORDER BY created_at DESC
            """,
            cutoff
        )
        groups = {}
        for r in rows:
            key = r["content"][:20]
            if key not in groups:
                groups[key] = []
            groups[key].append(r)
        
        for key, mems in groups.items():
            if len(mems) >= CORE_PROMOTION_COUNT:
                ids = [m["id"] for m in mems]
                await conn.execute(
                    "UPDATE memories SET type = 'core', updated_at = NOW() WHERE id = ANY($1::int[])",
                    ids
                )
                combined_content = f"【核心习惯/偏好】{key}…（基于 {len(mems)} 条记录自动升级）"
                await save_memory(
                    content=combined_content,
                    importance=max(m["importance"] for m in mems),
                    source_session=mems[0]["source_session"],
                    type="core"
                )
                print(f"⭐ 自动升级 Core: {key} (合并 {len(mems)} 条 atomic)")

async def get_last_summary_time(summary_type: str) -> Optional[datetime]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT end_time FROM summaries WHERE type = $1 ORDER BY end_time DESC LIMIT 1",
            summary_type
        )
        return row["end_time"] if row else None

async def generate_summary_if_needed(session_id: str):
    pool = await get_pool()
    async with pool.acquire() as conn:
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM conversations WHERE session_id = $1 AND created_at >= $2",
            session_id, today_start
        )
        last_daily = await get_last_summary_time("daily")
        if count >= 10 or (last_daily and (datetime.now() - last_daily).total_seconds() >= 86400):
            await _create_summary(session_id, "daily", days=1)
        
        last_weekly = await get_last_summary_time("weekly")
        if not last_weekly or (datetime.now() - last_weekly).total_seconds() >= 604800:
            await _create_summary(session_id, "weekly", days=7)
        
        last_monthly = await get_last_summary_time("monthly")
        if not last_monthly or (datetime.now() - last_monthly).total_seconds() >= 2592000:
            await _create_summary(session_id, "monthly", days=30)

async def _create_summary(session_id: str, summary_type: str, days: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        start_time = datetime.now() - timedelta(days=days)
        rows = await conn.fetch(
            "SELECT role, content, created_at FROM conversations WHERE session_id = $1 AND created_at >= $2 ORDER BY created_at",
            session_id, start_time
        )
        if not rows:
            return
        summary_content = f"【{summary_type.upper()} 总结】\n"
        for r in rows[-50:]:
            summary_content += f"{r['role']}: {r['content'][:100]}\n"
        summary_content += f"共 {len(rows)} 条对话。"
        
        await conn.execute(
            """
            INSERT INTO summaries (type, content, start_time, end_time)
            VALUES ($1, $2, $3, $4)
            """,
            summary_type, summary_content, start_time, datetime.now()
        )
        print(f"📝 生成 {summary_type} 摘要，覆盖 {len(rows)} 条对话")