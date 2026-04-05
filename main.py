"""
AI Memory Gateway — 带记忆系统的 LLM 转发网关
"""

import os
import json
import uuid
import asyncio
import re
import httpx
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from database import (
    init_tables, close_pool, save_message, search_memories, save_memory,
    get_all_memories_count, get_recent_memories, get_all_memories, get_pool,
    get_all_memories_detail, update_memory, delete_memory, delete_memories_batch,
    get_memories_for_context
)
from memory_extractor import extract_memories, score_memories

API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "anthropic/claude-sonnet-4")
PORT = int(os.getenv("PORT", "8080"))
MEMORY_ENABLED = os.getenv("MEMORY_ENABLED", "false").lower() == "true"
MAX_MEMORIES_INJECT = int(os.getenv("MAX_MEMORIES_INJECT", "15"))
MEMORY_EXTRACT_INTERVAL = int(os.getenv("MEMORY_EXTRACT_INTERVAL", "1"))
TIMEZONE_HOURS = int(os.getenv("TIMEZONE_HOURS", "8"))
FORCE_STREAM = os.getenv("FORCE_STREAM", "false").lower() == "true"
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "")
EXTRA_REFERER = os.getenv("EXTRA_REFERER", "https://ai-memory-gateway.local")
EXTRA_TITLE = os.getenv("EXTRA_TITLE", "AI Memory Gateway")

FORCE_SAVE_PATTERNS = [
    r"记住[：:]\s*(.+)",
    r"我喜欢(.+)",
    r"我讨厌(.+)",
    r"我需要(.+)",
    r"提醒我(.+)",
    r"别忘了(.+)",
]

_round_counter = 0

def load_system_prompt():
    prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.txt")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                return content
    except FileNotFoundError:
        pass
    print("ℹ️  未找到 system_prompt.txt")
    return ""

SYSTEM_PROMPT = load_system_prompt()
if SYSTEM_PROMPT:
    print(f"✅ 人设已加载，长度：{len(SYSTEM_PROMPT)} 字符")
else:
    print("ℹ️  无人设，纯转发模式")

@asynccontextmanager
async def lifespan(app: FastAPI):
    if MEMORY_ENABLED:
        try:
            await init_tables()
            count = await get_all_memories_count()
            print(f"✅ 记忆系统已启动，当前记忆数量：{count}")
        except Exception as e:
            print(f"⚠️  数据库初始化失败: {e}")
    else:
        print("ℹ️  记忆系统已关闭")
    yield
    if MEMORY_ENABLED:
        await close_pool()

app = FastAPI(title="AI Memory Gateway", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def extract_memory_from_message(content: str):
    for pattern in FORCE_SAVE_PATTERNS:
        match = re.search(pattern, content)
        if match:
            mem_content = match.group(1).strip()
            if mem_content:
                if "喜欢" in pattern or "讨厌" in pattern:
                    importance = 7
                elif "需要" in pattern or "提醒" in pattern:
                    importance = 8
                    mem_type = "plan"
                else:
                    importance = 6
                    mem_type = "atomic"
                return (mem_content, importance, mem_type)
    return None

async def build_system_prompt_with_memories(user_message: str) -> str:
    if not MEMORY_ENABLED:
        return SYSTEM_PROMPT
    try:
        memories = await get_memories_for_context("", user_message, limit=MAX_MEMORIES_INJECT)
        print(f"🔍 召回调试: 用户消息='{user_message[:50]}', 召回记忆数量={len(memories)}")
        if memories:
            print(f"📌 召回示例: {memories[0]['content'][:80]}")
        if not memories:
            return SYSTEM_PROMPT
        
        memory_lines = []
        for mem in memories:
            date_str = ""
            if mem.get("created_at"):
                try:
                    utc_str = str(mem['created_at'])[:19]
                    utc_dt = datetime.strptime(utc_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
                    local_dt = utc_dt + timedelta(hours=TIMEZONE_HOURS)
                    date_str = f"[{local_dt.strftime('%Y-%m-%d')}] "
                except:
                    date_str = f"[{str(mem['created_at'])[:10]}] "
            memory_lines.append(f"- {date_str}{mem['content']}")
        memory_text = "\n".join(memory_lines)
        
        enhanced_prompt = f"""{SYSTEM_PROMPT}

【从过往对话中检索到的相关记忆】
{memory_text}

# 记忆应用
- 像朋友般自然运用这些记忆，不刻意展示
- 仅在相关话题出现时引用
- 对重要信息保持一致性
- 新信息与记忆冲突时，以新信息为准"""
        print(f"📚 注入了 {len(memories)} 条相关记忆，总system长度={len(enhanced_prompt)}")
        return enhanced_prompt
    except Exception as e:
        print(f"⚠️  记忆检索失败: {e}")
        return SYSTEM_PROMPT

async def process_memories_background(session_id: str, user_msg: str, assistant_msg: str, model: str, context_messages: list = None):
    global _round_counter
    try:
        await save_message(session_id, "user", user_msg, model)
        await save_message(session_id, "assistant", assistant_msg, model)
        
        direct_mem = extract_memory_from_message(user_msg)
        if direct_mem:
            content, importance, mem_type = direct_mem
            await save_memory(
                content=content,
                importance=importance,
                source_session=session_id,
                mem_type=mem_type,
                is_completed=(mem_type == "plan" and "完成" in user_msg)
            )
            print(f"💾 直接保存记忆: {content[:50]}... (重要性={importance})")
        
        if MEMORY_EXTRACT_INTERVAL == 0:
            return
        _round_counter += 1
        if MEMORY_EXTRACT_INTERVAL > 1 and (_round_counter % MEMORY_EXTRACT_INTERVAL != 0):
            return
        
        existing = await get_recent_memories(limit=80)
        existing_contents = [r["content"] for r in existing]
        
        if context_messages:
            tail_count = MEMORY_EXTRACT_INTERVAL * 2
            recent_msgs = list(context_messages)[-tail_count:] if len(context_messages) > tail_count else list(context_messages)
            messages_for_extraction = recent_msgs + [{"role": "assistant", "content": assistant_msg}]
        else:
            messages_for_extraction = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        
        new_memories = await extract_memories(messages_for_extraction, existing_memories=existing_contents)
        
        META_BLACKLIST = [
            "记忆库", "记忆系统", "检索", "没有被记录", "没有被提取",
            "记忆遗漏", "尚未被记录", "写入不完整", "检索功能",
            "系统没有返回", "关键词匹配", "语义匹配", "语义检索",
            "阈值", "数据库", "seed", "导入", "部署",
            "bug", "debug", "端口", "网关",
        ]
        filtered = []
        for mem in new_memories:
            if any(kw in mem["content"] for kw in META_BLACKLIST):
                print(f"🚫 过滤meta记忆: {mem['content'][:60]}...")
                continue
            filtered.append(mem)
        
        for mem in filtered:
            await save_memory(
                content=mem["content"],
                importance=mem["importance"],
                source_session=session_id,
            )
        
        if filtered:
            total = await get_all_memories_count()
            print(f"💾 提取模型保存了 {len(filtered)} 条新记忆，总计 {total} 条")
    except Exception as e:
        print(f"⚠️  后台记忆处理失败: {e}")

@app.get("/")
async def health_check():
    memory_count = 0
    if MEMORY_ENABLED:
        try:
            memory_count = await get_all_memories_count()
        except:
            pass
    return {
        "status": "running",
        "memory_enabled": MEMORY_ENABLED,
        "memory_count": memory_count,
    }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": DEFAULT_MODEL, "object": "model", "created": 1700000000, "owned_by": "ai-memory-gateway"}]}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API_KEY 未设置"})
    
    body = await request.json()
    messages = body.get("messages", [])
    
    user_message = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                user_message = content
            elif isinstance(content, list):
                user_message = " ".join(item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text")
            break
    
    original_messages = [msg for msg in messages if msg.get("role") != "system"]
    
    if SYSTEM_PROMPT or (MEMORY_ENABLED and user_message):
        if MEMORY_ENABLED and user_message:
            enhanced_prompt = await build_system_prompt_with_memories(user_message)
        else:
            enhanced_prompt = SYSTEM_PROMPT
        
        if enhanced_prompt:
            has_system = any(msg.get("role") == "system" for msg in messages)
            if has_system:
                for i, msg in enumerate(messages):
                    if msg.get("role") == "system":
                        messages[i]["content"] = enhanced_prompt + "\n\n" + msg["content"]
                        break
            else:
                messages.insert(0, {"role": "system", "content": enhanced_prompt})
    
    # 调试打印
    for i, msg in enumerate(messages):
        if msg.get("role") == "system":
            print(f"📌 最终 system prompt (索引{i}): 长度={len(msg['content'])}, 前200字符={msg['content'][:200]}")
            break
    else:
        print("⚠️ 警告：最终 messages 中没有 system 角色！")
    # ===== 重建 messages（关键修复）=====
    original_chat_history = [msg for msg in messages if msg.get("role") != "system"]

    enhanced_prompt = SYSTEM_PROMPT
    if MEMORY_ENABLED and user_message:
    enhanced_prompt = await build_system_prompt_with_memories(user_message)

    new_messages = []

    if enhanced_prompt:
    new_messages.append({
    "role": "system",
    "content": enhanced_prompt
    })

    new_messages.extend(original_chat_history)

    messages = new_messages
    # ===== 修复结束 =====

    body["messages"] = messages
    model = body.get("model", DEFAULT_MODEL)
    body["model"] = model
    session_id = body.get("session_id") or "default"
    
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    if "openrouter" in API_BASE_URL:
        headers["HTTP-Referer"] = EXTRA_REFERER
        headers["X-Title"] = EXTRA_TITLE
    
    is_stream = body.get("stream", False)
    if FORCE_STREAM and not is_stream:
        is_stream = True
        body["stream"] = True
    
    if REASONING_EFFORT:
        body.pop("reasoning_effort", None)
        body.pop("google", None)
        body["reasoning_effort"] = REASONING_EFFORT
    
    print(f"📡 请求: model={model}, stream={is_stream}, memory={'on' if MEMORY_ENABLED else 'off'}")
    
    if is_stream:
        return StreamingResponse(
            stream_and_capture(headers, body, session_id, user_message, model, original_messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    else:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(API_BASE_URL, headers=headers, json=body)
            if response.status_code == 200:
                resp_data = response.json()
                assistant_msg = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                if MEMORY_ENABLED and user_message and assistant_msg:
                    asyncio.create_task(process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages))
                return JSONResponse(content=resp_data)
            else:
                return JSONResponse(status_code=response.status_code, content=response.json())

async def stream_and_capture(headers, body, session_id, user_message, model, original_messages):
    full_response = []
    line_buffer = ""
    async with httpx.AsyncClient(timeout=300) as client:
        async with client.stream("POST", API_BASE_URL, headers=headers, json=body) as response:
            async for chunk in response.aiter_bytes():
                yield chunk
                text = chunk.decode("utf-8", errors="ignore")
                line_buffer += text
                while "\n" in line_buffer:
                    line, line_buffer = line_buffer.split("\n", 1)
                    line = line.strip()
                    if line.startswith("data: ") and line != "data: [DONE]":
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response.append(content)
                        except:
                            pass
    assistant_msg = "".join(full_response)
    if MEMORY_ENABLED and user_message and assistant_msg:
        asyncio.create_task(process_memories_background(session_id, user_message, assistant_msg, model, context_messages=original_messages))

# 管理接口
@app.get("/import/seed-memories")
async def import_seed_memories():
    try:
        from seed_memories import run_seed_import
        return await run_seed_import()
    except ImportError:
        return {"error": "未找到 seed_memories.py"}

@app.get("/export/memories")
async def export_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    memories = await get_all_memories()
    for mem in memories:
        if mem.get("created_at"):
            mem["created_at"] = str(mem["created_at"])
    return {"total": len(memories), "exported_at": str(datetime.now()), "memories": memories}

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    if not MEMORY_ENABLED:
        return HTMLResponse("<h3>记忆系统未启用</h3>")
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/api/memories")
async def api_get_memories():
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    memories = await get_all_memories_detail()
    tz_offset = timezone(timedelta(hours=TIMEZONE_HOURS))
    for m in memories:
        if m.get("created_at"):
            dt = m["created_at"]
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            m["created_at"] = dt.astimezone(tz_offset).strftime("%Y-%m-%d %H:%M:%S")
    return {"memories": memories}

@app.put("/api/memories/{memory_id}")
async def api_update_memory(memory_id: int, request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    await update_memory(memory_id, content=data.get("content"), importance=data.get("importance"))
    return {"status": "ok"}

@app.delete("/api/memories/{memory_id}")
async def api_delete_memory(memory_id: int):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    await delete_memory(memory_id)
    return {"status": "ok"}

@app.post("/api/memories/batch-update")
async def api_batch_update(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    for item in data.get("updates", []):
        await update_memory(item["id"], content=item.get("content"), importance=item.get("importance"))
    return {"status": "ok"}

@app.post("/api/memories/batch-delete")
async def api_batch_delete(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    ids = (await request.json()).get("ids", [])
    if not ids:
        return {"error": "未选择记忆"}
    await delete_memories_batch(ids)
    return {"status": "ok"}

@app.post("/import/text")
async def import_text_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    lines = data.get("lines", [])
    skip_scoring = data.get("skip_scoring", False)
    if not lines:
        return {"error": "没有找到记忆条目"}
    if skip_scoring:
        scored = [{"content": t, "importance": 5} for t in lines]
    else:
        scored = await score_memories(lines)
    imported = 0
    skipped = 0
    for mem in scored:
        content = mem.get("content", "")
        if not content:
            continue
        pool = await get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
        if existing > 0:
            skipped += 1
            continue
        await save_memory(content=content, importance=mem.get("importance", 5), source_session="text-import")
        imported += 1
    total = await get_all_memories_count()
    return {"status": "done", "imported": imported, "skipped": skipped, "total": total}

@app.post("/import/memories")
async def import_memories(request: Request):
    if not MEMORY_ENABLED:
        return {"error": "记忆系统未启用"}
    data = await request.json()
    memories = data.get("memories", [])
    if not memories:
        return {"error": "没有找到记忆数据"}
    imported = 0
    skipped = 0
    for mem in memories:
        content = mem.get("content", "")
        if not content:
            continue
        pool = await get_pool()
        async with pool.acquire() as conn:
            existing = await conn.fetchval("SELECT COUNT(*) FROM memories WHERE content = $1", content)
        if existing > 0:
            skipped += 1
            continue
        await save_memory(content=content, importance=mem.get("importance", 5), source_session=mem.get("source_session", "json-import"))
        imported += 1
    total = await get_all_memories_count()
    return {"status": "done", "imported": imported, "skipped": skipped, "total": total}

if __name__ == "__main__":
    import uvicorn
    print(f"🚀 AI Memory Gateway 启动中... 端口 {PORT}")
    print(f"🧠 记忆系统：{'开启' if MEMORY_ENABLED else '关闭'}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)