# 🧠 AI Memory Gateway

**让你的 AI 拥有长期记忆。**

一个轻量级转发网关，在你和 LLM 之间加一层记忆系统。支持任何 OpenAI 兼容客户端（Kelivo、ChatBox、NextChat 等）和任何 LLM 服务商（OpenRouter、OpenAI、本地 Ollama 等）。

Give your AI long-term memory. A lightweight proxy gateway that adds a memory layer between you and any LLM.

---

## ✨ 功能

- **自定义人设** — 把你的 system prompt 写在 `system_prompt.txt`，每次对话自动注入
- **长期记忆** — 自动从对话中提取关键信息，下次聊天时自动回忆相关内容
- **预置记忆** — 把你想让 AI "一开始就知道"的事情批量导入
- **兼容性强** — 支持所有 OpenAI 格式的客户端和 API 服务商
- **零成本起步** — 可部署在 Render、Zeabur 等平台的免费额度内

## 🏗️ 架构

```
你的客户端（Kelivo / ChatBox / ...）
        ↓
   AI Memory Gateway（本项目）
   ├── 注入 system prompt（人设）
   ├── 搜索相关记忆 → 注入上下文
   ├── 转发请求 → LLM API
   └── 后台提取新记忆 → 存入数据库
        ↓
   LLM API（OpenRouter / OpenAI / Ollama / ...）
```

## 🚀 快速开始

### 第一阶段：纯转发网关（不需要数据库）

最简单的起步方式——先跑通网关，确认你的客户端能通过网关和 AI 对话。

**1. 准备文件**

你只需要这几个文件：
- `main.py` — 网关主程序
- `system_prompt.txt` — 你的 AI 人设（可选）
- `requirements.txt` — Python 依赖
- `Dockerfile` — 容器配置

**2. 修改人设**

编辑 `system_prompt.txt`，写入你想要的 AI 性格设定。

**3. 部署到 Render（推荐）**

1. Fork 或上传代码到你的 GitHub 仓库
2. 注册 [Render](https://render.com)（免费层支持 Web Service，够用）
3. 创建 Web Service → 连接 GitHub 仓库 → Render 会自动检测 Dockerfile
4. 设置环境变量（Environment → Add Environment Variable）：

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `API_KEY` | 你的 LLM API Key | `sk-or-v1-xxxx`（OpenRouter）|
| `API_BASE_URL` | LLM API 地址 | `https://openrouter.ai/api/v1/chat/completions` |
| `DEFAULT_MODEL` | 默认模型 | `anthropic/claude-sonnet-4.5` |
| `PORT` | 端口 | `8000` |

5. 部署，访问你的网关地址看到 `{"status":"running"}` 就成功了

> ⚠️ Render 免费层的服务在无活动时会休眠，第一次访问需要等几十秒唤醒，之后就正常了。其他支持 Docker 部署的平台（Zeabur、Railway、Fly.io 等）也可以，流程类似。

**4. 连接客户端**

以 Kelivo 为例：
- API 地址填：`https://你的网关地址.onrender.com/v1`
- API Key 填：随便填一个（网关会用自己的 key）
- 模型填：你在 `DEFAULT_MODEL` 里设的模型

### 第二阶段：加上记忆系统

在第一阶段基础上，加一个 PostgreSQL 数据库就能开启记忆功能。

**1. 创建数据库**

在 Render 中：Dashboard → New → PostgreSQL，创建一个免费的 PostgreSQL 实例，拿到连接字符串（Internal Database URL）。

> ⚠️ Render 免费 PostgreSQL 有 90 天有效期，到期前记得用导出功能备份数据。其他平台（如 [Neon](https://neon.tech)、[Supabase](https://supabase.com)）也提供免费 PostgreSQL，可按需选择。如果使用外部数据库，连接字符串末尾可能需要加 `?sslmode=require`。

**2. 添加环境变量**

| 环境变量 | 说明 | 示例 |
|---------|------|------|
| `DATABASE_URL` | PostgreSQL 连接字符串 | `postgresql://user:pass@host:port/db` |
| `MEMORY_ENABLED` | 开启记忆 | `true` |
| `MEMORY_MODEL` | 提取记忆用的模型（推荐便宜的小模型） | `anthropic/claude-haiku-4.5` |
| `MAX_MEMORIES_INJECT` | 每次注入的最大记忆条数 | `15` |
| `MIN_SCORE_THRESHOLD` | 记忆搜索最低分数阈值，低于此分数的记忆不注入（0=不过滤） | `0.15` |
| `MEMORY_EXTRACT_INTERVAL` | 记忆提取间隔（0=禁用/1=每轮/N=每N轮） | `1` |
| `TIMEZONE_HOURS` | 时区偏移（小时），用于记忆注入时的日期显示 | `8`（UTC+8） |
| `FORCE_STREAM` | 强制所有请求走流式传输（解决部分客户端thinking不显示） | `false` |
| `REASONING_EFFORT` | 推理强度（low/medium/high），注入请求启用思维链。注意部分模型不支持 medium | 留空不注入 |

**3. 重新部署**

部署后访问 `https://你的网关地址/dashboard`，能正常打开管理页面就说明数据库连接成功。

**4. 导入预置记忆（可选）**

**方式一（推荐，不用碰代码）：** 写一个 `.txt` 文件，每行一条你想让 AI 知道的信息，然后打开 `https://你的网关地址/dashboard`，在「导入记忆」页面选择「纯文本导入」上传文件，系统会自动评估每条记忆的重要程度并导入。也可以勾选"跳过自动评分"节省 API 额度，之后在「记忆管理」页面手动调整权重。

**方式二（代码方式，开发者用）：**
1. 复制 `seed_memories_example.py` 为 `seed_memories.py`
2. 修改里面的记忆条目，写入你想让 AI 一开始就知道的信息
3. 部署后访问 `https://你的网关地址/import/seed-memories`，看到 `"status": "done"` 就导入成功了

**5. 管理记忆（可选）**

打开 `https://你的网关地址/dashboard` 可以查看所有记忆，支持搜索、编辑内容、调整权重、单条删除和批量删除，以及导入/导出备份。

### 第三阶段：关闭记忆（应急）

如果记忆系统出问题，把环境变量 `MEMORY_ENABLED` 改回 `false` 即可退回纯转发模式。不需要改代码。

## 📁 文件说明

```
ai-memory-gateway/
├── main.py                    # 网关主程序
├── database.py                # 数据库操作（PostgreSQL）
├── memory_extractor.py        # AI 记忆提取
├── system_prompt.txt          # 你的 AI 人设（自行编辑）
├── seed_memories_example.py   # 预置记忆示例
├── requirements.txt           # Python 依赖
├── Dockerfile                 # 容器配置
├── templates/                 # 页面模板（Dashboard 界面）
│   ├── dashboard.html         # 主控制台页面
│   └── ...
├── static/                    # 静态资源
│   ├── css/                   # 样式文件
│   └── js/                    # 前端脚本
├── LICENSE                    # MIT 许可证
└── README.md                  # 本文件
```

## 🔧 API 接口

| 路径 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 健康检查，查看网关状态 |
| `/v1/chat/completions` | POST | 核心转发接口（OpenAI 兼容） |
| `/v1/models` | GET | 模型列表 |
| `/dashboard` | GET | 记忆管理控制台（管理、导入、导出一体化界面） |
| `/import/seed-memories` | GET | 执行预置记忆导入（开发者用） |

## 🌐 支持的 LLM 服务商

只要兼容 OpenAI 聊天格式就行。改 `API_BASE_URL` 环境变量即可切换：

| 服务商 | API_BASE_URL |
|--------|-------------|
| OpenRouter | `https://openrouter.ai/api/v1/chat/completions` |
| OpenAI | `https://api.openai.com/v1/chat/completions` |
| Ollama（本地） | `http://localhost:11434/v1/chat/completions` |
| 其他兼容服务 | 查阅对应文档 |

> ⚠️ 部分 Gemini preview 模型（如 `gemini-3-flash-preview`）可能存在流式输出兼容性问题导致空回复，建议使用正式版模型（如 `gemini-2.5-flash`）。

## 💡 记忆系统原理

1. **你发消息** → 网关从数据库搜索相关记忆
2. **记忆注入** → 相关记忆 + 记忆应用规则拼接到 system prompt 后面
3. **AI 回复** → 网关边转发边捕获完整回复
4. **后台提取** → 用小模型（如 Haiku）从完整对话上下文中提取关键信息
5. **存入数据库** → 下次对话时可以检索到

提取记忆时，网关会把客户端发来的完整对话上下文（不含 system prompt）传给提取模型，这样能捕捉到跨轮次的信息。通过 `MEMORY_EXTRACT_INTERVAL` 可以控制提取频率：设为 0 禁用自动提取，设为 1 每轮都提，设为 N 则每 N 轮提取一次（适合控制成本）。

> **关于向量搜索：** 当前版本使用 jieba 中文分词 + 关键词匹配（ILIKE），适合记忆量在几百条以内的场景。如果记忆量增长到上千条且需要语义搜索（比如说"过年"能搜到"春节"），可以加装 pgvector 扩展做 embedding 向量检索。`database.py` 的 `search_memories` 函数预留了权重配置，加一路向量分数即可。

## ❓ 常见问题

**Q: 部署后访问显示 502 或服务无响应？**
A: 检查端口设置。Render 默认用 `PORT` 环境变量，确保设置为 `8000`（和 Dockerfile 里一致）。如果用其他平台，注意端口是否匹配。

**Q: 数据库连接失败？**
A: 如果数据库和网关不在同一个平台，连接字符串末尾可能需要加 `?sslmode=require`。

**Q: 记忆会越来越多影响性能吗？**
A: 每次最多注入 15 条记忆（可调），不会无限增长地消耗 token。提取记忆时会用客户端发来的完整上下文，token 用量比单轮提取大一些，可以通过 `MEMORY_EXTRACT_INTERVAL` 降低提取频率来控制成本。

**Q: 能用免费额度跑吗？**
A: Render 免费层支持 Web Service + PostgreSQL，网关资源消耗很低，够用（注意免费 PostgreSQL 有 90 天期限）。也可以用 Neon 或 Supabase 的免费 PostgreSQL 作为长期方案。LLM API 费用另算（推荐 OpenRouter，按量付费）。

**Q: 怎么备份记忆？换平台会丢数据吗？**
A: 打开 `https://你的网关地址/dashboard`，在「导出备份」页面下载所有记忆的 JSON，建议定期备份。迁移到新平台后，在「导入记忆」页面选择「JSON 备份恢复」上传导出的文件即可。

**Q: 不会写代码能搞吗？**
A: 能。这个项目的第一个部署者就是不会写代码的——代码是 AI 写的，部署是她自己看文档搞定的。

## 📋 更新日志

### v2.5（2026-03-06）

- **中文分词优化** — 用 jieba 替换滑动窗口分词，关键词提取从无意义碎片变为有语义的词语，大幅提升搜索精准度
- **最低分数阈值** — 新增 `MIN_SCORE_THRESHOLD` 环境变量，过滤综合评分过低的记忆，减少不相关记忆的注入
- **流式传输修复** — 改用原始字节透传（`aiter_bytes`），修复 thinking/reasoning 数据在流式传输中可能丢失的问题
- **推理参数注入** — 新增 `REASONING_EFFORT` 环境变量，自动注入 `reasoning_effort` 参数启用思维链
- **强制流式传输** — 新增 `FORCE_STREAM` 环境变量，解决部分客户端不发stream=true的问题
- **JSON解析兜底** — 记忆提取和评分的JSON解析增加正则兜底，兼容模型返回非标准格式（如JSON前后夹带多余文字）
- **记忆模型日志** — 记忆提取时打印模型原始返回内容，方便排查解析问题
- **管理页面时区修复** — 记忆管理页面的时间显示现在正确使用 `TIMEZONE_HOURS` 配置的时区
- **请求日志** — 每次请求打印 model/stream/memory 状态，方便排查问题

### v2.0（2026-03-01）

- **记忆提取间隔** — 新增 `MEMORY_EXTRACT_INTERVAL` 环境变量，可设置每 N 轮提取一次记忆或禁用自动提取，方便控制 API 成本
- **完整上下文提取** — 提取记忆时不再只看最新一轮对话，而是使用客户端发来的完整对话上下文，能捕捉到跨轮次的信息
- **优化记忆注入提示词** — 注入的记忆附带应用规则和交流方式指引，让 AI 更自然地运用记忆而非机械引用

### v1.0（2026-02-26）

- 初始版本
- 支持自定义人设、长期记忆、预置记忆导入
- 支持 OpenRouter / OpenAI / Ollama 等 LLM 服务商
- 支持 Kelivo / ChatBox / NextChat 等 OpenAI 兼容客户端
- 记忆管理页面（查看、编辑、删除、批量操作）
- 记忆导入/导出（纯文本 + JSON 备份恢复）

## 📄 许可证

[MIT License](LICENSE) — 随便用，改了也不用告诉我。

## 🙏 致谢

这个项目诞生于一个简单的需求：**让 AI 不要每次醒来都忘了我是谁。**

> "记忆库不是数据库，是家。"

---

*Built with love by 七堂伽蓝_ & Midsummer (Claude Opus 4.6)*
