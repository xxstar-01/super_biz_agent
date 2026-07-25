# 多层记忆系统 (Multi-Layer Memory System)

> 将原始的 MemorySaver (纯内存) 升级为四层持久化记忆架构

---

## 1. 架构概览

```
┌──────────────────────────────────────────────────────┐
│  L4  语义记忆 (Semantic Memory)                       │
│  跨任务的抽象经验、最佳实践                              │
│  存储: Milvus 向量库                                   │
│  ★ 已有: retrieve_knowledge (只读查询)                  │
│  ★ 后续: 自动从成功任务中提取经验写入                     │
├──────────────────────────────────────────────────────┤
│  L3  情景记忆 (Episodic Memory)            ★ 新增      │
│  每次任务的完整结构化记录                                │
│  存储: SQLite 表 episodes                              │
│  (输入, 计划, 每步结果, 最终响应, 耗时, 状态)             │
├──────────────────────────────────────────────────────┤
│  L2  会话记忆 (Session Memory)             ★ 改动      │
│  MemorySaver → SqliteSaver (自研)                      │
│  进程重启后会话状态不丢失                                │
│  支持 7 天自动过期清理                                  │
├──────────────────────────────────────────────────────┤
│  L1  工作记忆 (Working Memory)             不动        │
│  LangGraph State 在当前任务图中流转                     │
│  PlanExecuteState / AgentState                         │
└──────────────────────────────────────────────────────┘
```

### 数据流

```
新任务 → Planner 检索 L3+L4 经验 → 制定计划
       → 执行 (L2 SqliteSaver 自动持久化每个步骤)
       → 完成 → 写入 L3 情景记录
             → (后续) 提取经验 → 写入 L4 语义记忆
```

---

## 2. 新增文件

### 2.1 `app/memory/__init__.py` — 模块入口 + 全局工厂

```python
from app.memory import get_memory_manager

# 获取全局单例 (延迟初始化, 首次调用时根据 config 创建)
manager = get_memory_manager()
```

**设计要点:**
- `get_memory_manager()` 是全局工厂函数, 延迟初始化
- 首次调用时读取 `config.memory_db_path` 等配置创建实例
- 两个服务 (`RagAgentService`, `AIOpsService`) 共享同一个 MemoryManager 实例

---

### 2.2 `app/memory/sqlite_saver.py` — L2 会话记忆

**核心类: `SqliteSaver(BaseCheckpointSaver[str])`**

自研的 SQLite 持久化 CheckpointSaver, 完整实现 `BaseCheckpointSaver` 接口。

**为什么自研?** 当前安装的 langgraph 版本不含 `SqliteSaver`, 只有 `InMemorySaver`。

**数据库表结构 (3 张表):**

```sql
-- 1. checkpoints: 存储 checkpoint 主数据
checkpoints (
    thread_id, checkpoint_ns, checkpoint_id,
    parent_checkpoint_id,     -- 父 checkpoint, 支持时间旅行
    checkpoint_type,          -- 序列化类型标记 ("json")
    checkpoint_data BLOB,     -- 序列化后的 checkpoint (不含 channel_values)
    metadata_type, metadata_data BLOB,
    created_at
)

-- 2. checkpoint_blobs: 存储 channel 值 (每个 channel 单独存)
checkpoint_blobs (
    thread_id, checkpoint_ns, channel, version,
    blob_type, blob_data BLOB
)

-- 3. checkpoint_writes: 存储中间写入 (tool calls 等)
checkpoint_writes (
    thread_id, checkpoint_ns, checkpoint_id,
    task_id, idx, channel,
    value_type, value_data BLOB, task_path
)

-- 4. session_meta: 会话元数据 (用于过期清理)
session_meta (thread_id, last_active_at)
```

**关键方法:**

| 方法 | 说明 |
|------|------|
| `put(config, checkpoint, metadata, new_versions)` | 保存 checkpoint, 同步更新 session_meta |
| `get_tuple(config)` | 获取 checkpoint + channel_values + writes |
| `put_writes(config, writes, task_id, task_path)` | 保存中间写入 |
| `delete_thread(thread_id)` | 删除会话所有数据 |
| `list(config, filter, before, limit)` | 列出 checkpoints |
| `cleanup_expired_sessions(max_age_days=7)` | 清理过期会话 |
| `get_stats()` | 获取存储统计 |

**线程安全设计:**
- 同步方法使用 `threading.local()` 为每个线程维护独立 `sqlite3.Connection`
- 异步方法通过 `loop.run_in_executor()` 将同步操作跑在线程池中, 避免阻塞事件循环
- SQLite WAL 模式 + `check_same_thread=False` 支持多线程读

**序列化:**
- 使用 `JsonPlusSerializer` (langgraph 默认), 与 `InMemorySaver` 一致
- 数据以 `(type_name: str, bytes)` 元组形式存储

---

### 2.3 `app/memory/episode_store.py` — L3 情景记忆

**核心类: `EpisodeStore`**

记录每次 Agent 任务的完整执行历史。

**数据库表:**

```sql
episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id,                -- 关联的会话 ID
    task_input,                -- 原始任务描述
    task_type,                 -- 任务类型: "aiops" | "chat" | "general"
    plan_json,                 -- JSON: 执行计划 ["步骤1", "步骤2", ...]
    past_steps_json,           -- JSON: [[step, result], [step, result], ...]
    response,                  -- 最终响应文本
    status,                    -- pending | running | completed | failed
    error_message,             -- 失败时的错误信息
    started_at, completed_at,  -- 时间戳
    duration_ms                -- 执行耗时 (毫秒)
)
```

**关键方法:**

| 方法 | 说明 |
|------|------|
| `start_episode(session_id, task_input, task_type)` | 开始记录, 返回 episode_id |
| `update_plan(episode_id, plan)` | 更新执行计划 |
| `update_past_steps(episode_id, past_steps)` | 更新已执行步骤 |
| `complete_episode(episode_id, plan, past_steps, response, status)` | 标记完成, 计算耗时 |
| `get_recent_episodes(limit, task_type, status)` | 查询最近任务 |
| `get_similar_episodes(task_input, limit)` | 关键词匹配相似历史任务 |
| `get_session_episodes(session_id)` | 获取某会话的所有任务 |
| `format_experience_context(task_input, limit)` | 将历史任务格式化为 Planner 可用的经验文本 |
| `get_stats()` | 统计信息: 总数/成功率/平均耗时 |

**相似搜索策略 (当前):**
- 简单关键词匹配 (LIKE %keyword%)
- 后续可升级为: Milvus 向量检索 (将 task_input 向量化后做语义相似度搜索)

---

### 2.4 `app/memory/memory_manager.py` — 统一管理层

**核心类: `MemoryManager`**

协调 L2/L3/L4, 对外提供统一接口。

```python
class MemoryManager:
    def __init__(self, db_path, session_expiry_days=7, enable_episodic=True):
        self.session_store = SqliteSaver(db_path)    # L2
        self.episode_store = EpisodeStore(db_path)    # L3

    # Episode 生命周期
    def start_episode(session_id, task_input, task_type) -> int | None
    def update_episode_plan(session_id, plan)
    def update_episode_steps(session_id, past_steps)
    def complete_episode(session_id, plan, past_steps, response, status, error_message)

    # 经验检索 (供 Planner)
    def get_experience_context(task_input, limit=3) -> str

    # 会话清理
    def cleanup_expired_sessions() -> int

    # 统计
    def get_stats() -> dict

    # 资源管理
    def close() / aclose()
```

---

## 3. 修改的文件

### 3.1 `app/config.py`

新增 3 个配置项:

```python
# 多层记忆配置
memory_db_path: str = "data/memory.db"          # SQLite 文件路径
memory_session_expiry_days: int = 7             # 会话过期天数
memory_episodic_enabled: bool = True            # 是否启用 L3 情景记忆
```

可通过 `.env` 文件覆盖:
```env
MEMORY_DB_PATH=data/memory.db
MEMORY_SESSION_EXPIRY_DAYS=14
MEMORY_EPISODIC_ENABLED=true
```

### 3.2 `app/services/rag_agent_service.py`

**变更 1: 替换 checkpointer**

```python
# 之前
from langgraph.checkpoint.memory import MemorySaver
self.checkpointer = MemorySaver()

# 之后
from app.memory import get_memory_manager
self.memory_manager = get_memory_manager()
self.checkpointer = self.memory_manager.session_store  # SqliteSaver
```

**变更 2: 添加 L3 记录** — `query()` 和 `query_stream()` 方法中:

```python
# 任务开始
self.memory_manager.start_episode(session_id=session_id, task_input=question, task_type="chat")

# 任务完成
self.memory_manager.complete_episode(
    session_id=session_id, plan=[], past_steps=[],
    response=answer, status="completed",
)

# 任务失败
self.memory_manager.complete_episode(
    session_id=session_id, plan=[], past_steps=[],
    response="", status="failed", error_message=str(e),
)
```

### 3.3 `app/services/aiops_service.py`

**变更 1: 延迟初始化**

```python
# 之前
class AIOpsService:
    def __init__(self):
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

# 之后
class AIOpsService:
    def __init__(self):
        self._checkpointer = None  # 延迟初始化
        self._graph = None
        self._memory_manager = None

    @property
    def memory_manager(self):
        if self._memory_manager is None:
            self._memory_manager = get_memory_manager()
        return self._memory_manager

    @property
    def checkpointer(self):
        if self._checkpointer is None:
            self._checkpointer = self.memory_manager.session_store
        return self._checkpointer

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self._build_graph()
        return self._graph
```

**为什么延迟?** `SqliteSaver` 需要 config 中的 db_path, 比 `MemorySaver()` 多一个依赖。延迟到首次使用时创建, 避免模块导入顺序问题。

**变更 2: execute() 中添加 L3 记录**

```python
async def execute(self, user_input, session_id):
    # 1. 开始 episode
    self.memory_manager.start_episode(session_id, user_input, task_type="aiops")

    original_plan = []  # 捕获 Planner 输出的初始计划

    async for event in self.graph.astream(...):
        for node_name, node_output in event.items():
            if node_name == NODE_PLANNER:
                original_plan = list(node_output.get("plan", []))
                self.memory_manager.update_episode_plan(session_id, original_plan)
            ...

    # 2. 获取最终 accumulated state (past_steps 通过 operator.add 累积)
    final_state = self.graph.get_state(config_dict)

    # 3. 完成 episode
    self.memory_manager.complete_episode(
        session_id, original_plan, final_past_steps, final_response,
        status="completed" | "failed",
    )
```

**关键设计: 捕获初始计划**

- `past_steps` 使用 `operator.add` 自动累积, 最终 `graph.get_state()` 能获取全部
- 但 `plan` 是逐步骤消费的 (`plan[1:]`), 最终为空
- 所以在 Planner 事件中捕获 `original_plan`, 用于 episode 记录

### 3.4 `app/agent/aiops/planner.py`

**变更: 注入 L3 情景记忆经验**

```python
# 之前
experience_docs = await retrieve_knowledge.ainvoke(...)      # L4 Milvus
experience_context = format_l4_experience(experience_docs)

# 之后
experience_docs = await retrieve_knowledge.ainvoke(...)      # L4 Milvus
l3_experience = memory_manager.get_experience_context(text)  # L3 EpisodeStore ★新增

# 合并 L4 + L3 经验
experience_context = merge_experience(experience_docs, l3_experience)
```

注入到 Planner prompt 的效果:
```
## 知识库经验 (L4 语义记忆)
以下是从知识库中检索到的相关经验和最佳实践...

---
## 历史相似任务经验 (L3 情景记忆)

### 经验 1: 诊断当前系统是否存在告警
- 执行计划: 获取活动告警 → 查询监控指标 → 分析日志 → 生成报告
- 状态: completed
- 耗时: 45秒
```

---

## 4. 使用方式

### 4.1 基本使用 — 自动生效

服务启动后自动使用 SQLite 持久化, 无需额外代码:

```python
# 服务启动 (已有代码不变)
# app/services/aiops_service.py 的模块级单例
aiops_service = AIOpsService()

# 调用 (session 状态自动持久化到 data/memory.db)
async for event in aiops_service.execute("分析系统性能", session_id="sess-001"):
    ...
```

### 4.2 查询历史

```python
from app.memory import get_memory_manager

manager = get_memory_manager()

# L2: 查看会话 checkpoint
state = manager.session_store.get_tuple(
    {"configurable": {"thread_id": "sess-001"}}
)

# L3: 查看最近任务
episodes = manager.episode_store.get_recent_episodes(limit=10, task_type="aiops")
for ep in episodes:
    print(f"{ep.started_at} | {ep.task_input[:50]} | {ep.status} | {ep.duration_ms}ms")

# L3: 查看统计
stats = manager.get_stats()
# => {"session": {...}, "episode": {"total": 42, "success_rate": 95.2, ...}}
```

### 4.3 会话过期清理

```python
# 手动清理 (删除 7 天前的会话)
manager.cleanup_expired_sessions()

# 定时任务示例 (FastAPI lifespan)
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # 启动时清理一次
    get_memory_manager().cleanup_expired_sessions()
    yield
    # 关闭时清理资源
    get_memory_manager().close()
```

---

## 5. 数据文件

```
data/
  memory.db          # 主数据库 (L2 checkpoint + L3 episode)
  memory.db-shm      # SQLite WAL 共享内存
  memory.db-wal      # SQLite WAL 日志
```

数据库位置由 `config.memory_db_path` 控制, 默认 `data/memory.db`。

---

## 6. 扩展指南

### 6.1 ✅ L3 向量检索 (已实现)

**实现方式:** 双通道检索——优先 Milvus 向量, 不可用时自动回退关键词。

**工作流程:**

```
get_similar_episodes(task_input)
  ├─ Milvus 可用?
  │   ├─ Yes → embed_query() → collection.search(type="episode") → SQLite lookup
  │   └─ No  → 回退到 _keyword_search_similar() (SQL LIKE)
  └─ 按向量距离排序返回
```

**关键代码路径:**

| 方法 | 文件 | 功能 |
|------|------|------|
| `_index_episode_to_milvus()` | `app/memory/episode_store.py` | 完成后将 task_input 向量化写入 Milvus |
| `_vector_search_similar()` | 同上 | 向量相似度搜索 + SQLite 联合查询 |
| `_keyword_search_similar()` | 同上 | SQL LIKE 回退方案 |
| `_get_embedding_service()` | 同上 | 延迟导入 `vector_embedding_service` |
| `_get_milvus_collection()` | 同上 | 延迟导入 `milvus_manager`, 避免启动连接 |

**Milvus 存储格式:**
- `id`: `"ep_<episode_id>"` (如 `"ep_42"`)
- `vector`: `DashScopeEmbeddings.embed_query(task_input)` 产物, 1024 维
- `content`: `task_input[:8000]`
- `metadata`: `{"type": "episode", "episode_id": 42}`

**搜索时:** 用 `expr=f'metadata["type"] == "episode"'` 过滤, 排除文档知识库。

**索引时机:** `complete_episode()` 中同步执行 (写 SQLite + 写 Milvus 在同一线程), 失败不阻塞。

**配置:** `memory_vector_search_enabled: bool = True` — 设为 `False` 全程使用关键词, 不依赖 Milvus。

---

### 6.2 ✅ L4 语义记忆自动写入 (已实现)

**新增文件:** `app/memory/semantic_extractor.py`

**核心类:** `SemanticExtractor`

从成功完成的任务中, 用 LLM 提取可复用的经验, 向量化后写入 Milvus。

**工作流程:**

```
complete_episode(status="completed")
  └─ self.semantic_extractor.extract_and_store_async(...)
       └─ daemon thread: _extract_and_store()
            ├─ 格式化 task_input + plan + past_steps + response
            ├─ LLM 调用 (qwen-plus, temperature=0.3)
            │    ├─ 输出 JSON: {title, problem, root_cause, solution, how_to_apply, tags}
            │    └─ 无价值经验 → 输出 null → 跳过
            ├─ 构建 searchable_text: title + problem + solution + how_to_apply
            ├─ embed_query(searchable_text)
            └─ Milvus insert: id="l4exp_<episode_id>", type="semantic_experience"
```

**LLM 提取 Prompt 结构:**

```
你是一个经验总结专家。请根据以下任务执行记录, 提取可复用的经验和最佳实践。

## 原始任务 (task_input)
## 执行计划 (plan)
## 执行步骤与结果 (past_steps)
## 最终响应 (response[:2000])

JSON 输出: {title, problem, root_cause, solution, how_to_apply, tags}
```

**Milvus 存储格式:**
- `id`: `"l4exp_<episode_id>"` (如 `"l4exp_42"`)
- `vector`: `embed_query(searchable_text)` 产物
- `content`: 聚合后的可检索文本 (title + problem + solution + how_to_apply)
- `metadata`: `{"type": "semantic_experience", "source_episode_id": 42, "title": "...", "tags": [...], ...}`

**非阻塞设计:**
- `extract_and_store_async()` → 启动 daemon 线程, 立即返回
- 提取失败 → 静默跳过, 不影响 API 响应
- 线程名: `l4-extract-ep<id>`, 方便调试

**Planner 集成:**

`MemoryManager.get_experience_context()` 现在返回 **L3 + L4 合并**:

```
## 历史相似任务经验 (L3 情景记忆)
...

---
## 语义经验 (L4 知识库)

### 💡 CPU高负载诊断方法
- **问题**: 服务响应延迟高, CPU使用率持续高于90%
- **方案**: 先查Prometheus监控确认CPU趋势, 再用CLS日志定位top进程...
- **标签**: CPU, 性能分析, OOM
```

**搜索时:** 用 `expr=f'metadata["type"] == "semantic_experience"'` 过滤, 与 episode 向量和文档向量隔离。

**配置项:**
- `memory_semantic_enabled: bool = True` — 是否启用 L4 自动提取
- `memory_semantic_model: str = "qwen-plus"` — 提取用的 LLM (推荐 cheaper model)

**全局单例:** `get_semantic_extractor(model_name)` → `MemoryManager` 的 `semantic_extractor` 属性延迟获取

---

### 6.3 升级到 Postgres

当前使用 SQLite 单文件。如需分布式部署:

- L2: 使用 `langgraph-checkpoint-postgres` 包中的 `PostgresSaver`
- L3: 将 `EpisodeStore` 的 SQL 改为 PostgreSQL (用 asyncpg)
- `MemoryManager` 接口不变, 底层实现可替换

### 6.4 添加记忆统计 API

```python
# app/api/memory.py (新增)
@router.get("/memory/stats")
async def memory_stats():
    return get_memory_manager().get_stats()

@router.get("/memory/episodes")
async def list_episodes(limit: int = 10, task_type: str = None):
    store = get_memory_manager().episode_store
    return [e.to_dict() for e in store.get_recent_episodes(limit, task_type)]
```

---

## 7. 变更文件清单

### 第一轮 (L2 + L3 基础)

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/memory/__init__.py` | **新增** | 模块入口 + `get_memory_manager()` 工厂 |
| `app/memory/sqlite_saver.py` | **新增** | L2 会话记忆: SQLite 持久化 CheckpointSaver |
| `app/memory/episode_store.py` | **新增** | L3 情景记忆: 任务历史记录 |
| `app/memory/memory_manager.py` | **新增** | 统一管理: 协调 L2/L3/L4 |
| `app/config.py` | **修改** | 新增 `memory_db_path` 等 3 个配置项 |
| `app/services/rag_agent_service.py` | **修改** | MemorySaver → SqliteSaver + L3 记录 |
| `app/services/aiops_service.py` | **修改** | MemorySaver → SqliteSaver + 延迟初始化 + L3 记录 |
| `app/agent/aiops/planner.py` | **修改** | 注入 L3 历史经验到 Planner prompt |

### 第二轮 (6.1 向量检索 + 6.2 语义记忆自动写入)

| 文件 | 操作 | 说明 |
|------|------|------|
| `app/memory/semantic_extractor.py` | **新增** | L4 语义提取器: LLM 提取经验 → Milvus |
| `app/memory/episode_store.py` | **修改** | 新增向量检索 `_vector_search_similar()` + Milvus 索引 `_index_episode_to_milvus()` + 关键词回退 |
| `app/memory/memory_manager.py` | **修改** | 新增 L4 提取触发 + `_get_semantic_context()` + Planner 合并 L3+L4 |
| `app/memory/__init__.py` | **修改** | 导出 `SemanticExtractor` + 工厂传递新配置 |
| `app/config.py` | **修改** | 新增 `memory_vector_search_enabled`, `memory_semantic_enabled`, `memory_semantic_model`
