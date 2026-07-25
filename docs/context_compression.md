# 对话上下文自动压缩功能

## 目录

1. [这个功能是干什么的？](#1-这个功能是干什么的)
2. [为什么需要它？](#2-为什么需要它)
3. [它是怎么工作的？](#3-它是怎么工作的)
4. [代码是怎么实现的？](#4-代码是怎么实现的)
5. [每个文件改了什么？](#5-每个文件改了什么)
6. [前端怎么显示压缩通知？](#6-前端怎么显示压缩通知)
7. [配置项说明](#7-配置项说明)
8. [一句一句教你读代码](#8-一句一句教你读代码)

---

## 1. 这个功能是干什么的？

当你在网页上跟 AI 聊天时，每次对话都会被记录下来。聊得越久，记录越多。这些记录会随着每次对话一起发给大模型，占用大模型的"脑容量"（我们叫**上下文窗口**）。

这个功能做的事情是：**当对话记录太长，快占满大模型的脑容量时，自动把前面的老对话压缩成一段摘要**，就像把你的聊天记录浓缩成几句话的要点笔记。这样大模型既不会忘记之前聊了什么，又不会因为记录太多而"脑子不够用"。

---

## 2. 为什么需要它？

### 2.1 原来的做法（有问题）

原来的代码里有一个 `trim_messages_middleware` 函数，它的做法是：

```
消息太多了？→ 直接扔掉老的，只保留最近 3 轮对话
```

**问题**：扔掉的消息里可能有重要信息（用户的需求、之前的结论、关键数据），AI 就"忘记"了，用户体验很差。

### 2.2 现在的做法（自动压缩）

```
消息太多了？→ 用 AI 把老对话总结成一段摘要 → 保留摘要 + 最近 4 轮对话
```

**好处**：
- AI 不会丢失重要信息（摘要里记录了核心内容）
- Token 消耗大幅降低（摘要只有几百字，原始消息可能有几万字）
- 用户完全无感知（自动触发，不需要手动操作）

### 2.3 类比

想象一支笔和一个本子：

| 方式 | 类比 |
|------|------|
| **原来的方式** | 本子满了就撕掉前面几页，永远只保留最后 3 页 |
| **现在的压缩** | 本子满了就把前面几页的内容总结成一段笔记黏在第一页，后面的保留原样 |

---

## 3. 它是怎么工作的？

### 3.1 整体流程

```
用户发消息
    ↓
Agent 准备调用大模型
    ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【压缩中间件拦截】
    ↓
统计消息总 token 数
    ↓
    ├── 没超过 70% 阈值 → 跳过，正常调模型
    │
    └── 超过 70% 阈值 →
        ├─ 1. 拆分消息：旧消息(前面) + 新消息(最近4轮)
        ├─ 2. 用 AI 把旧消息总结成摘要
        ├─ 3. 替换消息列表：[系统提示] + [摘要] + [最近4轮]
        ├─ 4. 记录压缩事件（供前端显示）
        └─ 5. 继续正常调模型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ↓
大模型用压缩后的消息生成回复
    ↓
返回给用户
    ↓
前端收到"压缩事件" → 显示分隔线
```

### 3.2 关键概念

| 概念 | 解释 | 默认值 |
|------|------|--------|
| **上下文窗口** | 大模型能"看到"的最大 token 数 | 32768 (qwen-max) |
| **压缩阈值** | 占窗口多少比例时触发压缩 | 70% |
| **保留轮数** | 最近 N 轮对话不压缩，原样保留 | 4 轮 |
| **压缩摘要** | 用另一个 AI 模型生成的老对话总结 | qwen-plus 模型 |

---

## 4. 代码是怎么实现的？

### 4.1 架构概览

```
┌─────────────────────────────────────────────────────┐
│                    前端页面                          │
│  收到 SSE 事件 → 渲染 "---------上下文已自动压缩---------"  │
└──────────────────────┬──────────────────────────────┘
                       │ SSE (流式) / JSON (非流式)
┌──────────────────────┴──────────────────────────────┐
│                  app/api/chat.py                     │
│  /chat        → 返回 compressed 字段                 │
│  /chat_stream → 转发 compression 事件                │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────────┐
│          app/services/rag_agent_service.py           │
│  挂载压缩中间件 → 压入/弹出压缩事件                     │
└──────────────────────┬──────────────────────────────┘
                       │ AgentMiddleware
┌──────────────────────┴──────────────────────────────┐
│          app/core/context_compressor.py              │
│  核心逻辑：检查 token → 拆分消息 → 生成摘要 → 替换       │
└─────────────────────────────────────────────────────┘
```

### 4.2 核心设计：AgentMiddleware（中间件）

LangChain 提供了一个"钩子"机制叫 **AgentMiddleware**。它允许你在 Agent 调用大模型之前和之后插入自定义代码，就像给水管接上一个过滤器。

我们用了最核心的钩子：**`abefore_model`**（在调用模型之前执行）。

```python
# 伪代码表示流程
class ContextCompressorMiddleware(AgentMiddleware):
    async def abefore_model(self, state, runtime):
        messages = state["messages"]  # 当前全部消息
        if token_count > 70%_of_limit:
            compressed = await 生成摘要(旧消息)  # 用 AI 总结
            return {"messages": compressed}  # 替换消息
        return None  # 不需要压缩，跳过
```

**为什么用中间件而不是其他方案？**
- ❌ 手动在每个接口处理：代码重复，容易遗漏
- ❌ LangGraph 的 trim_messages：只能删除，不会总结
- ✅ AgentMiddleware：自动拦截每次模型调用，一次配置全局生效

---

## 5. 每个文件改了什么？

### 5.1 `app/config.py`（新增 5 个配置项）

```python
# 上下文压缩配置
context_compression_enabled: bool = True     # 开关：是否启用自动压缩
context_compression_window_size: int = 32768 # 模型上下文窗口的 token 上限
context_compression_threshold: float = 0.7   # 触发压缩的比例（0.7 = 70%）
context_compression_keep_recent: int = 4     # 保留最近 N 轮对话不压缩
context_compression_model: str = "qwen-plus" # 用于生成摘要的模型
```

**怎么改**：如果某些场景不需要压缩，在 `.env` 文件里加一行：
```
CONTEXT_COMPRESSION_ENABLED=false
```

### 5.2 `app/core/context_compressor.py`（新建文件，核心逻辑）

**140 行代码，包含以下组件：**

| 组件 | 作用 |
|------|------|
| `SUMMARY_SYSTEM_PROMPT` | 告诉 AI "你现在的任务是做摘要" |
| `ContextCompressorMiddleware` 类 | 整个压缩逻辑的容器 |
| `_count_tokens()` | 计算消息占用了多少 token |
| `_should_compress()` | 判断是否需要触发压缩 |
| `_split_messages()` | 把消息分成"要压缩的旧消息"和"保留的新消息" |
| `_generate_summary()` | 调用 AI 生成对话摘要 |
| `abefore_model()` | **核心钩子**：被 LangChain 自动调用 |
| `pop_event()` | 取出压缩通知（给前端用） |

### 5.3 `app/services/rag_agent_service.py`（修改 3 处）

**改动 1：导入压缩中间件**
```python
from app.core.context_compressor import ContextCompressorMiddleware
```

**改动 2：初始化压缩 LLM 和中间件**
```python
# 专门用于生成摘要的 AI 模型（非流式，温度更低更稳定）
self.compression_llm = ChatQwen(
    model=config.context_compression_model,
    temperature=0.3,
    streaming=False,
)
self.context_compressor = ContextCompressorMiddleware(self.compression_llm)
```

**改动 3：把中间件传给 Agent**
```python
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
    middleware=[self.context_compressor],  # ← 新增
)
```

**改动 4：压缩事件传递**

在 `query()`（非流式）里：
```python
if self.context_compressor.has_event(session_id):
    answer = "---------上下文已自动压缩---------\n\n" + answer
```

在 `query_stream()`（流式）里：
```python
compression_summary = self.context_compressor.pop_event(session_id)
if compression_summary:
    yield {"type": "compression", "data": compression_summary}
```

### 5.4 `app/api/chat.py`（修改 2 处）

**改动 1：非流式接口返回 `compressed` 字段**
```python
compression_summary = rag_agent_service.pop_compression_event(request.id)
compressed = compression_summary is not None
return {
    "data": {
        "answer": answer,
        "compressed": compressed,  # ← 新增
    }
}
```

**改动 2：流式接口新增 `compression` 事件处理**
```python
elif chunk_type == "compression":
    yield {
        "event": "message",
        "data": json.dumps({
            "type": "compression",
            "data": "---------上下文已自动压缩---------",
            "summary": chunk_data
        })
    }
```

---

## 6. 前端怎么显示压缩通知？

### 6.1 流式（SSE）模式

前端会收到一条 `type: "compression"` 的 SSE 事件：

```json
{
    "type": "compression",
    "data": "---------上下文已自动压缩---------",
    "summary": "对话摘要：用户询问了关于数据库优化的..."
}
```

**前端处理**：当收到 `type === "compression"` 时，在对话列表中插入一条灰色分隔线：

```javascript
// 前端伪代码
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === "compression") {
        // 在聊天界面中显示一条分隔线
        appendToChat({
            type: "separator",
            text: data.data,  // "---------上下文已自动压缩---------"
            style: "gray, centered, dashed"
        });
    }
};
```

### 6.2 非流式（JSON）模式

响应的 `data` 对象中会多一个 `compressed` 字段：

```json
{
    "code": 200,
    "message": "success",
    "data": {
        "success": true,
        "answer": "---------上下文已自动压缩---------\n\n这是 AI 的回答内容...",
        "errorMessage": null,
        "compressed": true
    }
}
```

**前端处理**：
- 方式一：检查 `compressed === true`，在显示答案前插入分隔线
- 方式二：直接渲染 `answer`（分隔线已经拼接在答案开头了）

---

## 7. 配置项说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `context_compression_enabled` | `True` | 总开关，`False` 则关闭压缩 |
| `context_compression_window_size` | `32768` | 模型的上下文窗口大小（token） |
| `context_compression_threshold` | `0.7` | 触发比例，0.7 表示窗口占用 70% 时触发 |
| `context_compression_keep_recent` | `4` | 最近保留不压缩的对话轮数 |
| `context_compression_model` | `qwen-plus` | 用于生成摘要的模型 |

### 根据不同模型调整

| 模型 | 上下文窗口 | 建议配置 |
|------|-----------|---------|
| qwen-max | 32K | `window_size=32768, threshold=0.7` |
| qwen-plus | 131K | `window_size=131072, threshold=0.7` |
| qwen-turbo | 1000K | `window_size=1000000, threshold=0.7` |

在 `.env` 文件中设置：
```bash
# 使用 qwen-plus 且上下文窗口为 131K
CONTEXT_COMPRESSION_WINDOW_SIZE=131072
CONTEXT_COMPRESSION_MODEL=qwen-plus

# 关闭压缩
CONTEXT_COMPRESSION_ENABLED=false
```

---

## 8. 一句一句教你读代码

以下按执行顺序，逐行解释 `context_compressor.py` 的核心代码。

### 8.1 打开文件，声明它是什么

```python
"""上下文自动压缩中间件"""
```
→ 这是一个叫"文档字符串"的东西，Python 用它描述这个文件是干什么的。

### 8.2 导入需要用到的工具

```python
from langchain.agents.middleware import AgentMiddleware
```
→ 从 LangChain 框架中导入 `AgentMiddleware` 这个类。我们要继承它，就像继承"汽车"来造一辆"特斯拉"。

```python
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, RemoveMessage,
)
```
→ 从 LangChain 导入各种消息类型：
- `HumanMessage` = 用户说的话
- `AIMessage` = AI 说的话
- `SystemMessage` = 系统指令（比如"你是一个助手"）
- `RemoveMessage` = 用来删除消息的指令

### 8.3 告诉"压缩 AI"它应该怎么做

```python
SUMMARY_SYSTEM_PROMPT = """你是一个对话摘要生成器..."""
```
→ 这是一个"提示词模板"，告诉用于生成摘要的 AI 模型："你现在是摘要员，不是聊天机器人。请帮我把对话总结一下。"

### 8.4 每隔多少条消息才值得压缩

```python
def _should_compress(self, messages):
    if not config.context_compression_enabled:  # 如果用户关掉了压缩
        return False
    if len(messages) <= config.context_compression_keep_recent * 2 + 2:
        return False  # 消息太少，不用压缩
    token_count = self._count_tokens(messages)
    limit = int(window_size * 0.7)  # 比如 32768 * 0.7 = 22937
    return token_count > limit  # token 数超过限制就返回 True
```
→ 三个条件全部通过才触发压缩：
1. 用户没关掉压缩
2. 消息够多（至少超过保留轮数）
3. token 数超过阈值

### 8.5 拆分消息

```python
def _split_messages(self, messages):
    keep_count = config.context_compression_keep_recent * 2  # 4 轮 × 2 = 8 条
    body = list(messages[1:])  # 去掉第一条系统消息
    if len(body) <= keep_count:
        return [], body  # 消息还不够多，不分了
    old = body[:-keep_count]   # 前面的老消息 → 需要压缩
    recent = body[-keep_count:]  # 最后 8 条 → 保留原样
    return old, recent
```

假设一共有 50 条消息（不包括系统消息），保留最近 8 条：
```
[消息1] [消息2] ... [消息42] [消息43] ... [消息50]
 ←──────── old (42条) ──→            ←─ recent (8条) ─→
```

### 8.6 用 AI 生成摘要

```python
async def _generate_summary(self, old_messages, existing_summary=None):
    # 把消息变成可读的文本
    lines = []
    for msg in old_messages:
        role = "用户" if isinstance(msg, HumanMessage) else "AI"
        lines.append(f"[{role}]: {msg.content}")

    # 拼成完整对话
    conversation_text = "\n".join(lines)

    # 告诉"摘要 AI"：请总结这些对话
    prompt = f"对话历史：\n{conversation_text}\n\n请将以上对话总结为简洁的摘要。"

    # 调用 AI
    response = await self.compression_llm.ainvoke([
        SystemMessage(content="你是一个对话摘要生成器..."),
        HumanMessage(content=prompt)
    ])

    return response.content  # AI 生成的摘要文本
```

### 8.7 核心钩子：每次调模型前自动执行

```python
async def abefore_model(self, state, runtime):
    messages = state.get("messages", [])

    if not self._should_compress(messages):
        return None  # 不需要压缩，返回 None = 什么也不做

    # 需要压缩！
    thread_id = runtime.configurable.get("thread_id", "unknown")

    # 1. 查一下之前有没有摘要（有的话要合并）
    existing_summary = self._find_existing_summary(messages)

    # 2. 拆分消息
    old_msgs, recent_msgs = self._split_messages(messages)

    # 3. 生成摘要
    summary = await self._generate_summary(old_msgs, existing_summary)

    # 4. 保存压缩事件（前端会来取）
    self._events[thread_id] = summary

    # 5. 重建消息列表
    compressed = []
    compressed.append(messages[0])  # 保留原始系统提示
    compressed.append(SystemMessage(content=f"[对话历史摘要]\n{summary}"))
    compressed.extend(recent_msgs)  # 追加最近 4 轮对话

    # 6. 返回：用压缩后的消息替换全部消息
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),  # 先删除全部
            *compressed,  # 再写入压缩后的
        ]
    }
```

### 8.8 给前端取通知的接口

```python
def pop_event(self, thread_id):
    return self._events.pop(thread_id, None)
```
→ `pop` 的意思是"拿出来，同时删除"。每个事件只被取一次，避免重复显示。

```python
def has_event(self, thread_id):
    return thread_id in self._events
```
→ 只检查有没有事件，但不拿走。用于 `query()` 非流式方法。

---

## 总结

### 改动矩阵

| 文件 | 操作 | 修改量 |
|------|------|--------|
| `app/config.py` | 修改 | +5 行配置 |
| `app/core/context_compressor.py` | **新建** | ~140 行 |
| `app/services/rag_agent_service.py` | 修改 | +3 处 import/init/yield |
| `app/api/chat.py` | 修改 | +2 处事件处理 |

### 关键设计决策

1. **为什么用 AgentMiddleware？** → 自动拦截每次模型调用，无需手动在每个接口处理
2. **为什么单独建一个压缩 LLM？** → 摘要任务和对话任务需求不同，分开可以选更便宜的模型
3. **为什么用 SystemMessage 存摘要？** → 这样 LLM 会把摘要当成"系统提示"的一部分，更重视
4. **为什么保留最近 4 轮对话？** → 保证当前对话话题不丢失，同时大幅减少 token
