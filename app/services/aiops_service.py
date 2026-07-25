"""
通用 Plan-Execute-Replan 服务
基于 LangGraph 官方教程实现

L2 会话记忆: SqliteSaver 持久化 checkpoint
L3 情景记忆: EpisodeStore 记录任务历史
"""

from typing import AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
from loguru import logger

from app.agent.aiops import PlanExecuteState, planner, executor, replanner
from app.memory import get_memory_manager


# 节点名称常量
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """通用 Plan-Execute-Replan 服务 (带持久化记忆)"""

    def __init__(self):
        """初始化服务 - 延迟加载 checkpointer 和 graph"""
        self._checkpointer = None
        self._graph = None
        self._memory_manager = None
        logger.info("AIOpsService 实例已创建 (延迟初始化)")

    @property
    def memory_manager(self):
        """延迟获取 MemoryManager (首次访问时创建)"""
        if self._memory_manager is None:
            self._memory_manager = get_memory_manager()
        return self._memory_manager

    @property
    def checkpointer(self):
        """延迟获取 SqliteSaver (L2 会话记忆)"""
        if self._checkpointer is None:
            self._checkpointer = self.memory_manager.session_store
        return self._checkpointer

    @property
    def graph(self):
        """延迟构建 LangGraph 工作流"""
        if self._graph is None:
            self._graph = self._build_graph()
            logger.info("Plan-Execute-Replan 工作流图构建完成")
        return self._graph

    def _build_graph(self):
        """构建 Plan-Execute-Replan 工作流"""
        logger.info("构建工作流图...")

        # 创建状态图
        workflow = StateGraph(PlanExecuteState)

        # 添加节点
        workflow.add_node(NODE_PLANNER, planner)      # 制定计划
        workflow.add_node(NODE_EXECUTOR, executor)  # 执行步骤
        workflow.add_node(NODE_REPLANNER, replanner)  # 重新规划

        # 设置入口点
        workflow.set_entry_point(NODE_PLANNER)

        # 定义边
        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)     # planner -> executor
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)   # executor -> replanner

        # replanner 的条件边
        def should_continue(state: PlanExecuteState) -> str:
            """判断是否继续执行"""
            if state.get("response"):
                logger.info("已生成最终响应，结束流程")
                return END

            plan = state.get("plan", [])
            if plan:
                logger.info(f"继续执行，剩余 {len(plan)} 个步骤")
                return NODE_EXECUTOR

            logger.info("计划执行完毕，生成最终响应")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END
            }
        )

        # 编译工作流 (checkpointer 会在首次访问 graph 属性时注入)
        compiled_graph = workflow.compile(checkpointer=self.checkpointer)
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行 Plan-Execute-Replan 流程

        Args:
            user_input: 用户的任务描述
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 流式事件
        """
        logger.info(f"[会话 {session_id}] 开始执行任务: {user_input}")

        # L3 情景记忆: 开始记录任务
        self.memory_manager.start_episode(
            session_id=session_id,
            task_input=user_input,
            task_type="aiops",
        )

        # 捕获初始计划 (Planner 输出)
        original_plan: list[str] = []

        try:
            # 初始化状态
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            # 流式执行工作流
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                # 解析事件
                for node_name, node_output in event.items():
                    logger.info(f"节点 '{node_name}' 输出事件")

                    # 捕获 Planner 输出的初始计划
                    if node_name == NODE_PLANNER:
                        plan_from_output = node_output.get("plan", [])
                        if plan_from_output and not original_plan:
                            original_plan = list(plan_from_output)
                            # L3: 更新 episode 计划
                            self.memory_manager.update_episode_plan(
                                session_id, original_plan
                            )
                        yield self._format_planner_event(node_output)

                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)

                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            # 获取最终状态
            final_state = self.graph.get_state(config_dict)
            final_response = ""
            final_past_steps: list[tuple] = []

            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")
                final_past_steps = final_state.values.get("past_steps", [])

            # L3 情景记忆: 记录任务完成
            self.memory_manager.complete_episode(
                session_id=session_id,
                plan=original_plan,
                past_steps=final_past_steps,
                response=final_response,
                status="completed",
            )

            # 发送完成事件
            yield {
                "type": "complete",
                "stage": "complete",
                "message": "任务执行完成",
                "response": final_response
            }

            logger.info(f"[会话 {session_id}] 任务执行完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] 任务执行失败: {e}", exc_info=True)

            # L3 情景记忆: 记录任务失败
            self.memory_manager.complete_episode(
                session_id=session_id,
                plan=original_plan,
                past_steps=[],
                response="",
                status="failed",
                error_message=str(e),
            )

            yield {
                "type": "error",
                "stage": "error",
                "message": f"任务执行出错: {str(e)}"
            }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        AIOps 诊断接口（兼容旧接口）

        Args:
            session_id: 会话ID

        Yields:
            Dict[str, Any]: 诊断过程的流式事件
        """
        from textwrap import dedent
        aiops_task = dedent("""诊断当前系统是否存在告警，如果存在告警请详细分析告警原因并生成诊断报告，诊断报告输出格式要求：
                ```
                # 告警分析报告

                ---

                ## 📋 活跃告警清单

                | 告警名称 | 级别 | 目标服务 | 首次触发时间 | 最新触发时间 | 状态 |
                |---------|------|----------|-------------|-------------|------|
                | [告警1名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |
                | [告警2名称] | [级别] | [服务名] | [时间] | [时间] | 活跃 |

                ---

                ## 🔍 告警根因分析1 - [告警名称]

                ### 告警详情
                - **告警级别**: [级别]
                - **受影响服务**: [服务名]
                - **持续时间**: [X分钟]

                ### 症状描述
                [根据监控指标描述症状]

                ### 日志证据
                [引用查询到的关键日志]

                ### 根因结论
                [基于证据得出的根本原因]

                ---

                ## 🛠️ 处理方案执行1 - [告警名称]

                ### 已执行的排查步骤
                1. [步骤1]
                2. [步骤2]

                ### 处理建议
                [给出具体的处理建议]

                ### 预期效果
                [说明预期的效果]

                ---

                ## 🔍 告警根因分析2 - [告警名称]
                [如果有第2个告警，重复上述格式]

                ---

                ## 📊 结论

                ### 整体评估
                [总结所有告警的整体情况]

                ### 关键发现
                - [发现1]
                - [发现2]

                ### 后续建议
                1. [建议1]
                2. [建议2]

                ### 风险评估
                [评估当前风险等级和影响范围]
                ```

                **重要提醒**：
                - 最终输出必须是纯 Markdown 文本，不要包含 JSON 结构
                - 所有内容必须基于工具查询的真实数据，严禁编造
                - 如果某个步骤失败，在结论中如实说明，不要跳过""")

        async for event in self.execute(aiops_task, session_id):
            # 转换事件格式以兼容旧的 API
            if event.get("type") == "complete":
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "诊断流程完成",
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", "")
                    }
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        """格式化 Planner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "规划节点执行中"
            }

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"执行计划已制定，共 {len(plan)} 个步骤",
            "plan": plan
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        """格式化 Executor 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "执行节点运行中"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"步骤执行完成 ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "开始执行步骤"
            }

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        """格式化 Replanner 节点事件"""
        if not state:
            return {
                "type": "status",
                "stage": "replanner",
                "message": "评估节点运行中"
            }

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "最终报告已生成",
                "report": response
            }
        else:
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"评估完成，{'继续执行剩余步骤' if plan else '准备生成最终响应'}",
                "remaining_steps": len(plan)
            }


# 全局单例
aiops_service = AIOpsService()
