import asyncio
from agents import Agent
import logging
from hsw.nodes.executor import ExecutorNode
from hsw.nodes.aggregator import AggregatorNode
from hsw.models import TaskContext, NodeResult, ResultStatus, ComplexityDecision
from hsw.workers.agents import (
    create_retrieval_agent, create_extraction_agent, 
    create_analysis_agent, create_verification_agent
)
from hsw.workers.agents import create_supervisor_agent
from hsw.registry.agent_factory import create_dynamic_agent_factory
from hsw.graph.roma_graph import execute_roma_graph

class ROMATree:
    
    def __init__(self, root_goal: str, max_depth: int = 3, budget_tokens: int = 8000, budget_seconds: int = 30):
        self.root_goal = root_goal
        self.max_depth = max_depth
        self.budget_tokens = budget_tokens  
        self.budget_seconds = budget_seconds
        # Tracing handled by weave
        self.agent_cache: dict[str, Agent] = {}
        self.dynamic_factory = create_dynamic_agent_factory()
    
    async def execute(self) -> NodeResult:
        logging.info("🌳 ROMA TREE: Beginning ROMA analysis pipeline")
        
        # Use LangGraph for proper DAG execution
        result = await execute_roma_graph(
            goal=self.root_goal,
            max_depth=self.max_depth,
            budget_tokens=self.budget_tokens,
            budget_seconds=self.budget_seconds
        )
        
        logging.info(f"🏁 ROMA TREE: Completed pipeline - {result.status.value if result.status else 'unknown'} ({result.tokens_used} tokens, {result.seconds_elapsed:.1f}s)")
        return result
    
    async def _execute_node_recursive(self, context: TaskContext, agent: Agent) -> NodeResult:
        if context.depth >= context.max_depth:
            executor = ExecutorNode(
                context=context,
                agent=agent,
                subtask_data={
                    "id": f"max_depth_{context.node_id}",
                    "goal": context.goal,
                    "agent_type": "general"
                }
            )
            return await executor.process()
        
        logging.info("🔍 ROMA TREE: Starting atomizer for complexity assessment")
        atomizer_result = await atomizer.process()
        
        if atomizer_result.status != ResultStatus.SUCCESS:
            return atomizer_result
        
        decision = atomizer_result.output
        decision_value = decision.get("decision", "unknown")
        logging.info(f"🎯 ROMA TREE: Atomizer decision: {decision_value} (depth {context.depth})")
        
        if decision.get("decision") == ComplexityDecision.SIMPLE.value:
            logging.info("⚡ ROMA TREE: Simple task → direct execution")
            executor = ExecutorNode(
                context=context,
                agent=self._get_specialized_agent("analysis"),
                subtask_data={
                    "id": f"simple_{context.node_id}",
                    "goal": context.goal,
                    "agent_type": "analysis"
                }
            )
            return await executor.process()
        
        logging.info("🧩 ROMA TREE: Complex task → planning & decomposition")
        planner_result = await planner.process()
        
        if planner_result.status != ResultStatus.SUCCESS:
            return planner_result
        
        plan_data = planner_result.output
        subtask_count = len(plan_data.get("subtasks", []))
        logging.info(f"📋 ROMA TREE: Planner created {subtask_count} subtasks")
        subtasks = plan_data.get("subtasks", [])
        
        if not subtasks:
            return NodeResult(
                node_id=context.node_id,
                node_type="planner",
                status=ResultStatus.FAILED,
                output="No subtasks generated",
                error_message="Planning failed to create subtasks",
                tokens_used=planner_result.tokens_used,
                seconds_elapsed=planner_result.seconds_elapsed,
                child_results=[],
                trace_data={"error": "no_subtasks"}
            )
        
        logging.info(f"🏃 ROMA TREE: Executing {len(subtasks)} subtasks")
        child_results = await self._execute_subtasks(context, subtasks)
        
        success_count = len([r for r in child_results if r.status.value == "success"])
        logging.info(f"🔗 ROMA TREE: Aggregating {success_count}/{len(child_results)} successful subtask results")
        aggregator = AggregatorNode(
            context=context,
            agent=agent,
            child_results=child_results
        )
        
        return await aggregator.process()
    
    async def _execute_subtasks(self, parent_context: TaskContext, subtasks: list[dict]) -> list[NodeResult]:
        """Execute subtasks using simple sequential execution"""
        results = []
        context_data = parent_context.context_data.copy()
        
        # Determine sequential and parallel tasks based on dependencies
        sequential_tasks = []
        parallel_tasks = []
        
        for task in subtasks:
            if task.get("dependencies", []) or not task.get("can_run_parallel", False):
                sequential_tasks.append(task)
            else:
                parallel_tasks.append(task)
        
        # Execute sequential tasks first
        for task in sequential_tasks:
            result = await self._execute_single_task(parent_context, task, context_data)
            results.append(result)
            if result.status == ResultStatus.SUCCESS:
                self._update_context_from_result(task, result, context_data)
        
        # Execute parallel tasks
        if parallel_tasks:
            parallel_results = await self._execute_parallel_tasks(parent_context, parallel_tasks, context_data)
            results.extend(parallel_results)
            for i, result in enumerate(parallel_results):
                if result.status == ResultStatus.SUCCESS:
                    task = parallel_tasks[i]
                    self._update_context_from_result(task, result, context_data)
        
        return results
    
    async def _execute_single_task(self, parent_context: TaskContext, task: dict, context_data: dict) -> NodeResult:
        """Execute a single subtask"""
        child_context = self._create_child_context(parent_context, task, context_data)
        specialized_agent = self._get_specialized_agent(task.get("agent_type", "analysis"), task.get("goal", ""))
        
        executor = ExecutorNode(
            context=child_context,
            agent=specialized_agent,
            subtask_data=task
        )
        
        logging.info("ROMATree: executing task id=%s type=%s", task.get("id"), task.get("agent_type"))
        return await executor.process()
    
    async def _execute_parallel_tasks(self, parent_context: TaskContext, tasks: list[dict], context_data: dict) -> list[NodeResult]:
        """Execute multiple tasks in parallel"""
        if not tasks:
            return []
        
        parallel_futures = []
        for task in tasks:
            child_context = self._create_child_context(parent_context, task, context_data)
            specialized_agent = self._get_specialized_agent(task.get("agent_type", "analysis"), task.get("goal", ""))
            
            executor = ExecutorNode(
                context=child_context,
                agent=specialized_agent,
                subtask_data=task
            )
            
            parallel_futures.append(executor.process())
        
        logging.info("ROMATree: executing %s parallel tasks", len(tasks))
        parallel_results = await asyncio.gather(*parallel_futures, return_exceptions=True)
        
        results = []
        for i, result in enumerate(parallel_results):
            if isinstance(result, Exception):
                task = tasks[i]
                logging.error("ROMATree: parallel task failed id=%s error=%s", task.get("id"), str(result))
                error_result = NodeResult(
                    node_id=task["id"],
                    node_type="executor",
                    status=ResultStatus.FAILED,
                    output="",
                    error_message=str(result),
                    tokens_used=0,
                    seconds_elapsed=0.0,
                    child_results=[],
                    trace_data={"error": "parallel_execution_failed"}
                )
                results.append(error_result)
            else:
                logging.info("ROMATree: parallel task succeeded idx=%s", i)
                results.append(result)
        
        return results
    
    def _update_context_from_result(self, task: dict, result: NodeResult, context_data: dict) -> None:
        """Update context data with successful task result"""
        context_keys = task.get("context_keys", [task["id"]])
        for key in context_keys:
            context_data[key] = result.output
    
    def _create_child_context(self, parent_context: TaskContext, subtask: dict, context_data: dict) -> TaskContext:
        return TaskContext(
            node_id=subtask["id"],
            goal=subtask["goal"],
            parent_id=parent_context.node_id,
            depth=parent_context.depth + 1,
            max_depth=parent_context.max_depth,
            subtasks=[],
            dependencies=[],
            context_data=context_data,
            tools_available=subtask.get("tools_needed", ["WebSearchTool"]),
            agent_capabilities=parent_context.agent_capabilities,
            budget_tokens=subtask.get("estimated_tokens", parent_context.budget_tokens // 4),
            budget_seconds=subtask.get("estimated_seconds", parent_context.budget_seconds // 4),
            execution_strategy=parent_context.execution_strategy
        )
    
    def _get_agent(self, agent_type: str) -> Agent:
        if agent_type not in self.agent_cache:
            if agent_type == "supervisor":
                logging.info("ROMATree: create supervisor agent")
                self.agent_cache[agent_type] = create_supervisor_agent()
            else:
                self.agent_cache[agent_type] = self._get_specialized_agent(agent_type)
        
        return self.agent_cache[agent_type]
    
    def _get_specialized_agent(self, agent_type: str, task_goal: str = "") -> Agent:
        cache_key = f"{agent_type}_{hash(task_goal) if task_goal else 'default'}"
        
        if cache_key not in self.agent_cache:
            if task_goal:
                # Use dynamic factory for task-specific agents
                logging.info(f"🏭 ROMA TREE: Creating {agent_type} agent for task: '{task_goal}'")
                capabilities = self.dynamic_factory.infer_capabilities_from_task(task_goal, agent_type)
                self.agent_cache[cache_key] = self.dynamic_factory.create_agent_from_capabilities(capabilities, task_goal)
            else:
                # Fallback to static agents
                if agent_type == "retrieval":
                    logging.info("ROMATree: create retrieval agent")
                    self.agent_cache[cache_key] = create_retrieval_agent()
                elif agent_type == "extraction": 
                    logging.info("ROMATree: create extraction agent")
                    self.agent_cache[cache_key] = create_extraction_agent()
                elif agent_type == "analysis":
                    logging.info("ROMATree: create analysis agent")
                    self.agent_cache[cache_key] = create_analysis_agent()
                elif agent_type == "verification":
                    logging.info("ROMATree: create verification agent")
                    self.agent_cache[cache_key] = create_verification_agent()
                else:
                    logging.info("ROMATree: default to analysis agent for type=%s", agent_type)
                    self.agent_cache[cache_key] = create_analysis_agent()
        
        return self.agent_cache[cache_key]
