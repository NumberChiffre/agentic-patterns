from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from hsw.models import TaskContext, NodeResult, ResultStatus, ExecutionStrategy, ComplexityDecision
from hsw.nodes.atomizer import AtomizerNode
from hsw.nodes.planner import PlannerNode
from hsw.nodes.executor import ExecutorNode
from hsw.nodes.aggregator import AggregatorNode
from hsw.registry.agent_factory import create_dynamic_agent_factory
from hsw.workers.agents import create_supervisor_agent

class ROMAState(BaseModel):
    goal: str
    context: TaskContext
    decision: str = ""
    plan: dict = Field(default_factory=dict)
    subtask_results: list[NodeResult] = Field(default_factory=list)
    final_result: NodeResult | None = None
    supervisor_agent: object = None
    dynamic_factory: object = None
    
    class Config:
        arbitrary_types_allowed = True

async def atomizer_node(state: ROMAState) -> ROMAState:
    if not state.supervisor_agent:
        state.supervisor_agent = create_supervisor_agent()
    # Tracing handled by weave
    
    atomizer = AtomizerNode(
        context=state.context,
        agent=state.supervisor_agent
    )
    
    result = await atomizer.process()
    if result.status == ResultStatus.SUCCESS and isinstance(result.output, dict):
        state.decision = result.output.get("decision", ComplexityDecision.COMPLEX.value)
    else:
        state.decision = ComplexityDecision.COMPLEX.value
    
    return state

async def planner_node(state: ROMAState) -> ROMAState:
    planner = PlannerNode(
        context=state.context,
        agent=state.supervisor_agent
    )
    
    result = await planner.process()
    if result.status == ResultStatus.SUCCESS and isinstance(result.output, dict):
        state.plan = result.output
        # Update context with subtasks and dependencies
        state.context.subtasks = state.plan.get("subtasks", [])
    
    return state

async def executor_node(state: ROMAState) -> ROMAState:
    if not state.dynamic_factory:
        state.dynamic_factory = create_dynamic_agent_factory()
    
    subtasks = state.plan.get("subtasks", [])
    results = []
    
    # Determine execution approach based on strategy
    execution_strategy = state.context.execution_strategy
    should_execute_parallel = await _should_execute_parallel(subtasks, execution_strategy, state.context)
    
    if should_execute_parallel:
        results = await _execute_subtasks_parallel(subtasks, state)
    else:
        results = await _execute_subtasks_sequential(subtasks, state)
    
    state.subtask_results = results
    return state

async def _should_execute_parallel(subtasks: list[dict], strategy: ExecutionStrategy, context: TaskContext) -> bool:
    """Determine if subtasks should be executed in parallel based on strategy and dependencies"""
    
    # Force parallel execution for certain strategies
    if strategy in [ExecutionStrategy.SPEED_OPTIMIZED, ExecutionStrategy.PARALLEL]:
        return _has_independent_tasks(subtasks)
    
    # Force sequential for certain strategies
    if strategy in [ExecutionStrategy.SEQUENTIAL, ExecutionStrategy.QUALITY_OPTIMIZED]:
        return False
    
    # For balanced/cost-optimized, check dependencies and complexity
    if strategy in [ExecutionStrategy.BALANCED, ExecutionStrategy.COST_OPTIMIZED]:
        if not _has_independent_tasks(subtasks):
            return False
        
        # For cost optimization, only parallelize if we have sufficient budget
        if strategy == ExecutionStrategy.COST_OPTIMIZED:
            total_estimated_tokens = sum(task.get("estimated_tokens", 0) for task in subtasks)
            return total_estimated_tokens <= context.budget_tokens
    
    return _has_independent_tasks(subtasks)

def _has_independent_tasks(subtasks: list[dict]) -> bool:
    """Check if any subtasks can run independently"""
    for task in subtasks:
        if task.get("can_run_parallel", False) or not task.get("dependencies", []):
            return True
    return False

async def _execute_subtasks_parallel(subtasks: list[dict], state: ROMAState) -> list[NodeResult]:
    """Execute independent subtasks in parallel, others sequentially"""
    import asyncio
    
    results = []
    remaining_tasks = subtasks.copy()
    completed_task_ids = set()
    
    while remaining_tasks:
        # Find tasks that can run now (no dependencies or dependencies satisfied)
        ready_tasks = []
        for task in remaining_tasks:
            task_deps = task.get("dependencies", [])
            if not task_deps or all(dep_id in completed_task_ids for dep_id in task_deps):
                ready_tasks.append(task)
        
        if not ready_tasks:
            # If no ready tasks but tasks remain, there's a circular dependency - fallback to sequential
            return await _execute_subtasks_sequential(remaining_tasks, state)
        
        # Execute ready tasks in parallel
        if len(ready_tasks) > 1:
            # Execute multiple tasks in parallel
            parallel_results = await asyncio.gather(*[
                _execute_single_subtask(task, state) for task in ready_tasks
            ])
            results.extend(parallel_results)
        else:
            # Execute single task
            result = await _execute_single_subtask(ready_tasks[0], state)
            results.append(result)
        
        # Update completed tasks and remove from remaining
        for task in ready_tasks:
            completed_task_ids.add(task["id"])
            if task in remaining_tasks:
                remaining_tasks.remove(task)
                
        # Update context with successful results for next iteration
        for task, result in zip(ready_tasks, results[-len(ready_tasks):]):
            if result.status == ResultStatus.SUCCESS:
                context_keys = task.get("context_keys", [task["id"]])
                for key in context_keys:
                    state.context.context_data[key] = result.output
    
    return results

async def _execute_subtasks_sequential(subtasks: list[dict], state: ROMAState) -> list[NodeResult]:
    """Execute subtasks sequentially"""
    results = []
    
    for subtask in subtasks:
        result = await _execute_single_subtask(subtask, state)
        results.append(result)
        
        # Update context with successful results
        if result.status == ResultStatus.SUCCESS:
            context_keys = subtask.get("context_keys", [subtask["id"]])
            for key in context_keys:
                state.context.context_data[key] = result.output
    
    return results

async def _execute_single_subtask(subtask: dict, state: ROMAState) -> NodeResult:
    """Execute a single subtask with strategy-aware agent selection"""
    agent_type = subtask.get("agent_type", "analysis")
    task_goal = subtask.get("goal", "")
    
    # Create specialized agent with strategy-aware model selection
    capabilities = state.dynamic_factory.infer_capabilities_from_task(task_goal, agent_type)
    specialized_agent = _create_strategy_aware_agent(
        capabilities, task_goal, state.context.execution_strategy, state.dynamic_factory
    )
    
    # Create child context
    child_context = TaskContext(
        node_id=subtask["id"],
        goal=task_goal,
        parent_id=state.context.node_id,
        depth=state.context.depth + 1,
        max_depth=state.context.max_depth,
        subtasks=[],
        dependencies=[],
        context_data=state.context.context_data.copy(),
        tools_available=subtask.get("tools_needed", ["WebSearchTool"]),
        agent_capabilities=state.context.agent_capabilities,
        budget_tokens=subtask.get("estimated_tokens", state.context.budget_tokens // 4),
        budget_seconds=subtask.get("estimated_seconds", state.context.budget_seconds // 4),
        execution_strategy=state.context.execution_strategy
    )
    
    executor = ExecutorNode(
        context=child_context,
        agent=specialized_agent,
        subtask_data=subtask
    )
    
    return await executor.process()

def _create_strategy_aware_agent(capabilities: list, task_goal: str, strategy: ExecutionStrategy, factory):
    """Create an agent with model selection based on execution strategy"""
    
    # Get base agent
    agent = factory.create_agent_from_capabilities(capabilities, task_goal)
    
    # Adjust model based on strategy
    if strategy == ExecutionStrategy.SPEED_OPTIMIZED:
        # Use faster, smaller models
        agent.model = "gpt-4o-mini"
    elif strategy == ExecutionStrategy.QUALITY_OPTIMIZED:
        # Use best available models
        agent.model = "gpt-4o"
    elif strategy == ExecutionStrategy.COST_OPTIMIZED:
        # Use cheapest models
        agent.model = "gpt-4o-mini"
    else:
        # Balanced approach - use default model selection
        pass
    
    return agent

async def aggregator_node(state: ROMAState) -> ROMAState:
    aggregator = AggregatorNode(
        context=state.context,
        agent=state.supervisor_agent,
        child_results=state.subtask_results
    )
    
    result = await aggregator.process()
    state.final_result = result
    return state

async def simple_executor_node(state: ROMAState) -> ROMAState:
    if not state.dynamic_factory:
        state.dynamic_factory = create_dynamic_agent_factory()
    
    # Create analysis agent for simple task
    capabilities = state.dynamic_factory.infer_capabilities_from_task(state.goal, "analysis")
    analysis_agent = state.dynamic_factory.create_agent_from_capabilities(capabilities, state.goal)
    
    executor = ExecutorNode(
        context=state.context,
        agent=analysis_agent,
        subtask_data={
            "id": f"simple_{state.context.node_id}",
            "goal": state.goal,
            "agent_type": "analysis"
        }
    )
    
    result = await executor.process()
    state.final_result = result
    return state

def should_plan(state: ROMAState) -> str:
    return "planner" if state.decision == ComplexityDecision.COMPLEX.value else "simple_executor"

def create_roma_graph() -> StateGraph:
    workflow = StateGraph(ROMAState)
    
    # Add nodes
    workflow.add_node("atomizer", atomizer_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("simple_executor", simple_executor_node)
    
    # Add edges
    workflow.set_entry_point("atomizer")
    workflow.add_conditional_edges("atomizer", should_plan)
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "aggregator")
    workflow.add_edge("aggregator", END)
    workflow.add_edge("simple_executor", END)
    
    return workflow.compile()

async def execute_roma_graph(goal: str, max_depth: int = 3, budget_tokens: int = 8000, budget_seconds: int = 30) -> NodeResult:
    graph = create_roma_graph()
    
    initial_context = TaskContext(
        node_id="root",
        goal=goal,
        parent_id=None,
        depth=0,
        max_depth=max_depth,
        subtasks=[],
        dependencies=[],
        context_data={},
        tools_available=["WebSearchTool"],
        agent_capabilities=["research", "analysis", "verification"],
        budget_tokens=budget_tokens,
        budget_seconds=budget_seconds,
        execution_strategy=ExecutionStrategy.BALANCED
    )
    
    initial_state = ROMAState(
        goal=goal,
        context=initial_context
    )
    
    final_state = await graph.ainvoke(initial_state)
    
    # LangGraph returns a dict, so we need to access it properly
    if isinstance(final_state, dict) and "final_result" in final_state and final_state["final_result"]:
        return final_state["final_result"]
    elif hasattr(final_state, 'final_result') and final_state.final_result:
        return final_state.final_result
    else:
        # Fallback result
        return NodeResult(
            node_id="root",
            node_type="graph",
            status=ResultStatus.FAILED,
            output="Graph execution failed",
            error_message="No final result produced",
            tokens_used=0,
            seconds_elapsed=0.0,
            child_results=[],
            trace_data={"error": "graph_execution_failed"}
        )
