import asyncio
from hsw.interfaces import SubAgentSpawner, SubAgentCoordinator, SubAgentRequest
from hsw.models import NodeResult, TaskContext, ResultStatus
from hsw.registry.agent_factory import AgentCapability, create_dynamic_agent_factory
from hsw.tree.roma_tree import ROMATree

class DefaultSubAgentSpawner(SubAgentSpawner):
    """Default implementation of SubAgentSpawner interface"""
    
        self.agent_factory = agent_factory or create_dynamic_agent_factory()
        self.coordinator = DefaultSubAgentCoordinator(self.agent_factory, tracer)
    
    async def spawn_sub_agents(
        self, 
        requests: list[SubAgentRequest], 
        parent_context: TaskContext
    ) -> list[NodeResult]:
        """Spawn and execute sub-agents using ROMA trees for each request"""
        
        if not requests:
            return []
        
        # Determine execution pattern based on dependencies
        if self._has_dependencies(requests):
            return await self.coordinator.execute_hybrid(requests, parent_context)
        elif len(requests) > 1 and parent_context.execution_strategy.value in ["speed_optimized", "parallel"]:
            return await self.coordinator.execute_parallel(requests, parent_context)
        else:
            return await self.coordinator.execute_sequential(requests, parent_context)
    
    def should_spawn_sub_agents(self, task_complexity: dict) -> bool:
        """Determine if task warrants sub-agent spawning"""
        
        # Check depth limits
        if task_complexity.get("current_depth", 0) >= task_complexity.get("max_depth", 2) - 1:
            return False
        
        # Check budget constraints
        if (task_complexity.get("budget_tokens", 0) < 2000 or 
            task_complexity.get("budget_seconds", 0) < 10):
            return False
        
        # Check goal complexity
        goal = task_complexity.get("goal", "")
        if len(goal.split()) < 10:  # Simple tasks don't need sub-agents
            return False
        
        # Check complexity indicators
        complexity_indicators = [
            "analyze", "compare", "evaluate", "research", "investigate",
            "comprehensive", "detailed", "thorough", "multiple", "various",
            "complex", "extensive", "in-depth"
        ]
        
        goal_lower = goal.lower()
        complexity_score = sum(1 for indicator in complexity_indicators if indicator in goal_lower)
        
        return complexity_score >= 2
    
    def create_sub_agent_requests(self, goal: str, context: TaskContext) -> list[SubAgentRequest]:
        """Create sub-agent requests by analyzing goal complexity"""
        
        goal_lower = goal.lower()
        requests = []
        
        # Determine if this is a research-heavy task
        if any(word in goal_lower for word in ["research", "find", "gather", "investigate", "study"]):
            requests.append(SubAgentRequest(
                task_id=f"{context.node_id}_research",
                goal=f"Research and gather information about: {goal}",
                capabilities=[AgentCapability.RESEARCH, AgentCapability.ANALYSIS],
                tools_needed=["WebSearchTool"],
                context_keys=["research_findings"],
                estimated_tokens=context.budget_tokens // 3,
                estimated_seconds=context.budget_seconds // 3,
                can_run_parallel=False
            ))
        
        # Determine if this requires analysis
        if any(word in goal_lower for word in ["analyze", "compare", "evaluate", "assess", "examine"]):
            depends_on = [f"{context.node_id}_research"] if requests else []
            requests.append(SubAgentRequest(
                task_id=f"{context.node_id}_analysis",
                goal=f"Analyze and synthesize information for: {goal}",
                capabilities=[AgentCapability.ANALYSIS, AgentCapability.TECHNICAL_ANALYSIS],
                tools_needed=[],
                context_keys=["analysis_results"],
                estimated_tokens=context.budget_tokens // 3,
                estimated_seconds=context.budget_seconds // 3,
                depends_on=depends_on,
                can_run_parallel=not bool(depends_on)
            ))
        
        # Add verification step for quality
        if context.execution_strategy.value in ["quality_optimized", "balanced"]:
            depends_on = [req.task_id for req in requests]
            requests.append(SubAgentRequest(
                task_id=f"{context.node_id}_verification",
                goal=f"Verify and validate findings for: {goal}",
                capabilities=[AgentCapability.VERIFICATION],
                tools_needed=[],
                context_keys=["verification_results"],
                estimated_tokens=context.budget_tokens // 4,
                estimated_seconds=context.budget_seconds // 4,
                depends_on=depends_on,
                can_run_parallel=False
            ))
        
        # If no specific patterns matched, create a general decomposition
        if not requests:
            requests = self._create_generic_decomposition(goal, context)
        
        return requests
    
    def _has_dependencies(self, requests: list[SubAgentRequest]) -> bool:
        """Check if any requests have dependencies"""
        return any(req.depends_on for req in requests)
    
    def _create_generic_decomposition(self, goal: str, context: TaskContext) -> list[SubAgentRequest]:
        """Create a generic 3-step decomposition for complex goals"""
        budget_per_task = context.budget_tokens // 3
        time_per_task = context.budget_seconds // 3
        
        return [
            SubAgentRequest(
                task_id=f"{context.node_id}_step1",
                goal=f"Step 1 - Information gathering for: {goal}",
                capabilities=[AgentCapability.RESEARCH],
                tools_needed=["WebSearchTool"],
                context_keys=["step1_results"],
                estimated_tokens=budget_per_task,
                estimated_seconds=time_per_task,
                can_run_parallel=False
            ),
            SubAgentRequest(
                task_id=f"{context.node_id}_step2",
                goal=f"Step 2 - Analysis and processing for: {goal}",
                capabilities=[AgentCapability.ANALYSIS],
                tools_needed=[],
                context_keys=["step2_results"],
                estimated_tokens=budget_per_task,
                estimated_seconds=time_per_task,
                depends_on=[f"{context.node_id}_step1"],
                can_run_parallel=False
            ),
            SubAgentRequest(
                task_id=f"{context.node_id}_step3",
                goal=f"Step 3 - Final synthesis for: {goal}",
                capabilities=[AgentCapability.ANALYSIS, AgentCapability.VERIFICATION],
                tools_needed=[],
                context_keys=["step3_results"],
                estimated_tokens=budget_per_task,
                estimated_seconds=time_per_task,
                depends_on=[f"{context.node_id}_step2"],
                can_run_parallel=False
            )
        ]


class DefaultSubAgentCoordinator(SubAgentCoordinator):
    """Default implementation of SubAgentCoordinator interface"""
    
    def __init__(self, agent_factory, tracer):
        self.agent_factory = agent_factory
    
    async def execute_parallel(
        self, 
        requests: list[SubAgentRequest], 
        context: TaskContext
    ) -> list[NodeResult]:
        """Execute independent sub-agents in parallel using ROMA trees"""
        
        # Filter to only independent requests
        independent_requests = [req for req in requests if not req.depends_on]
        
        if not independent_requests:
            return await self.execute_sequential(requests, context)
        
        # Create ROMA trees for each independent request
        roma_tasks = []
        for request in independent_requests:
            roma_tree = self._create_roma_tree_for_request(request, context)
            roma_tasks.append(roma_tree.execute())
        
        # Execute in parallel
        results = await asyncio.gather(*roma_tasks, return_exceptions=True)
        
        # Convert exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(NodeResult(
                    node_id=independent_requests[i].task_id,
                    node_type="sub_agent",
                    status=ResultStatus.FAILED,
                    output="",
                    error_message=str(result),
                    tokens_used=0,
                    seconds_elapsed=0.0,
                    child_results=[],
                    trace_data={"parallel_execution_failed": True}
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def execute_sequential(
        self, 
        requests: list[SubAgentRequest], 
        context: TaskContext
    ) -> list[NodeResult]:
        """Execute sub-agents sequentially using ROMA trees"""
        
        results = []
        updated_context = context.model_copy()
        
        for request in requests:
            roma_tree = self._create_roma_tree_for_request(request, updated_context)
            
            try:
                result = await roma_tree.execute()
                results.append(result)
                
                # Update context with successful results
                if result.status == ResultStatus.SUCCESS:
                    for key in request.context_keys:
                        updated_context.context_data[key] = result.output
                        
            except Exception as e:
                error_result = NodeResult(
                    node_id=request.task_id,
                    node_type="sub_agent",
                    status=ResultStatus.FAILED,
                    output="",
                    error_message=str(e),
                    tokens_used=0,
                    seconds_elapsed=0.0,
                    child_results=[],
                    trace_data={"sequential_execution_failed": True}
                )
                results.append(error_result)
        
        return results
    
    async def execute_hybrid(
        self, 
        requests: list[SubAgentRequest], 
        context: TaskContext
    ) -> list[NodeResult]:
        """Execute sub-agents with mixed parallel/sequential patterns"""
        
        results = []
        remaining_requests = requests.copy()
        completed_task_ids = set()
        updated_context = context.model_copy()
        
        while remaining_requests:
            # Find requests ready to execute (no dependencies or dependencies satisfied)
            ready_requests = []
            for req in remaining_requests:
                if not req.depends_on or all(dep_id in completed_task_ids for dep_id in req.depends_on):
                    ready_requests.append(req)
            
            if not ready_requests:
                # Circular dependency - fallback to sequential execution
                return await self.execute_sequential(remaining_requests, updated_context)
            
            # Execute ready requests (parallel if multiple, sequential if one)
            if len(ready_requests) > 1:
                batch_results = await self.execute_parallel(ready_requests, updated_context)
            else:
                batch_results = await self.execute_sequential(ready_requests, updated_context)
            
            results.extend(batch_results)
            
            # Update completed task IDs and context
            for req, result in zip(ready_requests, batch_results):
                completed_task_ids.add(req.task_id)
                if result.status == ResultStatus.SUCCESS:
                    for key in req.context_keys:
                        updated_context.context_data[key] = result.output
            
            # Remove completed requests
            for req in ready_requests:
                if req in remaining_requests:
                    remaining_requests.remove(req)
        
        return results
    
    def _create_roma_tree_for_request(self, request: SubAgentRequest, parent_context: TaskContext) -> ROMATree:
        """Create a ROMA tree for a sub-agent request"""
        
        return ROMATree(
            root_goal=request.goal,
            max_depth=parent_context.max_depth - parent_context.depth,
            budget_tokens=request.estimated_tokens,
            budget_seconds=request.estimated_seconds
        )

