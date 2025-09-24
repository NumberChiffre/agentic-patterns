import time
import json
import re
import logging
import weave
from agents import Agent
from hsw.nodes.roma_node import ROMANode
from hsw.models import TaskContext, SubtaskDependency, NodeResult, NodeType, NodeState, ResultStatus, DependencyType, ExecutionStrategy
from hsw.workers.agents import stream_agent_response

logger = logging.getLogger(__name__)

class PlannerNode(ROMANode):
    
    def __init__(self, context: TaskContext, agent: Agent):
        super().__init__(
            node_id=f"planner_{context.node_id}",
            node_type=NodeType.PLANNER,
            state=NodeState.CREATED,
            context=context,
            agent=agent
        )
    
    @weave.op
    async def execute(self) -> NodeResult:
        logger.info(f"📋 PLANNER: Decomposing complex task - '{self.context.goal}' (depth {self.context.depth}/{self.context.max_depth})")
        self.state = NodeState.PLANNING
        start_time = time.time()
        
        planning_prompt = f"""
        ROMA Planner - Task Decomposition:
        
        GOAL: {self.context.goal}
        CURRENT CONTEXT: {self.context.context_data}
        AVAILABLE TOOLS: {', '.join(self.context.tools_available)}
        AGENT CAPABILITIES: {', '.join(self.context.agent_capabilities)}
        BUDGET: {self.context.budget_tokens} tokens, {self.context.budget_seconds} seconds
        DEPTH: {self.context.depth}/{self.context.max_depth}
        
        TASK: Break this complex goal into subtasks that can be executed by specialized agents.
        
        Consider:
        - What information needs to be gathered?
        - What analysis needs to be performed?  
        - What dependencies exist between subtasks?
        - Which subtasks can run in parallel vs sequential?
        - How to distribute the budget across subtasks?
        
        Create a decomposition plan with JSON format:
        {{
            "subtasks": [
                {{
                    "id": "unique_task_id",
                    "goal": "specific task description", 
                    "agent_type": "retrieval|extraction|analysis|verification|research|synthesis",
                    "tools_needed": ["WebSearchTool", "CalculatorTool"],
                    "dependencies": ["other_task_id"],
                    "can_run_parallel": true/false,
                    "estimated_tokens": number,
                    "estimated_seconds": number,
                    "context_keys": ["key1", "key2"]
                }}
            ],
            "execution_strategy": "description of how tasks should be coordinated",
            "success_criteria": "how to determine if plan succeeded"
        }}
        """
        
        tokens_used, response = await stream_agent_response(self.agent, planning_prompt)
        elapsed_time = time.time() - start_time
        
        try:
            plan_data = self._parse_plan(response)
            num_subtasks = len(plan_data.get("subtasks", []))
            strategy = plan_data.get("execution_strategy", "unknown")
            
            logger.info(f"✅ PLANNER: Created {num_subtasks} subtasks with {strategy} strategy:")
            for i, subtask in enumerate(plan_data.get("subtasks", [])):
                subtask_goal = subtask.get("goal", "Unknown goal")
                logger.info(f"   {i+1}. {subtask_goal}")
            dependencies = self._create_dependencies(plan_data["subtasks"])
            execution_strategy = self._parse_execution_strategy(plan_data.get("execution_strategy", ""))
            
            # Update context with subtasks, dependencies, and execution strategy
            self.context.subtasks = plan_data["subtasks"]
            self.context.dependencies = dependencies
            self.context.execution_strategy = execution_strategy
            
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=ResultStatus.SUCCESS,
                output=plan_data,
                error_message=None,
                tokens_used=tokens_used,
                seconds_elapsed=elapsed_time,
                child_results=[],
                trace_data={
                    "subtasks_created": len(plan_data["subtasks"]),
                    "parallel_tasks": len([t for t in plan_data["subtasks"] if t.get("can_run_parallel", False)]),
                    "strategy": plan_data.get("execution_strategy", "")
                }
            )
            
        except Exception as e:
            fallback_plan = self._create_fallback_plan()
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=ResultStatus.PARTIAL,
                output=fallback_plan,
                error_message=f"Plan parsing failed, using fallback: {str(e)}",
                tokens_used=tokens_used,
                seconds_elapsed=elapsed_time,
                child_results=[],
                trace_data={"fallback_used": True, "error": str(e)}
            )
    
    def _parse_plan(self, response: str) -> dict[str, list | str]:
        
        # Try to extract JSON with multiple strategies
        json_candidates = []
        
        # Strategy 1: Find complete JSON blocks
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_candidates.append(json_match.group(0))
        
        # Strategy 2: Look for JSON-like structures with common patterns
        json_pattern = re.compile(r'\{[^{}]*"subtasks"[^{}]*\[.*?\][^{}]*\}', re.DOTALL)
        matches = json_pattern.findall(response)
        json_candidates.extend(matches)
        
        # Strategy 3: Extract from code blocks
        code_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        json_candidates.extend(code_blocks)
        
        # Try parsing each candidate
        for candidate in json_candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict) and 'subtasks' in parsed:
                    return self._validate_and_fix_plan(parsed)
            except json.JSONDecodeError:
                # Try fixing common JSON issues
                try:
                    fixed_json = self._fix_malformed_json(candidate)
                    parsed = json.loads(fixed_json)
                    if isinstance(parsed, dict) and 'subtasks' in parsed:
                        return self._validate_and_fix_plan(parsed)
                except (json.JSONDecodeError, ValueError):
                    continue
        
        # If no valid JSON found, try to extract structured data from text
        return self._extract_plan_from_text(response)
    
    def _fix_malformed_json(self, json_str: str) -> str:
        # Fix common JSON formatting issues
        json_str = json_str.strip()
        
        # Fix missing quotes around keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        
        # Fix trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix single quotes to double quotes
        json_str = json_str.replace("'", '"')
        
        # Fix boolean values
        json_str = re.sub(r'\btrue\b', 'true', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bfalse\b', 'false', json_str, flags=re.IGNORECASE)
        json_str = re.sub(r'\bnull\b', 'null', json_str, flags=re.IGNORECASE)
        
        return json_str
    
    def _validate_and_fix_plan(self, plan: dict) -> dict[str, list | str]:
        # Ensure required fields exist
        if 'subtasks' not in plan:
            plan['subtasks'] = []
        
        if 'execution_strategy' not in plan:
            plan['execution_strategy'] = "Auto-generated execution strategy"
        
        if 'success_criteria' not in plan:
            plan['success_criteria'] = "All subtasks complete successfully"
        
        # Validate and fix each subtask
        valid_subtasks = []
        for i, subtask in enumerate(plan['subtasks']):
            if not isinstance(subtask, dict):
                continue
            
            # Ensure required fields
            if 'id' not in subtask:
                subtask['id'] = f"task_{i+1:03d}"
            if 'goal' not in subtask:
                subtask['goal'] = f"Complete subtask {i+1}"
            if 'agent_type' not in subtask:
                subtask['agent_type'] = "analysis"
            if 'dependencies' not in subtask:
                subtask['dependencies'] = []
            if 'can_run_parallel' not in subtask:
                subtask['can_run_parallel'] = False
            if 'estimated_tokens' not in subtask:
                subtask['estimated_tokens'] = self.context.budget_tokens // max(len(plan['subtasks']), 1)
            if 'estimated_seconds' not in subtask:
                subtask['estimated_seconds'] = self.context.budget_seconds // max(len(plan['subtasks']), 1)
            if 'tools_needed' not in subtask:
                subtask['tools_needed'] = []
            if 'context_keys' not in subtask:
                subtask['context_keys'] = []
            
            valid_subtasks.append(subtask)
        
        plan['subtasks'] = valid_subtasks
        return plan
    
    def _extract_plan_from_text(self, response: str) -> dict[str, list | str]:
        # Last resort: extract plan structure from natural language
        lines = response.split('\n')
        subtasks = []
        current_task = None
        task_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for task indicators
            if any(indicator in line.lower() for indicator in ['task', 'step', 'subtask', '1.', '2.', '3.', '-', '*']):
                if current_task:
                    subtasks.append(current_task)
                
                # Extract goal from line
                goal = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
                if not goal:
                    goal = f"Complete task {task_counter}"
                
                current_task = {
                    "id": f"extracted_task_{task_counter:03d}",
                    "goal": goal,
                    "agent_type": self._infer_agent_type(goal),
                    "tools_needed": self._infer_tools(goal),
                    "dependencies": [],
                    "can_run_parallel": False,
                    "estimated_tokens": self.context.budget_tokens // 3,
                    "estimated_seconds": self.context.budget_seconds // 3,
                    "context_keys": []
                }
                task_counter += 1
        
        if current_task:
            subtasks.append(current_task)
        
        # If no subtasks extracted, create minimal fallback
        if not subtasks:
            return self._create_fallback_plan()
        
        return {
            "subtasks": subtasks,
            "execution_strategy": "Extracted from natural language description",
            "success_criteria": "Complete all identified tasks"
        }
    
    def _infer_agent_type(self, goal: str) -> str:
        goal_lower = goal.lower()
        if any(word in goal_lower for word in ['search', 'find', 'research', 'gather', 'collect']):
            return 'retrieval'
        elif any(word in goal_lower for word in ['extract', 'identify', 'list', 'enumerate']):
            return 'extraction'
        elif any(word in goal_lower for word in ['analyze', 'compare', 'evaluate', 'assess']):
            return 'analysis'
        elif any(word in goal_lower for word in ['verify', 'check', 'validate', 'confirm']):
            return 'verification'
        else:
            return 'analysis'
    
    def _infer_tools(self, goal: str) -> list[str]:
        goal_lower = goal.lower()
        if any(word in goal_lower for word in ['search', 'find', 'research', 'web', 'online']):
            return ['WebSearchTool']
        return []
    
    def _parse_execution_strategy(self, strategy_text: str) -> ExecutionStrategy:
        """Extract execution strategy from planner output"""
        strategy_lower = strategy_text.lower()
        
        # Look for speed/performance keywords
        if any(keyword in strategy_lower for keyword in ['speed', 'fast', 'quick', 'rapid', 'parallel', 'concurrent']):
            return ExecutionStrategy.SPEED_OPTIMIZED
        
        # Look for cost optimization keywords
        elif any(keyword in strategy_lower for keyword in ['cost', 'cheap', 'efficient', 'minimal', 'budget']):
            return ExecutionStrategy.COST_OPTIMIZED
        
        # Look for quality keywords  
        elif any(keyword in strategy_lower for keyword in ['quality', 'accurate', 'thorough', 'comprehensive', 'detailed']):
            return ExecutionStrategy.QUALITY_OPTIMIZED
        
        # Look for sequential keywords
        elif any(keyword in strategy_lower for keyword in ['sequential', 'step-by-step', 'ordered', 'dependency']):
            return ExecutionStrategy.SEQUENTIAL
        
        # Look for parallel keywords
        elif any(keyword in strategy_lower for keyword in ['parallel', 'simultaneous', 'concurrent', 'independent']):
            return ExecutionStrategy.PARALLEL
        
        # Default to balanced strategy
        else:
            return ExecutionStrategy.BALANCED
    
    def _create_dependencies(self, subtasks: list[dict]) -> list[SubtaskDependency]:
        dependencies = []
        
        for task in subtasks:
            task_deps = task.get("dependencies", [])
            can_parallel = task.get("can_run_parallel", False)
            
            if not task_deps:
                dep_type = DependencyType.INDEPENDENT
            elif can_parallel:
                dep_type = DependencyType.PARALLEL
            else:
                dep_type = DependencyType.SEQUENTIAL
                
            dependencies.append(SubtaskDependency(
                task_id=task["id"],
                depends_on=task_deps,
                dependency_type=dep_type,
                can_run_parallel=can_parallel
            ))
        
        return dependencies
    
    def _create_fallback_plan(self) -> dict[str, list | str]:
        budget_per_task = self.context.budget_tokens // 3
        time_per_task = self.context.budget_seconds // 3
        
        return {
            "subtasks": [
                {
                    "id": "research_001",
                    "goal": f"Research information about: {self.context.goal}",
                    "agent_type": "retrieval",
                    "tools_needed": ["WebSearchTool"],
                    "dependencies": [],
                    "can_run_parallel": False,
                    "estimated_tokens": budget_per_task,
                    "estimated_seconds": time_per_task,
                    "context_keys": ["sources"]
                },
                {
                    "id": "analyze_001", 
                    "goal": f"Analyze and synthesize information for: {self.context.goal}",
                    "agent_type": "analysis",
                    "tools_needed": [],
                    "dependencies": ["research_001"],
                    "can_run_parallel": False,
                    "estimated_tokens": budget_per_task,
                    "estimated_seconds": time_per_task,
                    "context_keys": ["analysis"]
                },
                {
                    "id": "verify_001",
                    "goal": f"Verify and finalize answer for: {self.context.goal}",
                    "agent_type": "verification",
                    "tools_needed": [],
                    "dependencies": ["analyze_001"],
                    "can_run_parallel": False,
                    "estimated_tokens": budget_per_task,
                    "estimated_seconds": time_per_task,
                    "context_keys": ["verification"]
                }
            ],
            "execution_strategy": "Sequential execution with fallback plan",
            "success_criteria": "All subtasks complete successfully"
        }
