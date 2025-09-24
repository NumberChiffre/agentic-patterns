import time
import logging
from agents import Agent
from hsw.nodes.roma_node import ROMANode
from hsw.models import TaskContext, NodeResult, NodeType, NodeState, ResultStatus
from hsw.workers.agents import stream_agent_response

logger = logging.getLogger(__name__)

class ExecutorNode(ROMANode):
    subtask_data: dict = {}
    spawner: object | None = None
    
    def __init__(self, context: TaskContext, agent: Agent, subtask_data: dict, spawner=None):
        super().__init__(
            node_id=subtask_data["id"],
            node_type=NodeType.EXECUTOR,
            state=NodeState.CREATED,
            context=context,
            agent=agent
        )
        self.subtask_data = subtask_data
        # Sub-agent spawning is optional and injected to avoid inner imports
        self.spawner = spawner
    
    async def execute(self) -> NodeResult:
        task_type = self.subtask_data.get("type", "unknown")
        task_goal = self.subtask_data.get("goal", "")
        logger.info(f"🚀 EXECUTOR: Starting {task_type} worker - '{task_goal}'")
        
        self.state = NodeState.EXECUTING
        start_time = time.time()
        
        # Check if this executor should spawn sub-agents (recursive ROMA)
        task_complexity = {
            "goal": self.subtask_data.get("goal", ""),
            "current_depth": self.context.depth,
            "max_depth": self.context.max_depth,
            "budget_tokens": self.context.budget_tokens,
            "budget_seconds": self.context.budget_seconds
        }
        
        if self.spawner and self.spawner.should_spawn_sub_agents(task_complexity):
            logger.info("🔄 EXECUTOR: Spawning sub-agents for complex task")
            return await self._execute_with_sub_agents()
        
        execution_prompt = self._build_execution_prompt()
        logger.info(f"🤖 EXECUTOR: Running {task_type} agent with goal: '{task_goal}'")
        
        logger.info(f"💬 EXECUTOR: Sending prompt to {self.agent.name}")
        tokens_used, response = await stream_agent_response(self.agent, execution_prompt)
        elapsed_time = time.time() - start_time
        
        output = self._process_response(response)
        
        status = ResultStatus.SUCCESS if output else ResultStatus.FAILED
        logger.info(f"✅ EXECUTOR: {task_type} worker completed - {status.value} ({tokens_used} tokens, {elapsed_time:.1f}s)")
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type.value,
            status=status,
            output=output,
            error_message=None if output else "Empty response from agent",
            tokens_used=tokens_used,
            seconds_elapsed=elapsed_time,
            child_results=[],
            trace_data={
                "subtask_type": self.subtask_data.get("agent_type", "unknown"),
                "response_length": len(str(response)),
                "processed_output": len(str(output))
            }
        )
    
    def _build_execution_prompt(self) -> str:
        agent_type = self.subtask_data.get("agent_type", "general")
        goal = self.subtask_data["goal"]
        tools_needed = self.subtask_data.get("tools_needed", [])
        context_keys = self.subtask_data.get("context_keys", [])
        
        context_data = ""
        for key in context_keys:
            if key in self.context.context_data:
                context_data += f"\n{key.upper()}: {self.context.context_data[key]}"
        
        if agent_type == "retrieval":
            return f"""
            ROMA Executor - Information Retrieval:
            
            TASK: {goal}
            TOOLS AVAILABLE: {', '.join(tools_needed)}
            
            Your role: Search and gather comprehensive information to address this task.
            
            Instructions:
            1. Use available tools to search for relevant information
            2. Focus on factual, up-to-date sources
            3. Gather diverse perspectives if applicable
            4. Format results clearly with sources
            
            Provide detailed information with clear source attribution.
            """
            
        elif agent_type == "extraction":
            return f"""
            ROMA Executor - Information Extraction:
            
            TASK: {goal}
            CONTEXT DATA: {context_data}
            
            Your role: Extract key facts, entities, and structured data from provided context.
            
            Instructions:
            1. Identify salient facts and claims
            2. Normalize entities and dates where possible
            3. Return a concise list of fact statements
            4. Preserve attribution if available
            
            Provide a bullet list of extracted facts.
            """
            
        elif agent_type == "analysis":
            return f"""
            ROMA Executor - Analysis & Synthesis:
            
            TASK: {goal}
            CONTEXT DATA: {context_data}
            
            Your role: Analyze the provided information and synthesize insights.
            
            Instructions:
            1. Examine all provided context data carefully
            2. Identify key patterns, relationships, and insights
            3. Draw logical conclusions based on evidence
            4. Provide structured analysis with clear reasoning
            
            Deliver comprehensive analysis with supporting evidence.
            """
            
        elif agent_type == "verification":
            return f"""
            ROMA Executor - Verification & Quality Check:
            
            TASK: {goal}
            CONTEXT DATA: {context_data}
            
            Your role: Verify accuracy and completeness of information.
            
            Instructions:
            1. Check facts against provided sources
            2. Identify any inconsistencies or gaps
            3. Assess the quality and reliability of conclusions
            4. Provide coverage analysis and confidence ratings
            
            Return verification report with quality assessment.
            """
            
        elif agent_type == "research":
            return f"""
            ROMA Executor - Specialized Research:
            
            TASK: {goal}
            TOOLS AVAILABLE: {', '.join(tools_needed)}
            CONTEXT: {context_data}
            
            Your role: Conduct specialized research on the given topic.
            
            Instructions:
            1. Use domain expertise to gather relevant information
            2. Apply critical analysis to sources
            3. Focus on accuracy and relevance
            4. Provide comprehensive research findings
            
            Deliver thorough research results with source documentation.
            """
            
        else:
            return f"""
            ROMA Executor - General Task Execution:
            
            TASK: {goal}
            TOOLS AVAILABLE: {', '.join(tools_needed)}
            CONTEXT: {context_data}
            
            Execute this task using your best judgment and available tools.
            Provide clear, comprehensive results.
            """
    
    def _process_response(self, response: str) -> str | list | dict:
        if not response or not response.strip():
            return ""
        
        agent_type = self.subtask_data.get("agent_type", "general")
        
        if agent_type == "retrieval":
            return self._extract_sources(response)
        elif agent_type == "extraction":
            # Split into facts by lines/bullets
            lines = [ln.strip("- •* \t") for ln in response.split('\n')]
            facts = [ln for ln in lines if ln]
            return facts
        elif agent_type == "analysis":
            return response.strip()
        elif agent_type == "verification":
            return self._extract_verification_data(response)
        else:
            return response.strip()
    
    def _extract_sources(self, response: str) -> list[dict]:
        lines = response.split('\n')
        sources = []
        current_source = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith(('http://', 'https://', 'Source:')):
                if current_source:
                    sources.append(current_source)
                current_source = {"url": line, "content": "", "title": ""}
            elif current_source:
                current_source["content"] += line + " "
        
        if current_source:
            sources.append(current_source)
        
        if not sources:
            sources = [{"title": "Research Results", "content": response, "url": ""}]
        
        return sources
    
    def _extract_verification_data(self, response: str) -> dict[str, float | str | list]:
        coverage = 0.8
        confidence = 0.9
        
        if "high confidence" in response.lower():
            confidence = 0.95
        elif "low confidence" in response.lower():
            confidence = 0.6
            
        if "complete" in response.lower():
            coverage = 0.95
        elif "partial" in response.lower():
            coverage = 0.7
        
        return {
            "coverage": coverage,
            "confidence": confidence,
            "verification_details": response.strip(),
            "verified_claims": [],
            "concerns": []
        }
    
    async def _execute_with_sub_agents(self) -> NodeResult:
        """Execute complex task using formal SubAgentSpawner interface"""
        
        start_time = time.time()
        
        # Create sub-agent requests using the spawner interface
        goal = self.subtask_data.get("goal", "")
        sub_agent_requests = self.spawner.create_sub_agent_requests(goal, self.context)
        # Sub-agents spawned
        
        try:
            # Execute sub-agents using the spawner
            sub_results = await self.spawner.spawn_sub_agents(sub_agent_requests, self.context)
            elapsed_time = time.time() - start_time
            
            # Aggregate sub-results
            successful_results = [r for r in sub_results if r.status == ResultStatus.SUCCESS]
            total_tokens = sum(r.tokens_used for r in sub_results)
            
            # Determine overall status
            if len(successful_results) == len(sub_results):
                overall_status = ResultStatus.SUCCESS
                output = self._synthesize_sub_agent_outputs(sub_results, goal)
                error_message = None
            elif successful_results:
                overall_status = ResultStatus.PARTIAL  
                output = self._synthesize_sub_agent_outputs(successful_results, goal)
                error_message = f"Only {len(successful_results)}/{len(sub_results)} sub-agents succeeded"
            else:
                overall_status = ResultStatus.FAILED
                output = ""
                error_message = "All sub-agents failed"
            
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=overall_status,
                output=output,
                error_message=error_message,
                tokens_used=total_tokens,
                seconds_elapsed=elapsed_time,
                child_results=sub_results,
                trace_data={
                    "recursive_execution": True,
                    "sub_agents_count": len(sub_agent_requests),
                    "successful_sub_agents": len(successful_results),
                    "execution_method": "formal_spawner_interface"
                }
            )
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=ResultStatus.FAILED,
                output="",
                error_message=f"Sub-agent spawning failed: {str(e)}",
                tokens_used=0,
                seconds_elapsed=elapsed_time,
                child_results=[],
                trace_data={"recursive_execution_failed": True, "error": str(e)}
            )
    
    def _synthesize_sub_agent_outputs(self, results: list[NodeResult], original_goal: str) -> str | list | dict:
        """Synthesize outputs from multiple sub-agents into a coherent result"""
        
        if not results:
            return ""
        
        if len(results) == 1:
            return results[0].output
        
        # For multiple results, create a structured synthesis
        synthesis = {
            "original_goal": original_goal,
            "sub_agent_results": [],
            "summary": ""
        }
        
        combined_content = []
        for result in results:
            synthesis["sub_agent_results"].append({
                "task_id": result.node_id,
                "status": result.status.value,
                "output": result.output,
                "tokens_used": result.tokens_used
            })
            
            if isinstance(result.output, str) and result.output.strip():
                combined_content.append(result.output.strip())
            elif isinstance(result.output, (list, dict)):
                combined_content.append(str(result.output))
        
        synthesis["summary"] = " ".join(combined_content)
        return synthesis
