import time
import weave
import logging
from agents import Agent
from hsw.nodes.roma_node import ROMANode
from hsw.models import TaskContext, NodeResult, NodeType, NodeState, ResultStatus, ComplexityDecision
from hsw.workers.agents import stream_agent_response

logger = logging.getLogger(__name__)

class AtomizerNode(ROMANode):
    
    def __init__(self, context: TaskContext, agent: Agent):
        super().__init__(
            node_id=f"atomizer_{context.node_id}",
            node_type=NodeType.ATOMIZER,
            state=NodeState.CREATED,
            context=context,
            agent=agent
        )
    
    @weave.op
    async def execute(self) -> NodeResult:
        logger.info(f"🔍 ATOMIZER: Analyzing task complexity - '{self.context.goal}' (depth {self.context.depth}/{self.context.max_depth})")
        self.state = NodeState.ANALYZING
        start_time = time.time()
        
        analysis_prompt = f"""
        ROMA Atomizer Decision:
        
        TASK: {self.context.goal}
        CURRENT DEPTH: {self.context.depth}
        MAX DEPTH: {self.context.max_depth}
        AVAILABLE TOOLS: {', '.join(self.context.tools_available)}
        AGENT CAPABILITIES: {', '.join(self.context.agent_capabilities)}
        
        DECISION REQUIRED:
        Determine if this task is:
        1. SIMPLE - Can be executed directly by a single agent/tool
        2. COMPLEX - Needs decomposition into subtasks
        
        Consider:
        - Task complexity and scope
        - Available tools and capabilities  
        - Current recursion depth vs max depth
        - Whether subtasks would improve accuracy
        
        Respond with JSON:
        {{
            "decision": ComplexityDecision.SIMPLE.value or ComplexityDecision.COMPLEX.value,
            "reasoning": "explanation of why",
            "confidence": 0.0-1.0,
            "estimated_subtasks": number_if_complex,
            "recommended_approach": "description"
        }}
        """
        
        # Decision logged via weave tracing
        
        tokens_used, response = await stream_agent_response(self.agent, analysis_prompt)
        elapsed_time = time.time() - start_time
        
        try:
            decision_data = self._parse_decision(response)
            decision = decision_data.get("decision", "unknown")
            reasoning = decision_data.get("reasoning", "No reasoning provided")
            
            logger.info(f"✅ ATOMIZER: Decision = {decision}")
            logger.info(f"💭 ATOMIZER: Reasoning = {reasoning}")
            
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=ResultStatus.SUCCESS,
                output=decision_data,
                error_message=None,
                tokens_used=tokens_used,
                seconds_elapsed=elapsed_time,
                child_results=[],
            )
            
        except Exception as e:
            # Default safely to COMPLEX and treat as success to continue pipeline
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=ResultStatus.SUCCESS,
                output={"decision": ComplexityDecision.COMPLEX.value, "reasoning": f"Defaulted due to parse error: {str(e)}"},
                error_message=None,
                tokens_used=tokens_used,
                seconds_elapsed=elapsed_time,
                child_results=[],
                trace_data={"fallback_used": True}
            )
    
    def _parse_decision(self, response: str) -> dict[str, str | float | int]:
        import json
        import re
        
        json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        if ComplexityDecision.SIMPLE.value in response.upper():
            return {
                "decision": ComplexityDecision.SIMPLE.value,
                "reasoning": "Task appears to be simple enough for direct execution",
                "confidence": 0.7,
                "estimated_subtasks": 0,
                "recommended_approach": "Direct execution"
            }
        else:
            return {
                "decision": ComplexityDecision.COMPLEX.value, 
                "reasoning": "Task requires decomposition into subtasks",
                "confidence": 0.8,
                "estimated_subtasks": 3,
                "recommended_approach": "Break into subtasks"
            }
