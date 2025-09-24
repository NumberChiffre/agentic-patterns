import time
import logging
import weave
from agents import Agent
from pydantic import Field
from hsw.nodes.roma_node import ROMANode
from hsw.models import TaskContext, NodeResult, NodeType, NodeState, ResultStatus
from hsw.workers.agents import stream_agent_response

logger = logging.getLogger(__name__)

class AggregatorNode(ROMANode):
    child_results: list[NodeResult] = Field(default_factory=list)
    
    def __init__(self, context: TaskContext, agent: Agent, child_results: list[NodeResult]):
        super().__init__(
            node_id=f"aggregator_{context.node_id}",
            node_type=NodeType.AGGREGATOR,
            state=NodeState.CREATED,
            context=context,
            agent=agent
        )
        self.child_results = child_results
    
    @weave.op
    async def execute(self) -> NodeResult:
        successful_count = len([r for r in self.child_results if r.status == ResultStatus.SUCCESS])
        total_count = len(self.child_results)
        
        logger.info(f"🔄 AGGREGATOR: Synthesizing {successful_count}/{total_count} successful results for '{self.context.goal}'")
        logger.info("📊 AGGREGATOR: Child results summary:")
        for i, result in enumerate(self.child_results):
            status_emoji = "✅" if result.status == ResultStatus.SUCCESS else "❌"
            logger.info(f"   {i+1}. {result.node_id} {status_emoji} - {result.node_type}")
        
        self.state = NodeState.AGGREGATING
        start_time = time.time()
        
        aggregation_prompt = self._build_aggregation_prompt()
        
        tokens_used, response = await stream_agent_response(self.agent, aggregation_prompt)
        elapsed_time = time.time() - start_time
        
        final_output = self._synthesize_final_answer(response)
        
        logger.info(f"✅ AGGREGATOR: Synthesis complete ({tokens_used} tokens, {elapsed_time:.1f}s)")
        
        return NodeResult(
            node_id=self.node_id,
            node_type=self.node_type.value,
            status=ResultStatus.SUCCESS,
            output=final_output,
            error_message=None,
            tokens_used=tokens_used,
            seconds_elapsed=elapsed_time,
            child_results=self.child_results,
            trace_data={
                "aggregation_strategy": "comprehensive_synthesis",
                "child_results_processed": len(self.child_results),
                "final_output_length": len(str(final_output))
            }
        )
    
    def _build_aggregation_prompt(self) -> str:
        child_summaries = []
        
        for result in self.child_results:
            status_indicator = "✅" if result.status == ResultStatus.SUCCESS else "❌"
            summary = {
                "node_id": result.node_id,
                "type": result.node_type, 
                "status": status_indicator,
                "output_preview": str(result.output),
                "tokens_used": result.tokens_used
            }
            child_summaries.append(summary)
        
        return f"""
        ROMA Aggregator - Result Synthesis:
        
        ORIGINAL GOAL: {self.context.goal}
        
        CHILD RESULTS TO AGGREGATE:
        {chr(10).join([f"- {s['type']} ({s['status']}): {s['output_preview']}" for s in child_summaries])}
        
        TASK: Synthesize these results into a comprehensive, final answer.
        
        Instructions:
        1. Analyze all successful child results
        2. Identify key findings and insights from each
        3. Resolve any conflicts or inconsistencies
        4. Synthesize into a coherent, complete answer
        5. Ensure the final answer directly addresses the original goal
        6. Include confidence assessment and source attribution where applicable
        
        SYNTHESIS REQUIREMENTS:
        - Address the original goal comprehensively
        - Integrate insights from all successful child results
        - Provide clear, actionable conclusions
        - Maintain accuracy and avoid hallucination
        - Include reasoning for conclusions
        
        Deliver the final synthesized answer:
        """
    
    def _synthesize_final_answer(self, response: str) -> dict[str, str | float | list]:
        successful_results = [r for r in self.child_results if r.status == ResultStatus.SUCCESS]
        total_tokens = sum(r.tokens_used for r in self.child_results)
        total_time = sum(r.seconds_elapsed for r in self.child_results)
        
        confidence_score = self._calculate_confidence()
        completeness_score = len(successful_results) / len(self.child_results) if self.child_results else 0.0
        
        return {
            "final_answer": response.strip(),
            "synthesis_quality": {
                "confidence": confidence_score,
                "completeness": completeness_score,
                "source_count": len(self.child_results),
                "successful_subtasks": len(successful_results)
            },
            "execution_summary": {
                "total_tokens_used": total_tokens,
                "total_time_elapsed": total_time,
                "subtasks_executed": len(self.child_results),
                "success_rate": completeness_score
            },
            "child_contributions": [
                {
                    "node_id": r.node_id,
                    "node_type": r.node_type,
                    "contribution": "success" if r.status == ResultStatus.SUCCESS else "failed",
                    "key_insight": str(r.output)
                }
                for r in self.child_results
            ]
        }
    
    def _calculate_confidence(self) -> float:
        successful_results = [r for r in self.child_results if r.status == ResultStatus.SUCCESS]
        
        if not successful_results:
            return 0.0
        
        base_confidence = 0.7
        success_bonus = (len(successful_results) / len(self.child_results)) * 0.2
        depth_bonus = max(0, (self.context.max_depth - self.context.depth) / self.context.max_depth) * 0.1
        
        return min(1.0, base_confidence + success_bonus + depth_bonus)
