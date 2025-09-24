from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

from hsw.models import TaskContext, NodeResult, NodeType, NodeState, ResultStatus
import weave

class ROMANode(BaseModel, ABC):
    node_id: str
    node_type: NodeType
    state: NodeState
    context: TaskContext
    parent_node: ROMANode | None = None
    child_nodes: list[ROMANode] = Field(default_factory=list)
    agent: object | None = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @abstractmethod
    async def execute(self) -> NodeResult:
        pass
    
    @weave.op
    async def process(self) -> NodeResult:
        try:
            result = await self.execute()
            self.state = NodeState.COMPLETED if result.status == ResultStatus.SUCCESS else NodeState.FAILED
            return result
        except Exception as e:
            self.state = NodeState.FAILED
            return NodeResult(
                node_id=self.node_id,
                node_type=self.node_type.value,
                status=ResultStatus.FAILED,
                output="",
                error_message=str(e),
                tokens_used=0,
                seconds_elapsed=0.0,
                child_results=[]
            )
    
    def add_child(self, child: ROMANode) -> None:
        child.parent_node = self
        child.context.parent_id = self.node_id
        child.context.depth = self.context.depth + 1
        self.child_nodes.append(child)

ROMANode.model_rebuild()
