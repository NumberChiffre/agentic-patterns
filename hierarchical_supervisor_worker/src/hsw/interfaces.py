from abc import ABC, abstractmethod
from pydantic import BaseModel
from hsw.models import NodeResult, TaskContext
from hsw.registry.agent_factory import AgentCapability

class SubAgentRequest(BaseModel):
    task_id: str
    goal: str
    capabilities: list[AgentCapability]
    tools_needed: list[str] = []
    context_keys: list[str] = []
    estimated_tokens: int = 1000
    estimated_seconds: int = 10
    depends_on: list[str] = []
    can_run_parallel: bool = False

class SubAgentSpawner(ABC):
    """Formal interface for spawning sub-agents in ROMA tree execution"""
    
    @abstractmethod
    async def spawn_sub_agents(
        self, 
        requests: list[SubAgentRequest], 
        parent_context: TaskContext
    ) -> list[NodeResult]:
        """
        Spawn and execute sub-agents for complex tasks
        
        Args:
            requests: List of sub-agent requests defining tasks and requirements
            parent_context: Parent task context for budget inheritance and depth tracking
            
        Returns:
            List of NodeResults from sub-agent execution
        """
        pass
    
    @abstractmethod
    def should_spawn_sub_agents(self, task_complexity: dict) -> bool:
        """
        Determine if a task is complex enough to warrant sub-agent spawning
        
        Args:
            task_complexity: Dictionary with complexity indicators
            
        Returns:
            True if sub-agents should be spawned, False otherwise
        """
        pass
    
    @abstractmethod
    def create_sub_agent_requests(self, goal: str, context: TaskContext) -> list[SubAgentRequest]:
        """
        Analyze a goal and create appropriate sub-agent requests
        
        Args:
            goal: The complex goal to decompose
            context: Current task context
            
        Returns:
            List of sub-agent requests
        """
        pass

class SubAgentCoordinator(ABC):
    """Interface for coordinating sub-agent execution patterns"""
    
    @abstractmethod
    async def execute_parallel(
        self, 
        requests: list[SubAgentRequest], 
        context: TaskContext
    ) -> list[NodeResult]:
        """Execute independent sub-agents in parallel"""
        pass
    
    @abstractmethod 
    async def execute_sequential(
        self, 
        requests: list[SubAgentRequest], 
        context: TaskContext
    ) -> list[NodeResult]:
        """Execute dependent sub-agents sequentially"""
        pass
    
    @abstractmethod
    async def execute_hybrid(
        self, 
        requests: list[SubAgentRequest], 
        context: TaskContext
    ) -> list[NodeResult]:
        """Execute sub-agents with mixed parallel/sequential patterns based on dependencies"""
        pass

