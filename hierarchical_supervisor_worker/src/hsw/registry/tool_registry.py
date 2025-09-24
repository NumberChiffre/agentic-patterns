from abc import ABC, abstractmethod
from pydantic import BaseModel
from enum import Enum
from agents import WebSearchTool

class ToolCategory(Enum):
    SEARCH = "search"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    CALCULATION = "calculation"
    FILE_OPERATION = "file_operation"
    API_INTEGRATION = "api_integration"

class ToolSpec(BaseModel):
    name: str
    category: ToolCategory
    description: str
    required_params: list[str] = []
    optional_params: list[str] = []
    output_type: str = "text"
    cost_tier: str = "low"  # low, medium, high
    
class BaseTool(ABC):
    """Base interface for all tools in the registry"""
    
    @abstractmethod
    def get_spec(self) -> ToolSpec:
        """Return the tool specification"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> dict | str | list:
        pass
    
    @abstractmethod
    def validate_params(self, **kwargs) -> bool:
        """Validate parameters before execution"""
        pass

class WebSearchToolWrapper(BaseTool):
    """Wrapper for WebSearchTool to conform to BaseTool interface"""
    
    def __init__(self):
        self.tool = WebSearchTool()
    
    def get_spec(self) -> ToolSpec:
        return ToolSpec(
            name="WebSearchTool",
            category=ToolCategory.SEARCH,
            description="Search the web for information using search engines",
            required_params=["query"],
            optional_params=["max_results", "region"],
            output_type="list[dict]",
            cost_tier="medium"
        )
    
    async def execute(self, **kwargs) -> str:
        # WebSearchTool from agents package doesn't have async execute
        # This is a placeholder - actual implementation would depend on the tool's API
        query = kwargs.get("query", "")
        if not query:
            raise ValueError("Query parameter is required")
        return f"Search results for: {query}"
    
    def validate_params(self, **kwargs) -> bool:
        return "query" in kwargs and isinstance(kwargs["query"], str) and kwargs["query"].strip()

class AgentToolRegistry:
    """Registry for managing available tools and their assignment to agents"""
    
    def __init__(self):
        self.tools = {}  # tool_name -> BaseTool instance
        self.categories = {}  # category -> list[tool_name]
        self.agent_tool_cache = {}  # agent_signature -> list[tool_instances]
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register default tools available in the system"""
        web_search = WebSearchToolWrapper()
        self.register_tool(web_search)
    
    def register_tool(self, tool: BaseTool) -> None:
        """Register a new tool in the registry"""
        spec = tool.get_spec()
        self.tools[spec.name] = tool
        
        # Update category mapping
        if spec.category not in self.categories:
            self.categories[spec.category] = []
        self.categories[spec.category].append(spec.name)
    
    def unregister_tool(self, tool_name: str) -> None:
        """Remove a tool from the registry"""
        if tool_name in self.tools:
            tool_spec = self.tools[tool_name].get_spec()
            del self.tools[tool_name]
            
            # Update category mapping
            if tool_spec.category in self.categories:
                if tool_name in self.categories[tool_spec.category]:
                    self.categories[tool_spec.category].remove(tool_name)
                    if not self.categories[tool_spec.category]:
                        del self.categories[tool_spec.category]
    
    def get_tool(self, tool_name: str) -> BaseTool | None:
        """Get a tool by name"""
        return self.tools.get(tool_name)
    
    def get_tools_by_category(self, category: ToolCategory) -> list[BaseTool]:
        """Get all tools in a specific category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]
    
    def get_available_tools(self) -> list[ToolSpec]:
        """Get specifications for all available tools"""
        return [tool.get_spec() for tool in self.tools.values()]
    
    def recommend_tools_for_task(self, task_description: str, capabilities: list[str] = []) -> list[str]:
        """Recommend tools based on task description and agent capabilities"""
        task_lower = task_description.lower()
        recommended_tools = []
        
        # Search-based recommendations
        if any(keyword in task_lower for keyword in ["search", "find", "research", "web", "online", "information"]):
            search_tools = self.get_tools_by_category(ToolCategory.SEARCH)
            recommended_tools.extend([tool.get_spec().name for tool in search_tools])
        
        # Analysis-based recommendations
        if any(keyword in task_lower for keyword in ["analyze", "process", "compute", "calculate"]):
            analysis_tools = self.get_tools_by_category(ToolCategory.ANALYSIS)
            calculation_tools = self.get_tools_by_category(ToolCategory.CALCULATION)
            recommended_tools.extend([tool.get_spec().name for tool in analysis_tools + calculation_tools])
        
        # Data processing recommendations
        if any(keyword in task_lower for keyword in ["data", "file", "process", "transform", "extract"]):
            data_tools = self.get_tools_by_category(ToolCategory.DATA_PROCESSING)
            file_tools = self.get_tools_by_category(ToolCategory.FILE_OPERATION)
            recommended_tools.extend([tool.get_spec().name for tool in data_tools + file_tools])
        
        # Remove duplicates and return
        return list(set(recommended_tools))
    
    def create_agent_tools(self, tool_names: list[str], execution_strategy: str = "balanced") -> list[object]:
        """Create tool instances for an agent based on execution strategy"""
        agent_signature = f"{sorted(tool_names)}_{execution_strategy}"
        
        # Check cache first
        if agent_signature in self.agent_tool_cache:
            return self.agent_tool_cache[agent_signature]
        
        tool_instances = []
        
        for tool_name in tool_names:
            if tool_name in self.tools:
                # For now, create the original tool instance
                # In a full implementation, this would create strategy-aware tool configs
                if tool_name == "WebSearchTool":
                    tool_instances.append(WebSearchTool())
        
        # Cache the instances
        self.agent_tool_cache[agent_signature] = tool_instances
        return tool_instances
    
    def validate_tool_combination(self, tool_names: list[str]) -> dict[str, bool | list[str]]:
        """Validate if a combination of tools is valid and identify conflicts"""
        validation_result = {
            "valid": True,
            "missing_tools": [],
            "conflicts": [],
            "recommendations": []
        }
        
        # Check if all tools exist
        for tool_name in tool_names:
            if tool_name not in self.tools:
                validation_result["missing_tools"].append(tool_name)
                validation_result["valid"] = False
        
        # Check for tool conflicts (placeholder for future conflict detection)
        # In a real implementation, you might have rules about incompatible tools
        
        return validation_result
    
    def get_tool_costs(self, tool_names: list[str]) -> dict[str, str]:
        """Get cost information for tools"""
        costs = {}
        for tool_name in tool_names:
            if tool_name in self.tools:
                spec = self.tools[tool_name].get_spec()
                costs[tool_name] = spec.cost_tier
        return costs
    
    def optimize_tool_selection(
        self, 
        recommended_tools: list[str], 
        execution_strategy: str = "balanced",
        max_tools: int = 5
    ) -> list[str]:
        """Optimize tool selection based on execution strategy"""
        
        if len(recommended_tools) <= max_tools:
            return recommended_tools
        
        # Get tool specifications
        tool_specs = []
        for tool_name in recommended_tools:
            if tool_name in self.tools:
                tool_specs.append((tool_name, self.tools[tool_name].get_spec()))
        
        # Strategy-based selection
        if execution_strategy == "cost_optimized":
            # Prefer low-cost tools
            tool_specs.sort(key=lambda x: {"low": 0, "medium": 1, "high": 2}.get(x[1].cost_tier, 1))
        elif execution_strategy == "quality_optimized":
            # Prefer high-quality tools (inverse of cost for simplicity)
            tool_specs.sort(key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x[1].cost_tier, 1))
        elif execution_strategy == "speed_optimized":
            # Prefer tools in search category for quick information gathering
            tool_specs.sort(key=lambda x: 0 if x[1].category == ToolCategory.SEARCH else 1)
        
        # Return top N tools
        selected_tools = [tool_name for tool_name, _ in tool_specs[:max_tools]]
        return selected_tools
    
    def get_registry_stats(self) -> dict[str, int | dict]:
        """Get registry statistics"""
        return {
            "total_tools": len(self.tools),
            "categories": {cat.value: len(tools) for cat, tools in self.categories.items()},
            "cached_agent_configs": len(self.agent_tool_cache)
        }

# Global registry instance
_tool_registry = None

def get_tool_registry() -> AgentToolRegistry:
    """Get the global tool registry instance"""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = AgentToolRegistry()
    return _tool_registry

def register_custom_tool(tool: BaseTool) -> None:
    """Register a custom tool globally"""
    registry = get_tool_registry()
    registry.register_tool(tool)

