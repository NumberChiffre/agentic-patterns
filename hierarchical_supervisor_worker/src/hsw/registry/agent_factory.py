from agents import Agent
from enum import Enum
from pydantic import BaseModel
from hsw.registry.tool_registry import get_tool_registry

class AgentCapability(Enum):
    RESEARCH = "research"
    ANALYSIS = "analysis"
    EXTRACTION = "extraction"
    VERIFICATION = "verification"
    LEGAL_RESEARCH = "legal_research"
    MEDICAL_RESEARCH = "medical_research"
    TECHNICAL_ANALYSIS = "technical_analysis"
    BUSINESS_ANALYSIS = "business_analysis"
    CODE_ANALYSIS = "code_analysis"
    DATA_PROCESSING = "data_processing"

class AgentSpec(BaseModel):
    name: str
    capabilities: list[AgentCapability]
    instructions: str
    tools: list[str]
    model: str = "gpt-4o-mini"

class DynamicAgentFactory:
    
    def __init__(self):
        self.agent_cache = {}  # capability_signature -> Agent instance
        self.tool_registry = get_tool_registry()
        self.capability_templates = {
            AgentCapability.RESEARCH: {
                "instructions": "You are a research specialist. Search and gather comprehensive information from reliable sources. Focus on factual accuracy and source attribution.",
                "tools": ["WebSearchTool"],
                "model": "gpt-4o-mini"
            },
            AgentCapability.ANALYSIS: {
                "instructions": "You are an analysis specialist. Synthesize information into coherent insights with clear reasoning and evidence-based conclusions.",
                "tools": [],
                "model": "gpt-4o"
            },
            AgentCapability.EXTRACTION: {
                "instructions": "You are an extraction specialist. Extract key facts, entities, and structured data from provided sources with high precision.",
                "tools": [],
                "model": "gpt-4o-mini"
            },
            AgentCapability.VERIFICATION: {
                "instructions": "You are a verification specialist. Check claims against sources and assess coverage, confidence, and accuracy.",
                "tools": [],
                "model": "gpt-4o-mini"
            },
            AgentCapability.LEGAL_RESEARCH: {
                "instructions": "You are a legal research specialist. Focus on legal precedents, regulations, and compliance requirements with proper citation.",
                "tools": ["WebSearchTool"],
                "model": "gpt-4o"
            },
            AgentCapability.MEDICAL_RESEARCH: {
                "instructions": "You are a medical research specialist. Focus on evidence-based medicine, clinical guidelines, and peer-reviewed research.",
                "tools": ["WebSearchTool"],
                "model": "gpt-4o"
            },
            AgentCapability.TECHNICAL_ANALYSIS: {
                "instructions": "You are a technical analysis specialist. Analyze technical systems, architectures, and implementation details.",
                "tools": [],
                "model": "gpt-4o"
            },
            AgentCapability.BUSINESS_ANALYSIS: {
                "instructions": "You are a business analysis specialist. Focus on market dynamics, business models, and strategic implications.",
                "tools": ["WebSearchTool"],
                "model": "gpt-4o-mini"
            }
        }
    
    def create_agent_from_capabilities(self, capabilities: list[AgentCapability], task_context: str = "") -> Agent:
        if not capabilities:
            capabilities = [AgentCapability.ANALYSIS]
        
        # Create capability signature for caching
        capability_signature = self._create_capability_signature(capabilities, task_context)
        
        # Check cache first
        if capability_signature in self.agent_cache:
            return self.agent_cache[capability_signature]
        
        primary_capability = capabilities[0]
        template = self.capability_templates[primary_capability]
        
        # Combine instructions from multiple capabilities
        instructions = template["instructions"]
        if len(capabilities) > 1:
            additional_caps = [cap.value for cap in capabilities[1:]]
            instructions += f" Additionally, you have expertise in: {', '.join(additional_caps)}."
        
        if task_context:
            instructions += f" Context: {task_context}"
        
        # Combine tools from all capabilities
        all_tools = set(template["tools"])
        for cap in capabilities[1:]:
            if cap in self.capability_templates:
                all_tools.update(self.capability_templates[cap]["tools"])
        
        # Get additional tool recommendations from registry
        combined_context = f"{' '.join([cap.value for cap in capabilities])} {task_context}"
        recommended_tools = self.tool_registry.recommend_tools_for_task(combined_context, [cap.value for cap in capabilities])
        
        # Merge template tools with recommendations
        all_tools.update(recommended_tools)
        
        # Create tool instances using registry
        tool_instances = self.tool_registry.create_agent_tools(list(all_tools), "balanced")
        
        agent_name = f"{'_'.join([cap.value for cap in capabilities])}_agent"
        
        agent = Agent(
            name=agent_name,
            model=template["model"],
            instructions=instructions,
            tools=tool_instances
        )
        
        # Cache the agent
        self.agent_cache[capability_signature] = agent
        return agent
    
    def _create_capability_signature(self, capabilities: list[AgentCapability], task_context: str = "") -> str:
        """Create a unique signature for capability combinations"""
        # Sort capabilities to ensure consistent signatures
        sorted_caps = sorted([cap.value for cap in capabilities])
        
        # Create signature from capabilities and context hash
        caps_str = "|".join(sorted_caps)
        
        # Include a hash of task context for uniqueness, but limit context impact on caching
        # Only use context if it significantly changes the agent behavior
        context_hash = ""
        if task_context and len(task_context) > 50:  # Only cache-differentiate for substantial context
            import hashlib
            context_hash = hashlib.md5(task_context.encode()).hexdigest()[:8]
        
        signature = f"{caps_str}#{context_hash}" if context_hash else caps_str
        return signature
    
    def get_cache_stats(self) -> dict[str, int | list]:
        """Get cache statistics for monitoring"""
        return {
            "cached_agents": len(self.agent_cache),
            "cache_keys": list(self.agent_cache.keys())
        }
    
    def clear_cache(self) -> None:
        """Clear the agent cache"""
        self.agent_cache.clear()
    
    def create_strategy_aware_agent(
        self, 
        capabilities: list[AgentCapability], 
        task_context: str, 
        execution_strategy: str
    ) -> Agent:
        """Create an agent optimized for a specific execution strategy"""
        
        # Get recommended tools from registry with strategy optimization
        recommended_tools = self.tool_registry.recommend_tools_for_task(task_context, [cap.value for cap in capabilities])
        optimized_tools = self.tool_registry.optimize_tool_selection(recommended_tools, execution_strategy)
        
        # Create capability signature that includes strategy
        capability_signature = self._create_capability_signature(capabilities, f"{task_context}_{execution_strategy}")
        
        # Check cache first
        if capability_signature in self.agent_cache:
            return self.agent_cache[capability_signature]
        
        primary_capability = capabilities[0] if capabilities else AgentCapability.ANALYSIS
        template = self.capability_templates[primary_capability]
        
        # Adjust model based on strategy
        model = template["model"]
        if execution_strategy == "speed_optimized":
            model = "gpt-4o-mini"
        elif execution_strategy == "quality_optimized":
            model = "gpt-4o"
        elif execution_strategy == "cost_optimized":
            model = "gpt-4o-mini"
        
        # Build strategy-aware instructions
        instructions = template["instructions"]
        if len(capabilities) > 1:
            additional_caps = [cap.value for cap in capabilities[1:]]
            instructions += f" Additionally, you have expertise in: {', '.join(additional_caps)}."
        
        if task_context:
            instructions += f" Context: {task_context}"
        
        # Add strategy-specific guidance
        if execution_strategy == "speed_optimized":
            instructions += " Prioritize speed and efficiency. Provide concise, actionable responses."
        elif execution_strategy == "quality_optimized":
            instructions += " Prioritize accuracy and thoroughness. Take time to provide comprehensive, well-researched responses."
        elif execution_strategy == "cost_optimized":
            instructions += " Be efficient with token usage while maintaining quality. Focus on essential information."
        
        # Create tool instances using registry with strategy
        tool_instances = self.tool_registry.create_agent_tools(optimized_tools, execution_strategy)
        
        agent_name = f"{'_'.join([cap.value for cap in capabilities])}_agent_{execution_strategy}"
        
        agent = Agent(
            name=agent_name,
            model=model,
            instructions=instructions,
            tools=tool_instances
        )
        
        # Cache the agent
        self.agent_cache[capability_signature] = agent
        return agent
    
    def get_factory_stats(self) -> dict[str, int | dict]:
        """Get comprehensive factory statistics"""
        cache_stats = self.get_cache_stats()
        registry_stats = self.tool_registry.get_registry_stats()
        
        return {
            "agent_cache": cache_stats,
            "tool_registry": registry_stats,
            "capability_templates": len(self.capability_templates)
        }
    
    def create_agent_from_spec(self, spec: AgentSpec) -> Agent:
        # Create signature from spec for caching
        spec_signature = f"spec_{spec.name}_{spec.model}_{hash(spec.instructions)}"
        
        # Check cache first
        if spec_signature in self.agent_cache:
            return self.agent_cache[spec_signature]
        
        # Create tool instances using registry
        tool_instances = self.tool_registry.create_agent_tools(spec.tools, "balanced")
        
        agent = Agent(
            name=spec.name,
            model=spec.model,
            instructions=spec.instructions,
            tools=tool_instances
        )
        
        # Cache the agent
        self.agent_cache[spec_signature] = agent
        return agent
    
    def infer_capabilities_from_task(self, task_goal: str, agent_type: str) -> list[AgentCapability]:
        goal_lower = task_goal.lower()
        
        # Medical/health keywords
        if any(word in goal_lower for word in ["uti", "medical", "clinical", "treatment", "diagnosis", "healthcare", "patient", "evidence"]):
            if agent_type == "retrieval":
                return [AgentCapability.MEDICAL_RESEARCH, AgentCapability.RESEARCH]
            elif agent_type == "analysis":
                return [AgentCapability.MEDICAL_RESEARCH, AgentCapability.ANALYSIS]
        
        # Legal keywords
        if any(word in goal_lower for word in ["legal", "law", "regulation", "compliance", "patent", "contract"]):
            if agent_type == "retrieval":
                return [AgentCapability.LEGAL_RESEARCH, AgentCapability.RESEARCH]
            elif agent_type == "analysis":
                return [AgentCapability.LEGAL_RESEARCH, AgentCapability.ANALYSIS]
        
        # Technical keywords
        if any(word in goal_lower for word in ["code", "technical", "architecture", "system", "algorithm", "software"]):
            if agent_type == "analysis":
                return [AgentCapability.TECHNICAL_ANALYSIS, AgentCapability.ANALYSIS]
        
        # Business keywords
        if any(word in goal_lower for word in ["market", "business", "strategy", "financial", "economic"]):
            if agent_type == "retrieval":
                return [AgentCapability.BUSINESS_ANALYSIS, AgentCapability.RESEARCH]
            elif agent_type == "analysis":
                return [AgentCapability.BUSINESS_ANALYSIS, AgentCapability.ANALYSIS]
        
        # Default mapping
        if agent_type == "retrieval":
            return [AgentCapability.RESEARCH]
        elif agent_type == "extraction":
            return [AgentCapability.EXTRACTION]
        elif agent_type == "analysis":
            return [AgentCapability.ANALYSIS]
        elif agent_type == "verification":
            return [AgentCapability.VERIFICATION]
        else:
            return [AgentCapability.ANALYSIS]

def create_dynamic_agent_factory() -> DynamicAgentFactory:
    return DynamicAgentFactory()
