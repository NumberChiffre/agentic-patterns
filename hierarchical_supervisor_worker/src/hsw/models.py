from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict

# Enums consolidated here
class NodeType(Enum):
    ATOMIZER = "atomizer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    AGGREGATOR = "aggregator"

class NodeState(Enum):
    CREATED = "created"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"

class ResultStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    PENDING = "pending"

# TraceEvent removed - replaced with weave tracing

class DependencyType(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    INDEPENDENT = "independent"

class ExecutionStrategy(Enum):
    SPEED_OPTIMIZED = "speed_optimized"    # Maximize parallel execution, use faster models
    COST_OPTIMIZED = "cost_optimized"      # Minimize token usage, use cheaper models  
    QUALITY_OPTIMIZED = "quality_optimized" # Focus on accuracy, use best models
    BALANCED = "balanced"                  # Balance speed, cost, and quality
    SEQUENTIAL = "sequential"              # Force sequential execution
    PARALLEL = "parallel"                  # Force parallel execution where possible

class WorkerType(Enum):
    RETRIEVAL = "retrieval"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    VERIFICATION = "verification"

class ComplexityDecision(Enum):
    SIMPLE = "SIMPLE"
    COMPLEX = "COMPLEX"

class ComplexityAssessment(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    MULTI_FACETED = "multi-faceted"

class SourceType(Enum):
    ACADEMIC = "academic"
    INDUSTRY = "industry"
    NEWS = "news"
    GOVERNMENT = "government"
    EXPERT_OPINION = "expert_opinion"

class CredibilityScore(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class RelevanceScore(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class BiasAssessment(Enum):
    NEUTRAL = "neutral"
    SLIGHT_BIAS = "slight_bias"
    STRONG_BIAS = "strong_bias"

# Structured Output Models for Agents
class SourceInfo(BaseModel):
    title: str
    url: str
    content: str
    source_type: SourceType
    publication_date: str
    credibility_score: CredibilityScore
    relevance_score: RelevanceScore
    key_topics: list[str]
    bias_assessment: BiasAssessment

class RetrievalOutput(BaseModel):
    sources: list[SourceInfo]
    search_summary: str
    total_sources_found: int
    search_strategy_used: str

class ExtractionOutput(BaseModel):
    key_facts: list[str]
    entities: list[str]
    main_themes: list[str]
    data_points: list[str]
    extraction_confidence: float

class Citation(BaseModel):
    source_title: str
    source_url: str
    quote: str
    page_number: str | None = None

class AnalysisOutput(BaseModel):
    executive_summary: str
    current_landscape: str
    key_insights: list[str]
    future_implications: list[str]
    critical_considerations: list[str]
    strategic_recommendations: list[str]
    conclusion: str
    citations: list[Citation]
    methodology: str

class VerificationOutput(BaseModel):
    accuracy_score: float
    completeness_score: float
    source_reliability: float
    fact_checks: list[str]
    gaps_identified: list[str]
    confidence_assessment: str
    recommendations: list[str]

class SubtaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class PromptTag(Enum):
    DESCRIPTION = "description"
    ROLE = "role"
    CAPABILITIES = "capabilities"
    TOOLS = "tools"
    GUIDELINES = "guidelines"
    CONSTRAINTS = "constraints"
    OUTPUT_FORMAT = "output_format"
    EXAMPLES = "examples"
    CONTEXT = "context"
    EVALUATION = "evaluation"

class Subtask(BaseModel):
    id: str
    goal: str
    worker_type: WorkerType
    inputs: dict[str, str | int | list | dict]
    max_tokens: int
    max_seconds: int
    status: SubtaskStatus = SubtaskStatus.PENDING
    children: list["Subtask"] = Field(default_factory=list)

class SubtaskResult(BaseModel):
    subtask_id: str
    success: bool
    output: str | list | dict | None
    error_message: str | None = None
    tokens_used: int = 0
    seconds_elapsed: float = 0.0

class Plan(BaseModel):
    root_goal: str
    subtasks: list[Subtask]
    max_depth: int
    total_budget_tokens: int
    total_budget_seconds: int

class SupervisorResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    goal: str
    plan: Plan
    results: dict[str, SubtaskResult]
    final_output: dict[str, str | list | dict]
    success: bool
    total_tokens: int
    total_seconds: float
    session_id: str = Field(default_factory=lambda: f"session_{datetime.now().timestamp()}")

# Context and tracing models consolidated here
class SubtaskDependency(BaseModel):
    task_id: str
    depends_on: list[str]
    dependency_type: DependencyType
    can_run_parallel: bool

class TaskContext(BaseModel):
    node_id: str
    goal: str
    parent_id: str | None
    depth: int
    max_depth: int
    subtasks: list[dict[str, str | list | dict]]
    dependencies: list[SubtaskDependency]
    context_data: dict[str, str | list | dict]
    tools_available: list[str]
    agent_capabilities: list[str]
    budget_tokens: int
    budget_seconds: int
    execution_strategy: ExecutionStrategy = ExecutionStrategy.BALANCED

# StageTrace and StageTracer removed - replaced with weave tracing

class NodeResult(BaseModel):
    node_id: str
    node_type: str
    status: ResultStatus
    output: str | list | dict
    error_message: str | None
    tokens_used: int
    seconds_elapsed: float
    child_results: list["NodeResult"] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True
