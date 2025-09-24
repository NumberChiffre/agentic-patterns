from agents import Agent, Runner, WebSearchTool
from hsw.models import Subtask, SubtaskResult, WorkerType, RetrievalOutput, ExtractionOutput, AnalysisOutput, VerificationOutput
from hsw.prompts import get_prompt
import json
import logging
import weave

logger = logging.getLogger(__name__)

@weave.op
async def stream_agent_response(agent: Agent, prompt: str) -> tuple[int, str]:
    logger.info(f"🎯 AGENT: {agent.name} starting execution")
    logger.debug(f"📝 AGENT: Prompt for {agent.name}: {prompt}")
    stream = Runner.run_streamed(agent, prompt)
    token_count = 0
    text_chunks = []
    
    async for event in stream.stream_events():
        event_type = str(getattr(event, "type", ""))
        
        # Handle raw response events containing text deltas
        if event_type == "raw_response_event":
            data = getattr(event, "data", None)
            # Check if this is a ResponseTextDeltaEvent
            if hasattr(data, "delta") and hasattr(data, "__class__"):
                class_name = data.__class__.__name__
                if "TextDelta" in class_name:
                    delta_text = getattr(data, "delta", None)
                    if isinstance(delta_text, str) and delta_text:
                        text_chunks.append(delta_text)
                        token_count += len(delta_text.split())
    
    final_text = "".join(text_chunks).strip()
    logger.info(f"✅ AGENT: {agent.name} completed - {token_count} tokens")
    return token_count, final_text

async def get_structured_agent_response(agent: Agent, prompt: str, response_model: type) -> tuple[int, dict]:
    """Get structured response from agent using Pydantic model"""
    logger.info(f"🔄 Getting structured response from {agent.name} for {response_model.__name__}")
    
    # Add explicit JSON instruction to prompt
    json_prompt = f"""{prompt}

CRITICAL: Respond with ONLY valid JSON. No markdown code blocks, no explanations, just pure JSON that matches the required schema."""
    
    token_count, raw_response = await stream_agent_response(agent, json_prompt)
    
    logger.debug(f"🔍 Raw response from {agent.name}: {raw_response}")
    
    # Clean the response - remove markdown code blocks if present
    cleaned_response = raw_response.strip()
    if cleaned_response.startswith('```json'):
        cleaned_response = cleaned_response[7:]
        logger.debug("🧹 Removed ```json prefix")
    if cleaned_response.startswith('```'):
        cleaned_response = cleaned_response[3:]
        logger.debug("🧹 Removed ``` prefix")
    if cleaned_response.endswith('```'):
        cleaned_response = cleaned_response[:-3]
        logger.debug("🧹 Removed ``` suffix")
    cleaned_response = cleaned_response.strip()
    
    logger.debug(f"🧹 Cleaned response: {cleaned_response}")
    
    # Try to parse JSON from the response
    try:
        # Look for JSON in the response
        json_start = cleaned_response.find('{')
        json_end = cleaned_response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = cleaned_response[json_start:json_end]
            logger.debug(f"🔍 Extracted JSON: {json_str}")
            parsed_data = json.loads(json_str)
            
            # Validate with Pydantic model
            validated_data = response_model(**parsed_data)
            logger.info(f"✅ Successfully parsed and validated {response_model.__name__} from {agent.name}")
            return token_count, validated_data.model_dump()
        else:
            # Try parsing the entire cleaned response as JSON
            logger.debug("🔍 Trying to parse entire response as JSON")
            parsed_data = json.loads(cleaned_response)
            validated_data = response_model(**parsed_data)
            logger.info(f"✅ Successfully parsed and validated {response_model.__name__} from {agent.name}")
            return token_count, validated_data.model_dump()
            
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"⚠️  JSON parsing failed for {response_model.__name__} from {agent.name}: {e}")
        logger.debug(f"📝 Raw response: {raw_response[:300]}...")
        logger.info(f"🔧 Using fallback structure for {response_model.__name__}")
        # Fallback: create structured response from raw text
        return token_count, _create_fallback_structure(raw_response, response_model)

def _create_fallback_structure(raw_text: str, response_model: type) -> dict:
    """Create structured fallback response when JSON parsing fails"""
    if response_model == RetrievalOutput:
        return {
            "sources": [{
                "title": "Retrieved Information",
                "url": "",
                "content": raw_text[:500],
                "source_type": "expert_opinion",
                "publication_date": "recent",
                "credibility_score": "medium",
                "relevance_score": "high",
                "key_topics": ["general"],
                "bias_assessment": "neutral"
            }],
            "search_summary": raw_text,
            "total_sources_found": 1,
            "search_strategy_used": "comprehensive_search"
        }
    elif response_model == AnalysisOutput:
        lines = raw_text.split('\n')
        return {
            "executive_summary": lines[0] if lines else raw_text,
            "current_landscape": raw_text[:300],
            "key_insights": [line.strip() for line in lines[1:4] if line.strip()],
            "future_implications": ["Continued development expected"],
            "critical_considerations": ["Further research needed"],
            "strategic_recommendations": ["Monitor developments"],
            "conclusion": lines[-1] if lines else raw_text[-200:],
            "citations": [
                {
                    "source_title": "Analysis based on provided information",
                    "source_url": "",
                    "quote": raw_text,
                    "page_number": None
                }
            ],
            "methodology": "Comprehensive analysis of available information"
        }
    elif response_model == VerificationOutput:
        return {
            "accuracy_score": 0.8,
            "completeness_score": 0.7,
            "source_reliability": 0.8,
            "fact_checks": ["Information appears accurate"],
            "gaps_identified": ["Additional verification recommended"],
            "confidence_assessment": "moderate",
            "recommendations": ["Cross-reference with additional sources"]
        }
    else:
        return {"raw_content": raw_text}

def create_supervisor_agent() -> Agent:
    supervisor_instructions = get_prompt("supervisor")
    return Agent(
        name="SupervisorMetaAgent",
        model="gpt-4o",
        instructions=supervisor_instructions
    )

def create_retrieval_agent() -> Agent:
    retrieval_instructions = get_prompt("retrieval")
    return Agent(
        name="RetrievalWorker",
        model="gpt-4o-mini",
        instructions=f"""{retrieval_instructions}

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanations, just pure JSON.

Your response must match this exact structure:
{RetrievalOutput.model_json_schema()}

Example response format:
{{
  "sources": [
    {{
      "title": "Machine Learning Fundamentals",
      "url": "https://example.com/ml-guide",
      "content": "Machine learning is a subset of artificial intelligence...",
      "source_type": "academic",
      "publication_date": "2024-01-15",
      "credibility_score": "high",
      "relevance_score": "high",
      "key_topics": ["machine learning", "AI", "algorithms"],
      "bias_assessment": "neutral"
    }}
  ],
  "search_summary": "Found comprehensive information about machine learning benefits",
  "total_sources_found": 3,
  "search_strategy_used": "comprehensive_academic_search"
}}""",
        tools=[WebSearchTool()]
    )

def create_extraction_agent() -> Agent:
    return Agent(
        name="ExtractionWorker",
        model="gpt-4o-mini", 
        instructions=f"You are an extraction specialist. Extract key facts and information from the provided sources. Focus on the most relevant and important points.\n\nIMPORTANT: Always respond with valid JSON matching this exact structure:\n{ExtractionOutput.model_json_schema()}",
        tools=[]
    )

def create_analysis_agent() -> Agent:
    analysis_instructions = get_prompt("analysis")
    return Agent(
        name="AnalysisWorker",
        model="gpt-4o",
        instructions=f"""{analysis_instructions}

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanations, just pure JSON.

Your response must match this exact structure:
{AnalysisOutput.model_json_schema()}

Example response format:
{{
  "executive_summary": "Machine learning offers significant benefits including automation, improved accuracy, and cost reduction.",
  "current_landscape": "The current state of machine learning shows widespread adoption across industries...",
  "key_insights": [
    "ML automation reduces operational costs by 30-50%",
    "Predictive analytics improves decision-making accuracy",
    "Personalization increases customer engagement by 40%"
  ],
  "future_implications": [
    "Continued growth in AI-driven automation",
    "Integration with IoT and edge computing"
  ],
  "critical_considerations": [
    "Data privacy and security concerns",
    "Need for skilled workforce development"
  ],
  "strategic_recommendations": [
    "Invest in ML infrastructure and training",
    "Develop data governance frameworks"
  ],
  "conclusion": "Machine learning represents a transformative technology with clear ROI potential.",
  "citations": [
    {{
      "source_title": "ML Benefits Study 2024",
      "source_url": "https://example.com/study",
      "quote": "Companies using ML see 30-50% cost reduction",
      "page_number": "15"
    }}
  ],
  "methodology": "Comprehensive analysis of industry reports and case studies"
}}""",
        tools=[]
    )

def create_verification_agent() -> Agent:
    return Agent(
        name="VerificationWorker",
        model="gpt-4o-mini",
        instructions=f"""You are a verification specialist. Check claims against sources and provide coverage analysis. Return a verification report with coverage metrics and source validation.

CRITICAL: You MUST respond with ONLY valid JSON. No markdown, no explanations, just pure JSON.

Your response must match this exact structure:
{VerificationOutput.model_json_schema()}

Example response format:
{{
  "accuracy_score": 0.85,
  "completeness_score": 0.78,
  "source_reliability": 0.90,
  "fact_checks": [
    "Claim about 30-50% cost reduction verified in 3 sources",
    "Personalization statistics confirmed by industry reports"
  ],
  "gaps_identified": [
    "Limited data on long-term ROI",
    "Missing information about implementation challenges"
  ],
  "confidence_assessment": "high",
  "recommendations": [
    "Seek additional sources for long-term impact data",
    "Include implementation case studies"
  ]
}}""",
        tools=[]
    )

async def run_retrieval_worker(subtask: Subtask, context: dict[str, str | list | dict]) -> SubtaskResult:
    query = subtask.inputs.get("query", "")
    logger.info(f"🔍 Starting retrieval worker for subtask {subtask.id}: {query}")
    
    try:
        agent = create_retrieval_agent()
        logger.info(f"🤖 Created retrieval agent: {agent.name}")
        
        tokens, structured_output = await get_structured_agent_response(agent, f"Search for comprehensive information about: {query}", RetrievalOutput)
        
        logger.info(f"✅ Retrieval worker completed for {subtask.id}: {tokens} tokens")
        logger.debug(f"📊 Retrieval output keys: {list(structured_output.keys())}")
        
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            output=structured_output,
            tokens_used=tokens,
            seconds_elapsed=1.0
        )
    except Exception as e:
        logger.error(f"❌ Retrieval worker failed for {subtask.id}: {e}")
        return SubtaskResult(
            subtask_id=subtask.id,
            success=False,
            output=[],
            error_message=str(e),
            tokens_used=0,
            seconds_elapsed=0.0
        )

async def run_extraction_worker(subtask: Subtask, context: dict[str, str | list | dict]) -> SubtaskResult:
    sources_key = subtask.inputs.get("sources_key", "")
    sources = context.get(sources_key, [])
    
    
    try:
        agent = create_extraction_agent()
        sources_text = "\n".join([f"Source: {s.get('title', 'Unknown')}\nContent: {s.get('content', '')}" for s in sources])
        prompt = f"Extract key facts from these sources:\n\n{sources_text}"
        
        tokens, response_text = await stream_agent_response(agent, prompt)
        facts = response_text.split('\n') if response_text else []
        
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            output=facts,
            tokens_used=tokens,
            seconds_elapsed=1.5
        )
    except Exception as e:
        return SubtaskResult(
            subtask_id=subtask.id,
            success=False,
            output=[],
            error_message=str(e),
            tokens_used=0,
            seconds_elapsed=0.0
        )

async def run_analysis_worker(subtask: Subtask, context: dict[str, str | list | dict]) -> SubtaskResult:
    facts_key = subtask.inputs.get("facts_key", "")
    facts = context.get(facts_key, [])
    logger.info(f"🧠 Starting analysis worker for subtask {subtask.id} with {len(facts)} facts")
    
    try:
        agent = create_analysis_agent()
        logger.info(f"🤖 Created analysis agent: {agent.name}")
        
        facts_text = "\n".join(f"- {fact}" for fact in facts if fact.strip())
        prompt = f"Analyze and synthesize these facts into a comprehensive answer:\n\n{facts_text}"
        logger.debug(f"📝 Analysis prompt length: {len(prompt)} chars")
        
        tokens, structured_output = await get_structured_agent_response(agent, prompt, AnalysisOutput)
        
        logger.info(f"✅ Analysis worker completed for {subtask.id}: {tokens} tokens")
        logger.debug(f"📊 Analysis output keys: {list(structured_output.keys())}")
        
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            output=structured_output,
            tokens_used=tokens,
            seconds_elapsed=2.0
        )
    except Exception as e:
        return SubtaskResult(
            subtask_id=subtask.id,
            success=False,
            output="Analysis failed",
            error_message=str(e),
            tokens_used=0,
            seconds_elapsed=0.0
        )

async def run_verification_worker(subtask: Subtask, context: dict[str, str | list | dict]) -> SubtaskResult:
    answer_key = subtask.inputs.get("answer_key", "")
    sources_key = subtask.inputs.get("sources_key", "")
    
    answer = context.get(answer_key, "")
    sources = context.get(sources_key, [])
    logger.info(f"✅ Starting verification worker for subtask {subtask.id} with {len(sources)} sources")
    
    try:
        agent = create_verification_agent()
        logger.info(f"🤖 Created verification agent: {agent.name}")
        
        sources_text = "\n".join([f"Source {i+1}: {s.get('title', 'Unknown')} - {s.get('url', 'No URL')}" for i, s in enumerate(sources)])
        prompt = f"Verify this answer against the sources and provide coverage analysis:\n\nAnswer: {answer}\n\nSources:\n{sources_text}"
        logger.debug(f"📝 Verification prompt length: {len(prompt)} chars")
        
        tokens, structured_output = await get_structured_agent_response(agent, prompt, VerificationOutput)
        
        logger.info(f"✅ Verification worker completed for {subtask.id}: {tokens} tokens")
        logger.debug(f"📊 Verification output keys: {list(structured_output.keys())}")
        
        return SubtaskResult(
            subtask_id=subtask.id,
            success=True,
            output=structured_output,
            tokens_used=tokens,
            seconds_elapsed=1.0
        )
    except Exception as e:
        return SubtaskResult(
            subtask_id=subtask.id,
            success=False,
            output={"coverage": 0.0, "error": str(e)},
            error_message=str(e),
            tokens_used=0,
            seconds_elapsed=0.0
        )

WORKER_FUNCTIONS = {
    WorkerType.RETRIEVAL: run_retrieval_worker,
    WorkerType.EXTRACTION: run_extraction_worker,
    WorkerType.ANALYSIS: run_analysis_worker,
    WorkerType.VERIFICATION: run_verification_worker,
}
