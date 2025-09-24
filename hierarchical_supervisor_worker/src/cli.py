import argparse
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from hsw.supervisor.supervisor import run_hierarchical_supervisor
import logging
import weave

def format_result_as_markdown(result, query: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = f"""# HSW Report

**Query**: {query}
**Timestamp**: {timestamp}
**Status**: {'✅ SUCCESS' if result.success else '❌ FAILED'}
**Tokens**: {result.total_tokens:,} | **Duration**: {result.total_seconds:.2f}s

## Results

"""
    
    for task_id, task_result in result.results.items():
        status = "✅" if task_result.success else "❌"
        md += f"### {task_id.title()} {status}\n\n"
        
        if task_result.success and isinstance(task_result.output, dict):
            # Handle different output types simply
            if "sources" in task_result.output:
                # Retrieval output
                sources = task_result.output.get("sources", [])
                md += f"**Found {len(sources)} sources**\n\n"
                for i, source in enumerate(sources):  # Show ALL sources
                    md += f"{i+1}. **{source.get('title', 'Unknown')}**\n"
                    if source.get('url'):
                        md += f"   - URL: {source.get('url')}\n"
                    if source.get('source_type'):
                        md += f"   - Type: {source.get('source_type')}\n"
                    if source.get('credibility_score'):
                        md += f"   - Credibility: {source.get('credibility_score')}\n"
                    if source.get('content'):
                        md += f"   - Content: {source.get('content')}\n"  # FULL content
                    md += "\n"
                        
            elif "executive_summary" in task_result.output:
                # Analysis output - FULL structured display
                md += f"**Executive Summary**: {task_result.output.get('executive_summary', 'N/A')}\n\n"
                
                if task_result.output.get('current_landscape'):
                    md += f"**Current Landscape**: {task_result.output.get('current_landscape')}\n\n"
                
                insights = task_result.output.get('key_insights', [])
                if insights:
                    md += "**Key Insights**:\n"
                    for insight in insights:  # Show ALL insights
                        md += f"- {insight}\n"
                    md += "\n"
                
                implications = task_result.output.get('future_implications', [])
                if implications:
                    md += "**Future Implications**:\n"
                    for implication in implications:  # Show ALL implications
                        md += f"- {implication}\n"
                    md += "\n"
                
                considerations = task_result.output.get('critical_considerations', [])
                if considerations:
                    md += "**Critical Considerations**:\n"
                    for consideration in considerations:  # Show ALL considerations
                        md += f"- {consideration}\n"
                    md += "\n"
                
                recommendations = task_result.output.get('strategic_recommendations', [])
                if recommendations:
                    md += "**Strategic Recommendations**:\n"
                    for recommendation in recommendations:  # Show ALL recommendations
                        md += f"- {recommendation}\n"
                    md += "\n"
                
                if task_result.output.get('conclusion'):
                    md += f"**Conclusion**: {task_result.output.get('conclusion')}\n\n"
                
                if task_result.output.get('methodology'):
                    md += f"**Methodology**: {task_result.output.get('methodology')}\n\n"
                
                citations = task_result.output.get('citations', [])
                if citations:
                    md += "**Citations**:\n"
                    for i, citation in enumerate(citations):  # Show ALL citations
                        md += f"{i+1}. **{citation.get('source_title', 'Unknown')}**\n"
                        if citation.get('source_url'):
                            md += f"   - URL: {citation.get('source_url')}\n"
                        if citation.get('quote'):
                            md += f"   - Quote: \"{citation.get('quote')}\"\n"
                        if citation.get('page_number'):
                            md += f"   - Page: {citation.get('page_number')}\n"
                        md += "\n"
                        
            elif "accuracy_score" in task_result.output:
                # Verification output - FULL structured display
                md += f"**Accuracy Score**: {task_result.output.get('accuracy_score', 0):.2f}\n"
                md += f"**Completeness Score**: {task_result.output.get('completeness_score', 0):.2f}\n"
                md += f"**Source Reliability**: {task_result.output.get('source_reliability', 0):.2f}\n"
                md += f"**Confidence Assessment**: {task_result.output.get('confidence_assessment', 'unknown')}\n\n"
                
                fact_checks = task_result.output.get('fact_checks', [])
                if fact_checks:
                    md += "**Fact Checks**:\n"
                    for check in fact_checks:  # Show ALL fact checks
                        md += f"- {check}\n"
                    md += "\n"
                
                gaps = task_result.output.get('gaps_identified', [])
                if gaps:
                    md += "**Gaps Identified**:\n"
                    for gap in gaps:  # Show ALL gaps
                        md += f"- {gap}\n"
                    md += "\n"
                
                recommendations = task_result.output.get('recommendations', [])
                if recommendations:
                    md += "**Recommendations**:\n"
                    for rec in recommendations:  # Show ALL recommendations
                        md += f"- {rec}\n"
                    md += "\n"
            
            else:
                # Generic dict output - FULL display
                for key, value in task_result.output.items():  # Show ALL items
                    if isinstance(value, list):
                        md += f"**{key.title()}**:\n"
                        for item in value:  # Show ALL list items
                            md += f"- {item}\n"
                        md += "\n"
                    else:
                        md += f"**{key.title()}**: {str(value)}\n"  # FULL value
                md += "\n"
        
        elif task_result.success:
            # String or other output - FULL display
            output_str = str(task_result.output)  # FULL output, no truncation
            md += f"{output_str}\n\n"
        
        else:
            md += f"**Error**: {task_result.error_message}\n\n"
    
    md += f"""## Final Output

{json.dumps(result.final_output, indent=2)}

---
*Generated by HSW System - Session: {result.session_id}*
"""
    
    return md

async def async_main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="HSW Demo")
    parser.add_argument("command", choices=["demo"], help="Command to run")
    parser.add_argument("--query", required=True, help="Query to process")
    parser.add_argument("--save-md", action="store_true", help="Save as markdown")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal logging output")
    
    args = parser.parse_args()
    
    # Set up clean INFO logging by default for execution flow visibility
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format='%(message)s'  # Clean format without level/timestamp noise
    )
    
    # Always silence noisy external libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('weave').setLevel(logging.WARNING)
    logging.getLogger('wandb').setLevel(logging.WARNING)
        
    # Initialize weave tracing (with fallback if API is down)
    project = os.getenv("WEAVE_PROJECT", "hierarchical-supervisor-worker")
    try:
        weave.init(project)
        if not args.quiet:
            print("✅ Weave tracing initialized")
    except Exception as e:
        if not args.quiet:
            print(f"⚠️  Weave initialization failed ({e}), continuing without tracing")
    
    try:
        result = await run_hierarchical_supervisor(args.query)
        
        if args.save_md:
            md_content = format_result_as_markdown(result, args.query)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            md_path = Path("logs") / f"hsw_report_{timestamp}.md"
            md_path.parent.mkdir(exist_ok=True)
            md_path.write_text(md_content)
            print(f"Report saved: {md_path}")
        else:
            print(format_result_as_markdown(result, args.query))
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

def main():
    return asyncio.run(async_main())

if __name__ == "__main__":
    import sys
    sys.exit(main())