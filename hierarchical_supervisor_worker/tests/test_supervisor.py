import pytest
from hsw.supervisor.supervisor import run_hierarchical_supervisor

@pytest.mark.asyncio
async def test_run_hierarchical_supervisor():
    result = await run_hierarchical_supervisor("Test query about machine learning")
    
    assert result.success is True
    assert result.goal == "Test query about machine learning"
    
    # Meta-agent creates dynamic task IDs, check for the types we expect
    task_ids = list(result.results.keys())
    assert len(task_ids) >= 3  # At least 3 tasks for simple queries
    
    # Check that we have the expected task types (with _001 suffixes)
    retrieve_tasks = [tid for tid in task_ids if tid.startswith("retrieve")]
    analyze_tasks = [tid for tid in task_ids if tid.startswith("analyze")]
    verify_tasks = [tid for tid in task_ids if tid.startswith("verify")]
    
    assert len(retrieve_tasks) >= 1, f"Expected at least 1 retrieve task, got {task_ids}"
    assert len(analyze_tasks) >= 1, f"Expected at least 1 analyze task, got {task_ids}"
    assert len(verify_tasks) >= 1, f"Expected at least 1 verify task, got {task_ids}"
    
    # Check that all subtasks succeeded
    for task_result in result.results.values():
        assert task_result.success is True
    
    assert result.total_tokens >= 0
    assert result.total_seconds >= 0

@pytest.mark.asyncio
async def test_supervisor_output_structure():
    result = await run_hierarchical_supervisor("Simple test")
    
    assert "answer" in result.final_output
    assert "verification" in result.final_output
    assert "sources" in result.final_output
    assert isinstance(result.final_output["sources"], list)
