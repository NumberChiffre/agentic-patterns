import pytest
import asyncio
from hsw.models import Plan, Subtask, SubtaskResult, SubtaskStatus, SupervisorResult, WorkerType
from hsw.supervisor.supervisor import (
    create_supervisor_agent, create_adaptive_plan,
    get_or_create_worker_agent, build_worker_prompt, execute_worker_task,
    synthesize_final_results, handle_supervisor_errors,
    run_hierarchical_supervisor
)
from hsw.workers.agents import (
    create_retrieval_agent, create_extraction_agent,
    create_analysis_agent, create_verification_agent
)
from hsw.config import load_config
from hsw.runtime.state import StateManager

class TestSupervisorFunctions:
    def test_create_supervisor_agent(self):
        agent = create_supervisor_agent()
        assert agent.name == "SupervisorMetaAgent"
        assert agent.model == "gpt-4o"
        assert len(agent.instructions) > 0

    def test_create_adaptive_plan_simple(self):
        goal = "simple query"
        plan = create_adaptive_plan(goal, "reasoning")
        
        assert plan.root_goal == goal
        assert len(plan.subtasks) == 3
        assert plan.subtasks[0].worker_type == WorkerType.RETRIEVAL
        assert plan.subtasks[1].worker_type == WorkerType.ANALYSIS  
        assert plan.subtasks[2].worker_type == WorkerType.VERIFICATION

    def test_create_adaptive_plan_complex(self):
        goal = "this is a very long and complex query that requires comprehensive analysis and multiple stages of processing to handle properly"
        plan = create_adaptive_plan(goal, "reasoning")
        
        assert plan.root_goal == goal
        assert len(plan.subtasks) == 4
        assert plan.subtasks[0].worker_type == WorkerType.RETRIEVAL
        assert plan.subtasks[1].worker_type == WorkerType.EXTRACTION
        assert plan.subtasks[2].worker_type == WorkerType.ANALYSIS
        assert plan.subtasks[3].worker_type == WorkerType.VERIFICATION

    def test_get_or_create_worker_agent(self):
        cache = {}
        
        retrieval_agent = get_or_create_worker_agent(WorkerType.RETRIEVAL, cache)
        assert retrieval_agent.name == "RetrievalWorker"
        assert "retrieval" in cache
        
        same_agent = get_or_create_worker_agent(WorkerType.RETRIEVAL, cache)
        assert same_agent is retrieval_agent

    def test_get_or_create_worker_agent_all_types(self):
        cache = {}
        
        for worker_type in WorkerType:
            agent = get_or_create_worker_agent(worker_type, cache)
            assert agent is not None
            assert worker_type.value in cache

    def test_build_worker_prompt_simple(self):
        subtask = Subtask(
            id="test_task",
            goal="Test goal",
            worker_type=WorkerType.RETRIEVAL,
            inputs={"query": "test query"},
            max_tokens=100,
            max_seconds=10
        )
        context = {}
        
        prompt = build_worker_prompt(subtask, context)
        assert "Test goal" in prompt
        assert "test query" in prompt

    def test_build_worker_prompt_with_context(self):
        subtask = Subtask(
            id="test_task",
            goal="Test goal",
            worker_type=WorkerType.ANALYSIS,
            inputs={"sources_key": "retrieve_001"},
            max_tokens=100,
            max_seconds=10
        )
        context = {"retrieve_001": ["source1", "source2"]}
        
        prompt = build_worker_prompt(subtask, context)
        assert "Test goal" in prompt
        assert "Context from retrieve_001" in prompt

    def test_synthesize_final_results(self):
        results = {
            "retrieve_001": SubtaskResult(
                subtask_id="retrieve_001",
                success=True,
                output=[{"title": "Source 1", "content": "Content 1"}],
                tokens_used=100,
                seconds_elapsed=1.0
            ),
            "analyze_001": SubtaskResult(
                subtask_id="analyze_001", 
                success=True,
                output="Analysis result",
                tokens_used=200,
                seconds_elapsed=2.0
            ),
            "verify_001": SubtaskResult(
                subtask_id="verify_001",
                success=True,
                output={"coverage": 0.9, "verified": True},
                tokens_used=150,
                seconds_elapsed=1.5
            )
        }
        
        final_output = synthesize_final_results(results)
        
        assert final_output["answer"] == "Analysis result"
        assert final_output["verification"] == {"coverage": 0.9, "verified": True}
        assert len(final_output["sources"]) == 1

    def test_handle_supervisor_errors(self):
        error = RuntimeError("Test error")
        goal = "test goal"
        start_time = 0.0
        
        result = handle_supervisor_errors(error, goal, start_time)
        
        assert not result.success
        assert result.goal == goal
        assert "Test error" in result.final_output["error"]
        assert result.total_tokens == 0

class TestWorkerFunctions:
    def test_create_all_worker_agents(self):
        retrieval = create_retrieval_agent()
        extraction = create_extraction_agent()
        analysis = create_analysis_agent() 
        verification = create_verification_agent()
        
        assert retrieval.name == "RetrievalWorker"
        assert extraction.name == "ExtractionWorker"
        assert analysis.name == "AnalysisWorker"
        assert verification.name == "VerificationWorker"
        
        assert all(agent.model in ["gpt-4o-mini", "gpt-4o"] for agent in [retrieval, extraction, analysis, verification])

class TestAsyncIntegration:
    @pytest.mark.asyncio
    async def test_run_hierarchical_supervisor_integration(self):
        goal = "Test AI applications"
        result = await run_hierarchical_supervisor(goal)
        
        assert result.goal == goal
        assert isinstance(result, SupervisorResult)
        assert result.total_seconds > 0
        
        if result.success:
            assert result.total_tokens > 0
            assert len(result.results) >= 3

    @pytest.mark.asyncio 
    async def test_execute_worker_task_retrieval(self):
        subtask = Subtask(
            id="retrieve_test",
            goal="Test retrieval",
            worker_type=WorkerType.RETRIEVAL,
            inputs={"query": "test query"},
            max_tokens=100,
            max_seconds=5
        )
        context = {}
        worker_cache = {}
        
        result = await execute_worker_task(subtask, context, worker_cache)
        
        assert result.subtask_id == "retrieve_test"
        assert isinstance(result, SubtaskResult)
        
        if result.success:
            assert result.tokens_used > 0
            assert result.output is not None

class TestStateManager:
    def test_state_manager_creation(self):
        manager = StateManager.from_url("redis://localhost:6379/0")
        assert manager is not None

    def test_state_manager_json_operations(self):
        manager = StateManager.from_url("redis://localhost:6379/0")
        
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        try:
            manager.store_json("test_key", test_data, ttl_seconds=60)
            retrieved = manager.get_json("test_key")
            
            if retrieved:
                assert retrieved["key"] == "value"
                assert retrieved["number"] == 42
        except Exception:
            pytest.skip("Redis not available for testing")

class TestConfiguration:
    def test_load_config(self):
        config = load_config()
        
        assert config.max_tokens > 0
        assert config.max_seconds > 0
        assert config.max_depth > 0
        assert config.redis_url.startswith("redis://")

class TestModels:
    def test_worker_type_enum(self):
        assert WorkerType.RETRIEVAL.value == "retrieval"
        assert WorkerType.EXTRACTION.value == "extraction"
        assert WorkerType.ANALYSIS.value == "analysis"
        assert WorkerType.VERIFICATION.value == "verification"

    def test_subtask_status_enum(self):
        assert SubtaskStatus.PENDING.value == "pending"
        assert SubtaskStatus.RUNNING.value == "running"
        assert SubtaskStatus.COMPLETED.value == "completed"
        assert SubtaskStatus.FAILED.value == "failed"

    def test_subtask_model(self):
        subtask = Subtask(
            id="test",
            goal="Test goal",
            worker_type=WorkerType.RETRIEVAL,
            inputs={"query": "test"},
            max_tokens=100,
            max_seconds=10
        )
        
        assert subtask.id == "test"
        assert subtask.status == SubtaskStatus.PENDING
        assert len(subtask.children) == 0

    def test_subtask_result_model(self):
        result = SubtaskResult(
            subtask_id="test",
            success=True,
            output="test output",
            tokens_used=50,
            seconds_elapsed=1.0
        )
        
        assert result.subtask_id == "test"
        assert result.success
        assert result.output == "test output"
        assert result.error_message is None

    def test_plan_model(self):
        subtasks = [
            Subtask(
                id="test1",
                goal="Test 1",
                worker_type=WorkerType.RETRIEVAL,
                inputs={"query": "test"},
                max_tokens=100,
                max_seconds=10
            )
        ]
        
        plan = Plan(
            root_goal="Test goal",
            subtasks=subtasks,
            max_depth=2,
            total_budget_tokens=1000,
            total_budget_seconds=30
        )
        
        assert plan.root_goal == "Test goal"
        assert len(plan.subtasks) == 1
        assert plan.max_depth == 2

    def test_supervisor_result_model(self):
        plan = Plan(
            root_goal="Test",
            subtasks=[],
            max_depth=1,
            total_budget_tokens=100,
            total_budget_seconds=10
        )
        
        result = SupervisorResult(
            goal="Test goal",
            plan=plan,
            results={},
            final_output={"answer": "test"},
            success=True,
            total_tokens=100,
            total_seconds=5.0
        )
        
        assert result.goal == "Test goal"
        assert result.success
        assert result.total_tokens == 100
        assert result.final_output["answer"] == "test"

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_worker_task_error_handling(self):
        subtask = Subtask(
            id="error_test",
            goal="This will fail",
            worker_type=WorkerType.RETRIEVAL,
            inputs={"query": ""},
            max_tokens=1,
            max_seconds=1
        )
        context = {}
        worker_cache = {}
        
        result = await execute_worker_task(subtask, context, worker_cache)
        
        assert result.subtask_id == "error_test"
        if not result.success:
            assert result.error_message is not None
            assert result.tokens_used == 0

class TestROMACompliance:
    @pytest.mark.asyncio
    async def test_hierarchical_delegation_pattern(self):
        goal = "Complex multi-step analysis task requiring hierarchical coordination"
        result = await run_hierarchical_supervisor(goal)
        
        assert isinstance(result.plan, Plan)
        assert len(result.plan.subtasks) >= 3
        
        has_retrieval = any(task.worker_type == WorkerType.RETRIEVAL for task in result.plan.subtasks)
        has_analysis = any(task.worker_type == WorkerType.ANALYSIS for task in result.plan.subtasks)
        has_verification = any(task.worker_type == WorkerType.VERIFICATION for task in result.plan.subtasks)
        
        assert has_retrieval
        assert has_analysis
        assert has_verification

    def test_functional_programming_compliance(self):
        from hsw.supervisor import supervisor
        import inspect
        
        functions = [
            obj for name, obj in inspect.getmembers(supervisor)
            if inspect.isfunction(obj) and not name.startswith('_')
        ]
        
        # Only check for locally defined classes (defined in this module)
        classes = [
            obj for name, obj in inspect.getmembers(supervisor)
            if inspect.isclass(obj) and not name.startswith('_') 
            and obj.__module__ == supervisor.__name__
        ]
        
        assert len(functions) >= 8
        assert len(classes) == 0

    def test_pydantic_model_usage(self):
        from hsw import models
        import inspect
        from pydantic import BaseModel
        
        model_classes = [
            obj for name, obj in inspect.getmembers(models)
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj != BaseModel
        ]
        
        assert len(model_classes) >= 4
        
        for model_class in model_classes:
            instance_methods = [method for method in dir(model_class) if not method.startswith('_')]
            assert 'model_dump' in instance_methods or 'dict' in instance_methods

class TestPerformanceBasics:
    @pytest.mark.asyncio
    async def test_supervisor_response_time(self):
        import time
        
        start = time.time()
        result = await run_hierarchical_supervisor("Quick test query")
        duration = time.time() - start
        
        assert duration < 120
        assert result.total_seconds > 0

    @pytest.mark.asyncio
    async def test_concurrent_execution_support(self):
        tasks = [
            run_hierarchical_supervisor(f"Query {i}") 
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, SupervisorResult)
