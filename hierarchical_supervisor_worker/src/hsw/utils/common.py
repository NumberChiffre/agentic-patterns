import time
from hsw.models import WorkerType, SubtaskStatus

def validate_worker_type(worker_type_str: str) -> WorkerType:
    for worker_type in WorkerType:
        if worker_type.value == worker_type_str:
            return worker_type
    raise ValueError(f"Invalid worker type: {worker_type_str}")

def calculate_tokens_from_text(text: str) -> int:
    return len(text.split())

def format_elapsed_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def extract_task_dependencies(task_inputs: dict[str, str | int | list | dict]) -> list[str]:
    dependencies = []
    for key, value in task_inputs.items():
        if key.endswith("_key") and isinstance(value, str):
            dependencies.append(value)
    return dependencies

def calculate_total_budget(subtasks: list) -> tuple[int, int]:
    total_tokens = sum(getattr(task, 'max_tokens', 0) for task in subtasks)
    total_seconds = sum(getattr(task, 'max_seconds', 0) for task in subtasks)
    return total_tokens, total_seconds

def is_task_ready_for_execution(task, completed_tasks: set[str]) -> bool:
    dependencies = extract_task_dependencies(getattr(task, 'inputs', {}))
    return all(dep in completed_tasks for dep in dependencies)

def generate_session_key(prefix: str = "hsw_session") -> str:
    return f"{prefix}_{int(time.time())}"

def safe_get_nested_value(data: dict, keys: list[str], default=None):
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

def merge_context_data(context1: dict, context2: dict) -> dict:
    merged = context1.copy()
    merged.update(context2)
    return merged

def format_error_message(error: Exception, task_id: str = "") -> str:
    error_type = type(error).__name__
    error_msg = str(error)
    prefix = f"Task {task_id}: " if task_id else ""
    return f"{prefix}{error_type}: {error_msg}"

class TaskStatusTracker:
    def __init__(self):
        self.status_counts = {status: 0 for status in SubtaskStatus}
    
    def update_status(self, old_status: SubtaskStatus, new_status: SubtaskStatus):
        self.status_counts[old_status] -= 1
        self.status_counts[new_status] += 1
    
    def get_completion_rate(self) -> float:
        total_tasks = sum(self.status_counts.values())
        if total_tasks == 0:
            return 0.0
        completed_tasks = self.status_counts[SubtaskStatus.COMPLETED]
        return completed_tasks / total_tasks
    
    def has_failures(self) -> bool:
        return self.status_counts[SubtaskStatus.FAILED] > 0

def optimize_execution_order(tasks: list, context: dict[str, str | list | dict]) -> list:
    independent_tasks = []
    dependent_tasks = []
    
    for task in tasks:
        dependencies = extract_task_dependencies(getattr(task, 'inputs', {}))
        if not dependencies or all(dep in context for dep in dependencies):
            independent_tasks.append(task)
        else:
            dependent_tasks.append(task)
    
    dependent_tasks.sort(key=lambda t: len(extract_task_dependencies(getattr(t, 'inputs', {}))))
    
    return independent_tasks + dependent_tasks

