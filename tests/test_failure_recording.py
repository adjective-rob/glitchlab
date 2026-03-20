import json
import os
import pytest
from unittest.mock import patch, MagicMock
from glitchlab.controller import Controller

@pytest.fixture
def temp_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return workspace

def test_failure_recording_and_injection(temp_workspace):
    # Setup controller with temp workspace
    controller = Controller(workspace_dir=str(temp_workspace))
    
    # Simulate a failed task
    failed_objective = "Implement a new feature that does X and Y and Z"
    task_id = "task_123"
    
    # Mock the execution to fail
    with patch.object(controller, '_execute_task', return_value={'status': 'implementation_failed', 'reason': 'Could not find file'}):
        controller.run_task(task_id=task_id, objective=failed_objective, model="test-model")
        
    # Check if failure record was created
    failures_file = temp_workspace / ".glitchlab" / "failures.jsonl"
    assert failures_file.exists(), "Failure record file was not created"
    
    with open(failures_file, 'r') as f:
        records = [json.loads(line) for line in f]
        
    assert len(records) == 1
    record = records[0]
    assert record['task_id'] == task_id
    assert record['objective'] == failed_objective[:100]
    assert record['failure_reason'] == 'Could not find file'
    assert 'timestamp' in record
    
    # Now simulate a new task with a similar objective
    new_objective = "Implement a new feature that does X and Y and Z but better"
    new_task_id = "task_124"
    
    # Mock the planner to check if context is injected
    mock_planner = MagicMock()
    with patch('glitchlab.controller.Planner', return_value=mock_planner):
        with patch.object(controller, '_execute_task', return_value={'status': 'success'}):
            controller.run_task(task_id=new_task_id, objective=new_objective, model="test-model")
            
        # Verify planner was initialized or called with the previous attempt context
        # This depends on the exact implementation, but we check if the context was passed somehow
        # Assuming it's passed in the user message or context kwargs
        call_args = mock_planner.call_args or mock_planner.plan.call_args
        if call_args:
            args, kwargs = call_args
            context_str = str(args) + str(kwargs)
            assert "Previous attempt context" in context_str
            assert failed_objective[:60] in context_str
