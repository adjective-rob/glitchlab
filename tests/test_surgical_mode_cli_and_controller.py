from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import glitchlab.cli as cli
from glitchlab.config_loader import PipelineStep


def test_run_cli_passes_surgical_and_controller_uses_surgical_profile(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".glitchlab" / "tasks" / "queue").mkdir(parents=True)
    task_file = repo / ".glitchlab" / "tasks" / "queue" / "next.yaml"
    task_file.write_text("id: t1\nobjective: test surgical mode\nrisk: low\n")

    base_pipeline = [PipelineStep(name="planner", agent_role="planner")]
    surgical_pipeline = [PipelineStep(name="implementer", agent_role="implementer")]

    loaded_profiles = []
    controller_inits = []
    observed = {}

    class DummyController:
        def __init__(self, repo_path, config, allow_core=False, auto_approve=False, surgical=False, test_command=None):
            controller_inits.append(
                {
                    "repo_path": repo_path,
                    "config": config,
                    "allow_core": allow_core,
                    "auto_approve": auto_approve,
                    "surgical": surgical,
                    "test_command": test_command,
                }
            )

        def run(self, task):
            return {"status": "committed"}

    def fake_load_config(repo_path=None, profile=None):
        loaded_profiles.append(profile)
        pipeline = surgical_pipeline if profile == "surgical" else base_pipeline
        return SimpleNamespace(
            pipeline=list(pipeline),
            limits=SimpleNamespace(max_fix_attempts=4),
            boundaries=SimpleNamespace(protected_paths=[]),
            automation=SimpleNamespace(auto_merge_pr=False),
        )

    monkeypatch.setattr(cli, "load_config", fake_load_config)
    monkeypatch.setattr(cli, "Controller", DummyController)
    monkeypatch.setattr(cli, "_print_banner", lambda: None)
    monkeypatch.setattr(cli, "_configure_logging", lambda verbose: None)
    monkeypatch.setattr(cli, "_detect_test_command", lambda repo: "pytest -q")
    monkeypatch.setattr(cli, "AuditLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli.Task, "from_yaml", staticmethod(lambda path: SimpleNamespace(task_id="t1", objective="test surgical mode", mode=None, risk_level="low")))

    runner = CliRunner()
    result = runner.invoke(cli.app, ["run", "--repo", str(repo), "--local-task", "--surgical"])
    assert result.exit_code == 0, result.output
    assert controller_inits and controller_inits[0]["surgical"] is True

    import glitchlab.controller as controller_mod

    monkeypatch.setattr(controller_mod, "pre_task_git_fetch", lambda repo_path: None)
    monkeypatch.setattr(controller_mod, "check_repo_clean", lambda repo_path: None)
    monkeypatch.setattr(controller_mod, "print_banner", lambda task: None)
    monkeypatch.setattr(controller_mod, "startup", lambda ctx, task: "failure-context")
    monkeypatch.setattr(controller_mod, "finalize", lambda ctx, task, *args: {"status": "ok"})
    monkeypatch.setattr(controller_mod, "post_run", lambda ctx, task, result: result)
    monkeypatch.setattr(controller_mod, "write_session_entry", lambda ctx, task, result: None)
    monkeypatch.setattr(controller_mod.bus, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller_mod, "Router", lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(controller_mod, "BoundaryEnforcer", lambda protected_paths: SimpleNamespace(protected_paths=protected_paths))
    monkeypatch.setattr(controller_mod, "PreludeContext", lambda repo_path: SimpleNamespace())
    monkeypatch.setattr(controller_mod, "TaskHistory", lambda repo_path: SimpleNamespace(record=lambda result: None))
    monkeypatch.setattr(controller_mod, "AGENT_REGISTRY", {})
    monkeypatch.setattr(controller_mod, "get_agent", lambda role, router: None)

    def fake_controller_load_config(repo_path=None, profile=None):
        loaded_profiles.append(profile)
        pipeline = surgical_pipeline if profile == "surgical" else base_pipeline
        return SimpleNamespace(
            pipeline=list(pipeline),
            limits=SimpleNamespace(max_fix_attempts=4),
            boundaries=SimpleNamespace(protected_paths=[]),
            automation=SimpleNamespace(auto_merge_pr=False),
        )

    monkeypatch.setattr(controller_mod, "load_config", fake_controller_load_config)

    def fake_execute_pipeline(self, ctx, task, failure_context, result_dict):
        observed["ctx_surgical"] = ctx.surgical
        observed["pipeline_roles"] = [step.agent_role for step in ctx.config.pipeline]
        observed["max_fix_attempts"] = ctx.config.limits.max_fix_attempts
        return controller_mod.PipelineState(result=result_dict)

    monkeypatch.setattr(controller_mod.Controller, "_execute_pipeline", fake_execute_pipeline)

    controller = controller_mod.Controller(repo_path=repo, config=fake_controller_load_config(repo), surgical=True)
    task = SimpleNamespace(task_id="t1", objective="test surgical mode", mode=None, risk_level="low")
    run_result = controller.run(task)

    assert run_result["status"] == "ok"
    assert "surgical" in loaded_profiles
    assert observed["ctx_surgical"] is True
    assert observed["pipeline_roles"] == ["implementer"]
    assert observed["max_fix_attempts"] == 1
