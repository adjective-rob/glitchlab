# Project Roadmap

## Now
- Improve test coverage on core pipeline files (agent_runners, step_handlers, lifecycle)
- Fix the pre-existing failing test (test_task_writer_create_task_omits_mode)
- Add error handling for edge cases in the implementer agentic loop (malformed tool responses, empty content)
- Ensure every public function in glitchlab/ has a docstring

## Next
- Add integration tests that run a full pipeline end-to-end with mocked LLM responses
- Improve event bus coverage (verify all pipeline events emit correctly)
- Add CLI smoke tests for all commands (run, batch, audit, history, init)

## Deferred
- Test file docstrings — low value, do not generate tasks for missing_doc in tests/
- Prelude v2 integration
