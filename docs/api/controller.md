

### Error Handling & Retries
- `execute()`: Now handles `TestFailureException` by incrementing a retry counter and re-invoking the agent logic.
- Default Retry Limit: 3