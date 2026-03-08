# ADR-015: Zephyr SBOF Integration and EventBus Architecture Upgrade (v4.2.0)

**Status:** accepted
**Date:** 2026-03-07
**Version:** 4.2.0

## Context

GLITCHLAB's agent-driven development engine required two critical enhancements to meet modern supply-chain security and observability requirements:

1. **Cryptographic Attestation Gap**: Agent actions (tool calls, plan steps, code mutations) lacked cryptographic signing and tamper-evident signatures. This made it difficult to provide audit-ready provenance for supply-chain security compliance and to detect unauthorized modifications to the event log.

2. **Deterministic Traceability Gap**: The EventBus lacked first-class fields to uniquely identify and correlate events across agent execution loops. Without `run_id`, `action_id`, and metadata fields, it was challenging to reconstruct the exact sequence of agent decisions and their outcomes for debugging and compliance purposes.

## Decision

### 1. Zephyr SBOF Integration

Implemented cryptographic signing and attestation for all agent actions through the **Zephyr SBOF (Secure Build Orchestration Framework)** integration:

- **Every agent action** (tool call, plan step, code mutation) is now signed with a tamper-evident signature before being committed to the event log.
- **Signature format**: HMAC-SHA256 with a project-specific signing key, ensuring authenticity and integrity.
- **Attestation metadata**: Each signed action includes:
  - Timestamp of action execution
  - Agent identifier and version
  - Action type and parameters
  - Cryptographic hash of the action payload
- **Audit trail**: All signatures are stored alongside events, enabling post-hoc verification and compliance audits.
- **Tamper detection**: Any modification to a signed action invalidates its signature, immediately alerting operators to unauthorized changes.

**Benefits:**
- Supply-chain security compliance (SLSA, SBOM, attestation requirements)
- Cryptographic proof of action authenticity
- Audit-ready provenance for regulatory compliance
- Tamper detection and forensic analysis capabilities

### 2. EventBus Architecture Upgrade

Enhanced the internal EventBus with three new first-class fields on every event:

- **`run_id` (UUID)**: Uniquely identifies a complete agent execution run. All events within a single task execution share the same `run_id`, enabling perfect correlation across the entire agent loop.
- **`action_id` (UUID)**: Uniquely identifies a specific agent action within a run. Enables precise tracing of individual decisions and their outcomes.
- **`metadata` (dict)**: Carries contextual information about the action:
  - Agent name and version
  - Task context (task ID, priority, mode)
  - Execution timing (start, end, duration)
  - Resource consumption (tokens, API calls)
  - Outcome indicators (success, failure, retry count)

**Benefits:**
- **Deterministic traceability**: Reconstruct the exact sequence of agent decisions and outcomes.
- **Perfect correlation**: Link all side effects (logs, metrics, errors) to specific actions.
- **Debugging**: Quickly identify which agent action caused a failure or unexpected behavior.
- **Performance analysis**: Track resource consumption per action and per run.
- **Compliance**: Demonstrate complete audit trail of all agent decisions.

## Rationale

### Why Zephyr SBOF?

Modern software supply chains require cryptographic attestation of build and deployment artifacts. By integrating Zephyr SBOF, GLITCHLAB can:
- Provide SLSA (Supply-chain Levels for Software Artifacts) compliance
- Generate Software Bill of Materials (SBOM) with signed provenance
- Enable downstream systems to verify the authenticity of agent-generated code
- Support regulatory requirements (SOC 2, ISO 27001, etc.)

### Why EventBus Upgrade?

Agent-driven development is inherently non-deterministic due to LLM variability and external tool responses. By adding `run_id`, `action_id`, and metadata:
- We create a deterministic audit trail that can be replayed and analyzed
- We enable perfect correlation of events across distributed systems
- We support compliance requirements for action traceability
- We improve debugging and observability for complex multi-agent scenarios

## Consequences

### Positive
- ✅ Supply-chain security compliance (SLSA, SBOM, attestation)
- ✅ Cryptographic proof of action authenticity
- ✅ Perfect deterministic traceability of agent loops
- ✅ Enhanced debugging and observability
- ✅ Audit-ready provenance for regulatory compliance
- ✅ Tamper detection and forensic analysis

### Negative
- ⚠️ Increased event payload size (signatures, UUIDs, metadata)
- ⚠️ Slight performance overhead for cryptographic signing
- ⚠️ Additional storage requirements for event logs
- ⚠️ Complexity in event schema (new required fields)

### Mitigation
- Event batching and compression to reduce storage overhead
- Async signing to minimize performance impact
- Clear documentation and examples for event consumers
- Backward compatibility layer for legacy event consumers

## Implementation Details

### Zephyr SBOF Integration
- Signing occurs in the EventBus publisher before event dispatch
- Signatures are stored as part of event metadata
- Verification can be performed by event consumers or external audit tools
- Signing key is managed via environment variables or secure key store

### EventBus Architecture
- `run_id` is generated at task start and propagated to all events
- `action_id` is generated for each discrete agent action
- `metadata` is populated by the agent performing the action
- All three fields are required (non-nullable) on every event

## Related ADRs
- ADR-012: Agnostic Event Bus for System Decoupling
- ADR-0003: Extended GlitchEvent Schema with Traceability Fields

## References
- SLSA Framework: https://slsa.dev/
- SBOM Specification: https://cyclonedx.org/
- HMAC-SHA256: https://tools.ietf.org/html/rfc4868
