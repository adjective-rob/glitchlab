# ADR-019: YAML-Based Inter-Agent Messaging (v4.3.0)

**Status:** Accepted  
**Date:** 2026-03-08  
**Version:** 4.3.0  
**Deciders:** GLITCHLAB Core Team  

---

## **Context**

As GLITCHLAB's agent roster has grown and inter-agent communication has become more complex, the previous ad-hoc serialization approach has become difficult to maintain, extend, and audit. Agents need a standardized, human-readable, and easily inspectable format for passing structured data to one another.

The key requirements are:
- **Human-readable:** Developers and operators should be able to inspect messages without special tools.
- **Structured:** Messages must carry typed, nested data with clear schemas.
- **Extensible:** New message types and fields should be added without breaking existing agents.
- **Auditable:** The full message history should be easy to log, replay, and verify.
- **Language-agnostic:** The format should be independent of Python implementation details.

---

## **Decision**

We adopt **YAML** as the standard format for all inter-agent message communication in GLITCHLAB v4.3.0 and beyond.

### **Rationale**

1. **Human-Readable:** YAML is designed for human readability. Operators can inspect agent messages in logs without parsing binary or JSON.

2. **Structured & Typed:** YAML supports nested objects, lists, and scalar types. Combined with Pydantic models, we can enforce strict schemas on the Python side while keeping the wire format clean.

3. **Extensible:** YAML's flexibility allows agents to add optional fields without breaking consumers. Pydantic's `extra="ignore"` pattern handles forward compatibility gracefully.

4. **Audit Trail:** YAML messages are easy to log, diff, and replay. The format is stable and deterministic (with proper serialization rules).

5. **Ecosystem:** YAML is widely supported across languages and tools. If GLITCHLAB ever needs to integrate with external systems or multi-language agent implementations, YAML is a safe choice.

6. **Debugging:** When an agent fails or produces unexpected output, YAML logs are immediately inspectable without custom deserialization logic.

---

## **Implementation**

### **Message Structure**

All inter-agent messages follow this top-level structure:

```yaml
message_type: <string>  # e.g., "plan_request", "implementation_result", "test_report"
agent_id: <string>      # e.g., "professor_zap", "patch", "reroute"
timestamp: <ISO8601>    # e.g., "2026-03-08T12:34:56Z"
payload: <object>       # Message-specific data
metadata: <object>      # Optional: run_id, action_id, etc.
```

### **Serialization Rules**

- All messages are serialized to YAML using `pyyaml>=6.0.3` (already a dependency).
- Pydantic models are converted to dicts via `.model_dump()` before YAML serialization.
- Deserialization uses Pydantic's `model_validate()` to enforce schemas.
- Timestamps are ISO8601 strings for consistency and readability.

### **Backward Compatibility**

- Agents receiving messages with unknown fields ignore them (Pydantic `extra="ignore"`).
- New message types are added without affecting existing agents.
- The EventBus continues to emit `run_id`, `action_id`, and `metadata` fields automatically.

---

## **Consequences**

### **Positive**

- ✅ **Debuggability:** Operators can inspect agent communication in real time.
- ✅ **Auditability:** Full message history is human-readable and easy to log.
- ✅ **Extensibility:** New agents and message types integrate cleanly.
- ✅ **Compliance:** YAML logs satisfy supply-chain and audit requirements.
- ✅ **Maintainability:** No custom serialization logic; Pydantic + YAML is standard.

### **Negative**

- ⚠️ **Performance:** YAML parsing is slower than JSON or binary formats. For high-throughput scenarios, this may be a bottleneck (mitigated by agent-level caching and batching).
- ⚠️ **Schema Drift:** Without strict versioning, message schemas can drift over time. Mitigation: use Pydantic's `model_version` and explicit schema versioning in message types.

### **Neutral**

- ℹ️ **Dependency:** Adds `pyyaml` as a required dependency (already present in v4.3.0).

---

## **Alternatives Considered**

1. **JSON:** Simpler, faster, but less human-readable for complex nested structures. Rejected because operator experience is a priority.

2. **Protocol Buffers:** Highly efficient and strongly typed, but requires code generation and is overkill for GLITCHLAB's scale. Rejected for complexity.

3. **MessagePack:** Binary format, fast, but not human-readable. Rejected because auditability is critical.

4. **Ad-hoc Serialization:** Current approach. Rejected because it's unmaintainable and error-prone as the agent roster grows.

---

## **Related ADRs**

- **ADR-015:** Zephyr SBOF Integration and EventBus Architecture Upgrade (v4.2.0) — defines the EventBus fields that YAML messages carry.

---

## **References**

- YAML 1.2 Spec: https://yaml.org/spec/1.2/spec.html
- Pydantic Serialization: https://docs.pydantic.dev/latest/concepts/serialization/
- GLITCHLAB EventBus: `glitchlab/eventbus.py`

---

## **Sign-Off**

- **Accepted by:** GLITCHLAB Core Team
- **Date:** 2026-03-08
- **Version:** 4.3.0
