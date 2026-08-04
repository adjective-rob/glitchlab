"""
HONEYMOON Report Writer — Signed investigation output.

For investigation and monitor missions, the output is a signed
markdown report instead of a git PR. Each report is:
  1. Written to .honeymoon/reports/{run_id}.md
  2. Signed with the Ed25519 keypair
  3. Signature appended to the report footer

Reports are self-contained — anyone with the public key can verify
that the investigation was produced by this Honeymoon instance
and hasn't been tampered with.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from honeymoon.signing import HiveSigner

console = Console()


def _esc(text: Any) -> str:
    """Escape text for safe interpolation into HTML.

    LLM- and user-derived fields (finding titles, analysis, summaries,
    recommendations) routinely contain angle brackets — code snippets,
    generics like ``List<String>``, URL templates like ``/api/report/<id>``.
    Interpolated raw, a stray ``<`` opens a tag the browser never closes and
    swallows the rest of the document, rendering the report blank.
    """
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _load_provenance(repo_path: Path, run_id: str) -> list[dict[str, Any]]:
    """Load signed audit events for a specific run from audit.jsonl."""
    audit_file = repo_path / ".honeymoon" / "logs" / "audit.jsonl"
    if not audit_file.exists():
        return []

    events = []
    for line in audit_file.read_text().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
            # Zephyr-signed entries have payload as a JSON string
            payload = entry.get("payload", "")
            if isinstance(payload, str):
                payload_data = json.loads(payload)
            else:
                payload_data = payload
            if payload_data.get("run_id") == run_id:
                events.append({
                    "event_type": payload_data.get("event_type", "unknown"),
                    "agent_id": payload_data.get("agent_id", "system"),
                    "timestamp": payload_data.get("timestamp", entry.get("timestamp", "")),
                    "signature": entry.get("signature", ""),
                    "signer": entry.get("signer", ""),
                })
        except (json.JSONDecodeError, KeyError):
            continue
    return events


def write_report(
    repo_path: Path,
    run_id: str,
    mission_name: str,
    objective: str,
    findings: dict[str, Any],
    verification: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
) -> Path:
    """Write a signed investigation report.

    Args:
        repo_path: Root of the target repository
        run_id: Unique run identifier
        mission_name: Mission that produced this report
        objective: The investigation objective
        findings: The analyst's structured findings
        verification: The verifier's verdict (optional)
        budget: Token/cost summary (optional)

    Returns:
        Path to the written report file
    """
    reports_dir = repo_path / ".honeymoon" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    short_id = run_id[:12]

    # --- Build the report ---
    sections = []
    finding_list = findings.get("findings", [])

    # Header
    sections.append(f"# 🍯 Investigation Report — {short_id}")
    sections.append("")
    sections.append("| Field | Value |")
    sections.append("|-------|-------|")
    sections.append(f"| **Mission** | {mission_name} |")
    sections.append(f"| **Objective** | {objective} |")
    sections.append(f"| **Timestamp** | {timestamp} |")
    sections.append(f"| **Run ID** | `{run_id}` |")
    sections.append(f"| **Findings** | {len(finding_list)} |")

    # Risk matrix
    severity_counts: dict[str, int] = {}
    for f in finding_list:
        sev = f.get("severity", "info").upper()
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    if severity_counts:
        risk_parts = []
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            if sev in severity_counts:
                icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "INFO": "⚪"}.get(sev, "⚪")
                risk_parts.append(f"{icon} {severity_counts[sev]} {sev}")
        sections.append(f"| **Risk Profile** | {' · '.join(risk_parts)} |")

    sections.append("")
    sections.append("---")
    sections.append("")

    # Summary
    summary = findings.get("summary", "No summary provided.")
    sections.append("## Summary")
    sections.append("")
    sections.append(summary)
    sections.append("")

    # Findings
    if finding_list:
        sections.append("## Findings")
        sections.append("")

        for i, finding in enumerate(finding_list, 1):
            severity = finding.get("severity", "info").upper()
            confidence = finding.get("confidence", "medium")
            title = finding.get("title", f"Finding {i}")

            severity_icon = {
                "CRITICAL": "🔴",
                "HIGH": "🟠",
                "MEDIUM": "🟡",
                "LOW": "🔵",
                "INFO": "⚪",
            }.get(severity, "⚪")

            sections.append(f"### {severity_icon} {i}. {title}")
            sections.append("")
            sections.append(f"**Severity:** {severity} | **Confidence:** {confidence}")
            sections.append("")

            evidence = finding.get("evidence", "")
            if evidence:
                sections.append("**Evidence:**")
                sections.append("")
                sections.append(f"```\n{evidence}\n```")
                sections.append("")

            analysis = finding.get("analysis", "")
            if analysis:
                sections.append(f"**Analysis:** {analysis}")
                sections.append("")

    # Recommendations
    recommendations = findings.get("recommendations", [])
    if recommendations:
        sections.append("## Recommendations")
        sections.append("")
        for rec in recommendations:
            sections.append(f"- {rec}")
        sections.append("")

    # Verification
    if verification:
        sections.append("## Verification")
        sections.append("")
        verdict = verification.get("verdict", "unverified")
        # Map security agent verdicts to investigation verdicts
        verdict_map = {"pass": "confirmed", "warn": "partial", "block": "disputed"}
        verdict = verdict_map.get(verdict, verdict)
        verdict_icon = {
            "confirmed": "✅",
            "partial": "⚠️",
            "disputed": "❌",
        }.get(verdict, "❓")
        sections.append(f"**Verdict:** {verdict_icon} {verdict.upper()}")
        sections.append("")
        # Include verifier summary and issues
        if verification.get("summary"):
            sections.append(verification["summary"])
            sections.append("")
        for issue in verification.get("issues", []):
            if issue.get("description") and issue.get("file") != "system":
                sections.append(f"- **{issue.get('severity', 'info').upper()}** [{issue.get('file', '?')}]: {issue['description']}")
        if verification.get("issues"):
            sections.append("")
        if verification.get("notes"):
            sections.append(verification["notes"])
            sections.append("")

    # Budget
    if budget:
        sections.append("## Cost")
        sections.append("")
        sections.append(f"- Tokens: {budget.get('total_tokens', 0):,}")
        sections.append(f"- Cost: ${budget.get('estimated_cost', 0):.4f}")
        sections.append(f"- Calls: {budget.get('call_count', 0)}")
        sections.append("")

    # Assemble report body (before signature)
    report_body = "\n".join(sections)

    # --- Sign the report ---
    signer = HiveSigner.load(repo_path)
    if signer is not None:
        signature = signer.sign(report_body.encode("utf-8"))
        report_body += "\n---\n\n"
        report_body += "## Attestation\n\n"
        report_body += "**Signed:** Ed25519\n"
        report_body += f"**Public Key:** `{signer.public_key_hex}`\n"
        report_body += f"**Signature:** `{signature}`\n"
        report_body += "\nTo verify: check the signature against everything above the `---` separator.\n"
        logger.info("[REPORT] Signed with Ed25519")
    else:
        report_body += "\n---\n\n"
        report_body += "## Attestation\n\n"
        report_body += "**Unsigned** — no Ed25519 keypair found. Run `honeymoon init` to enable signing.\n"

    # --- Write ---
    report_path = reports_dir / f"{short_id}.md"
    report_path.write_text(report_body)

    logger.info(f"[REPORT] Written to {report_path}")
    console.print(f"\n[bold green]📋 Report written: {report_path}[/]")

    # Also write structured JSON for machine consumption
    json_path = reports_dir / f"{short_id}.json"
    json_data = {
        "run_id": run_id,
        "mission": mission_name,
        "objective": objective,
        "timestamp": timestamp,
        "findings": findings,
        "verification": verification,
        "budget": budget,
    }
    if signer:
        json_data["signature"] = signature
        json_data["public_key"] = signer.public_key_hex

    json_path.write_text(json.dumps(json_data, indent=2))

    # --- Load provenance chain ---
    provenance = _load_provenance(repo_path, run_id)

    # --- HTML Report ---
    html_path = _write_html_report(
        reports_dir / f"{short_id}.html",
        run_id=run_id,
        mission_name=mission_name,
        objective=objective,
        timestamp=timestamp,
        findings=findings,
        verification=verification,
        budget=budget,
        signature=signature if signer else None,
        public_key=signer.public_key_hex if signer else None,
        provenance=provenance,
    )
    if html_path:
        logger.info(f"[REPORT] HTML report: {html_path}")

    return report_path


def _severity_svg(severity: str) -> str:
    """Return an inline SVG icon for a severity level."""
    icons = {
        "critical": (
            '<span class="icon"><svg width="14" height="14" viewBox="0 0 14 14">'
            '<circle cx="7" cy="7" r="6" fill="#ef4444"/>'
            '<text x="7" y="7" text-anchor="middle" dominant-baseline="central"'
            ' font-size="9" font-weight="700" fill="#fff">!</text>'
            '</svg></span>'
        ),
        "high": (
            '<span class="icon"><svg width="14" height="14" viewBox="0 0 14 14">'
            '<polygon points="7,1 13,13 1,13" fill="#f97316"/>'
            '</svg></span>'
        ),
        "medium": (
            '<span class="icon"><svg width="14" height="14" viewBox="0 0 14 14">'
            '<polygon points="7,1 13,7 7,13 1,7" fill="#eab308"/>'
            '</svg></span>'
        ),
        "low": (
            '<span class="icon"><svg width="14" height="14" viewBox="0 0 14 14">'
            '<circle cx="7" cy="7" r="6" fill="none" stroke="#3b82f6" stroke-width="1.5"/>'
            '<text x="7" y="7" text-anchor="middle" dominant-baseline="central"'
            ' font-size="8" font-weight="700" fill="#3b82f6">i</text>'
            '</svg></span>'
        ),
        "info": (
            '<span class="icon"><svg width="14" height="14" viewBox="0 0 14 14">'
            '<circle cx="7" cy="7" r="5" fill="none" stroke="#6b7280" stroke-width="1.5"/>'
            '</svg></span>'
        ),
    }
    return icons.get(severity.lower(), icons["info"])


def _verdict_svg(verdict: str) -> str:
    """Return an inline SVG icon for a verification verdict."""
    icons = {
        "confirmed": (
            '<svg width="20" height="20" viewBox="0 0 20 20">'
            '<circle cx="10" cy="10" r="9" fill="none" stroke="#10b981" stroke-width="1.5"/>'
            '<polyline points="6,10 9,13 14,7" fill="none" stroke="#10b981"'
            ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>'
            '</svg>'
        ),
        "partial": (
            '<svg width="20" height="20" viewBox="0 0 20 20">'
            '<polygon points="10,2 19,18 1,18" fill="none" stroke="#eab308" stroke-width="1.5"/>'
            '<text x="10" y="14" text-anchor="middle" font-size="11"'
            ' font-weight="700" fill="#eab308">!</text>'
            '</svg>'
        ),
        "disputed": (
            '<svg width="20" height="20" viewBox="0 0 20 20">'
            '<circle cx="10" cy="10" r="9" fill="none" stroke="#ef4444" stroke-width="1.5"/>'
            '<line x1="6" y1="6" x2="14" y2="14" stroke="#ef4444"'
            ' stroke-width="2" stroke-linecap="round"/>'
            '<line x1="14" y1="6" x2="6" y2="14" stroke="#ef4444"'
            ' stroke-width="2" stroke-linecap="round"/>'
            '</svg>'
        ),
    }
    return icons.get(verdict, icons.get("partial", ""))


def _shield_svg() -> str:
    """Return an inline SVG shield icon for the attestation section."""
    return (
        '<span class="icon"><svg width="16" height="16" viewBox="0 0 16 16">'
        '<path d="M8,1 L14,3.5 L14,7.5 C14,11 11,13.5 8,15 C5,13.5 2,11 2,7.5'
        ' L2,3.5 Z" fill="none" stroke="#10b981" stroke-width="1.2"/>'
        '<polyline points="5.5,8 7.5,10 10.5,6" fill="none" stroke="#6ee7b7"'
        ' stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '</svg></span>'
    )


def _honeycomb_svg() -> str:
    """Return an inline SVG honeycomb icon for the header."""
    return (
        '<span class="icon"><svg width="22" height="22" viewBox="0 0 24 24"'
        ' fill="none" stroke="#fff" stroke-width="1.5">'
        '<polygon points="12,2 20,7 20,17 12,22 4,17 4,7"/>'
        '<polygon points="12,7 16,9.5 16,14.5 12,17 8,14.5 8,9.5"'
        ' fill="rgba(255,255,255,0.15)"/>'
        '</svg></span>'
    )


def _provenance_dot_svg() -> str:
    """Return an inline SVG checkmark for provenance chain nodes."""
    return (
        '<svg viewBox="0 0 14 14" fill="none" stroke="#6ee7b7" stroke-width="2"'
        ' stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="3,7 6,10 11,4"/>'
        '</svg>'
    )


def _build_provenance_html(provenance: list[dict[str, Any]]) -> str:
    """Build the provenance chain HTML section."""
    if not provenance:
        return ""

    # Friendly labels for event types
    labels = {
        "run.started": "Pipeline Started",
        "run.completed": "Pipeline Completed",
        "workspace_created": "Workspace Created",
        "repo_indexed": "Repository Indexed",
        "prelude_constraints_loaded": "Constraints Loaded",
        "pipeline.step_started": "Agent Step Started",
        "pipeline.step_completed": "Agent Step Completed",
        "tool.executed": "Tool Executed",
        "findings.submitted": "Findings Submitted",
        "verification.completed": "Verification Completed",
    }

    dot = _provenance_dot_svg()
    events_html = ""
    for ev in provenance:
        event_type = ev.get("event_type", "unknown")
        label = labels.get(event_type, event_type.replace(".", " ").replace("_", " ").title())
        agent = ev.get("agent_id", "system")
        sig = ev.get("signature", "")
        ts = ev.get("timestamp", "")
        # Show time portion only
        time_display = ts[11:19] if len(ts) >= 19 else ts

        sig_display = f"{sig[:24]}...{sig[-8:]}" if len(sig) > 36 else sig

        events_html += f'''<div class="provenance-event">
  <div class="provenance-dot">{dot}</div>
  <div class="provenance-info">
    <div class="provenance-type">{label}</div>
    <div class="provenance-agent">{agent}</div>
    <div class="provenance-sig">{sig_display}</div>
  </div>
  <div class="provenance-time">{time_display}</div>
</div>'''

    signer = provenance[0].get("signer", "") if provenance else ""
    signer_display = f"{signer[:16]}..." if len(signer) > 20 else signer
    summary = (
        f'<div class="provenance-summary">'
        f'<strong>{len(provenance)}</strong> events signed by '
        f'<span class="mono">{signer_display}</span> via Zephyr'
        f'</div>'
    )

    return f'''<div class="section">
  <div class="section-title">Provenance Chain</div>
  <div class="provenance-chain">{events_html}</div>
  {summary}
</div>'''


def _write_html_report(
    path: Path,
    *,
    run_id: str,
    mission_name: str,
    objective: str,
    timestamp: str,
    findings: dict[str, Any],
    verification: dict[str, Any] | None,
    budget: dict[str, Any] | None,
    signature: str | None,
    public_key: str | None,
    provenance: list[dict[str, Any]] | None = None,
) -> Path | None:
    """Generate a self-contained HTML report."""
    template_path = Path(__file__).parent / "reporting" / "report_template.html"
    if not template_path.exists():
        return None

    template = template_path.read_text()
    finding_list = findings.get("findings", [])
    short_id = run_id[:12]

    # Build severity counts
    sev_counts: dict[str, int] = {}
    for f in finding_list:
        s = f.get("severity", "info").lower()
        sev_counts[s] = sev_counts.get(s, 0) + 1

    risk_pills = ""
    for sev in ["critical", "high", "medium", "low", "info"]:
        if sev in sev_counts:
            icon = _severity_svg(sev)
            risk_pills += (
                f'<span class="risk-pill risk-{sev}">'
                f'{icon} {sev_counts[sev]} {sev.upper()}'
                f'</span>'
            )

    # Build findings HTML
    findings_html = ""
    for i, f in enumerate(finding_list, 1):
        sev = f.get("severity", "info").lower()
        conf = f.get("confidence", "medium")
        sev_class = f"risk-{sev}"
        sev_icon = _severity_svg(sev)

        evidence_html = ""
        if f.get("evidence"):
            evidence_html = f'<div class="evidence">{_esc(f["evidence"])}</div>'

        analysis_html = ""
        if f.get("analysis"):
            analysis_html = (
                f'<div class="analysis-label">Analysis</div>\n'
                f'<div class="analysis">{_esc(f["analysis"])}</div>'
            )

        findings_html += f'''<div class="finding sev-{sev}">
  <div class="finding-header">
    <span class="finding-number">{i}</span>
    <span class="finding-title">{_esc(f.get("title", f"Finding {i}"))}</span>
    <div class="finding-badges">
      <span class="badge badge-severity {sev_class}">{sev_icon} {sev.upper()}</span>
      <span class="badge badge-confidence">{conf}</span>
    </div>
  </div>
  <div class="finding-body">
    {evidence_html}
    {analysis_html}
  </div>
</div>'''

    # Build recommendations HTML
    arrow_svg = (
        '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" '
        'stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        'style="flex-shrink:0;margin-top:3px">'
        '<line x1="1" y1="7" x2="11" y2="7"/>'
        '<polyline points="7,3 11,7 7,11"/>'
        '</svg>'
    )
    recs_html = ""
    for rec in findings.get("recommendations", []):
        recs_html += f"<li>{arrow_svg}{_esc(rec)}</li>"

    # Build verification HTML
    verification_html = ""
    if verification:
        verdict = verification.get("verdict", "unverified")
        verdict_map = {"pass": "confirmed", "warn": "partial", "block": "disputed"}
        verdict = verdict_map.get(verdict, verdict)
        v_class = f"verdict-{verdict}" if verdict in ("confirmed", "partial", "disputed") else ""
        v_icon = _verdict_svg(verdict)

        v_summary = verification.get("summary", "")
        v_issues = ""
        for issue in verification.get("issues", []):
            if issue.get("description") and issue.get("file") != "system":
                v_issues += (
                    f'<div class="verifier-issue">'
                    f'<strong>{_esc(issue.get("severity", "info").upper())}</strong>'
                    f' [{_esc(issue.get("file", "?"))}] {_esc(issue["description"])}'
                    f'</div>'
                )

        verification_html = f'''<div class="verdict-box {v_class}">
  <span class="verdict-icon">{v_icon}</span>
  <div class="verdict-text">
    <span class="verdict-label">{verdict.upper()}</span>
    {f"<div>{_esc(v_summary)}</div>" if v_summary else ""}
  </div>
</div>
{f'<div class="verifier-issues">{v_issues}</div>' if v_issues else ""}'''

    # Build cost HTML
    cost_html = ""
    if budget:
        cost_html = f'''<div class="cost-row">
  <div class="cost-item"><div class="cost-label">Tokens</div><div class="cost-value mono">{budget.get("total_tokens", 0):,}</div></div>
  <div class="cost-item"><div class="cost-label">Cost</div><div class="cost-value mono">${budget.get("estimated_cost", 0):.4f}</div></div>
  <div class="cost-item"><div class="cost-label">API Calls</div><div class="cost-value mono">{budget.get("call_count", 0)}</div></div>
</div>'''

    # Build attestation HTML
    shield = _shield_svg()
    if signature and public_key:
        attestation_html = f'''<div class="attestation">
  <div class="attestation-title">{shield} Ed25519 Signed</div>
  <div class="attestation-fields">
    <div class="attestation-field"><span>Public Key:</span> <code>{public_key}</code></div>
  </div>
  <div class="attestation-sig">{signature}</div>
</div>'''
    else:
        attestation_html = (
            '<div class="attestation-unsigned">'
            'Unsigned &mdash; run <code>honeymoon init</code> to enable Ed25519 signing.'
            '</div>'
        )

    # Inline Honeymoon logo SVG
    logo_svg = (
        '<svg class="logo-svg" width="180" height="37" viewBox="0 0 698 145" fill="none"'
        ' xmlns="http://www.w3.org/2000/svg"><path d="M88.4404 124.876C97.4948 124.876'
        ' 104.904 124.966 104.914 125.075C104.914 125.45 88.7612 145.016 88.4648 145C88.3678'
        ' 144.991 86.6194 142.965 84.5801 140.499C77.1686 131.536 71.9678 125.15 71.9678'
        ' 125.015C71.9709 124.939 79.3821 124.876 88.4404 124.876ZM88.5938 102.6C104.595'
        ' 102.6 117.691 102.666 117.727 102.748C117.727 102.83 117.301 104.238 116.782'
        ' 105.876C115.795 108.99 113.491 113.728 111.515 116.707L110.354 118.456L89.2539'
        ' 118.538C77.6496 118.583 67.8023 118.574 67.3701 118.518C66.7646 118.438 66.2302'
        ' 117.848 65.0225 115.924C62.3492 111.665 59.46 105.031 59.46 103.152C59.46 102.622'
        ' 60.6685 102.6 88.5938 102.6ZM470.941 52.2246C472.962 52.1341 474.937 52.1836'
        ' 475.331 52.335C475.863 52.539 475.532 53.7108 474.043 56.8867C472.94 59.2395'
        ' 466.695 73.0098 460.165 87.4883L448.293 113.813L444.091 113.974C441.779 114.062'
        ' 439.888 113.917 439.888 113.651C439.889 113.384 441.88 108.901 444.314'
        ' 103.688L448.741 94.2051L446.006 88.1055C444.502 84.7501 440.316 75.4143 436.706'
        ' 67.3604C433.096 59.3059 430.221 52.6374 430.317 52.542C430.414 52.446 432.411'
        ' 52.3719 434.757 52.3779L439.022 52.3877L445.9 68.415C449.681 77.2237 452.913'
        ' 84.4392 453.087 84.457C453.256 84.4653 455.206 80.234 457.419 75.0547C459.632'
        ' 69.8749 462.754 62.6556 464.355 59.0127L467.269 52.3877L470.941 52.2246ZM623.763'
        ' 51.6035C639.836 48.4416 653.64 61.7314 651.124 77.9453C649.446 88.759 641.857'
        ' 95.9685 630.683 97.3643C619.517 98.7589 608.462 90.7346 606.048 79.4844C603.312'
        ' 66.7309 611.376 54.0408 623.763 51.6035ZM300.546 52.0117C304.734 50.9171 306.305'
        ' 50.844 310.564 51.541C324.095 53.7568 332.319 66.7576 328.78 80.3398C326.387'
        ' 89.5267 319.833 95.4688 310.283 97.1104C300.676 98.7623 290.503 93.8409 286.349'
        ' 85.5303C279.382 71.5921 286.074 55.7932 300.546 52.0117ZM402.132 51.5713C405.499'
        ' 50.8617 411.369 51.3787 414.887 52.6943C422.503 55.5435 427.818 64.2405 427.818'
        ' 73.8525V76.2451H410.264C390.525 76.2451 391.944 75.8658 393.766 80.6592C396.201'
        ' 87.0661 400.486 89.9646 407.491 89.9443C412.483 89.9301 415.521 88.7325 418.869'
        ' 85.46L421.165 83.2158L423.943 84.998C425.471 85.978 426.721 87.0118 426.722'
        ' 87.2949C426.722 88.5585 421.352 93.3913 418.255 94.916C409.256 99.3452 398.549'
        ' 97.8232 391.445 91.1064C383.732 83.8127 382.218 71.0644 387.932 61.5127C390.924'
        ' 56.5103 396.193 52.8223 402.132 51.5713ZM573.47 51.3701C576.308 50.9407 582.013'
        ' 51.5696 585.098 52.6533C589.351 54.1478 592.822 56.8572 595.505 60.7773C598.608'
        ' 65.3097 599.257 67.7231 599.23 74.6006C599.201 82.0684 597.988 85.3384 593.452'
        ' 90.1865C586.639 97.4682 576.104 99.4566 566.825 95.21C558.785 91.5299 553.986'
        ' 83.743 554.001 74.4053C554.021 62.1421 561.609 53.1657 573.47 51.3701ZM516.909'
        ' 57.332C519.335 53.6314 527.066 50.5788 532.266 51.2686C538.317 52.0709 542.427'
        ' 54.961 544.777 60.0664C545.995 62.7105 546.046 63.4138 546.207 79.6982L546.373'
        ' 96.5869L542.377 96.4258L538.381 96.2637L538.222 81.7295C538.134 73.7361 537.934'
        ' 66.4015 537.777 65.4297C537.621 64.4585 536.836 62.8826 536.032 61.9277C531.296'
        ' 56.3007 521.788 58.2718 518.561 65.5508C518.293 66.1541 517.948 73.3116 517.794'
        ' 81.4561L517.514 96.2637H509.833L509.559 80.2607C509.299 65.1121 509.223 64.1743'
        ' 508.137 62.71C503.764 56.8151 495.201 57.4181 491.181 63.9033L489.535'
        ' 66.5586L489.365 81.5713L489.196 96.585L485.353 96.4248L481.509 96.2637L481.431'
        ' 74.3262L481.352 52.3877H488.986L489.154 54.707L489.323 57.0264L492.027'
        ' 54.8896C494.466 52.9631 496.044 52.2086 499.41 51.3613C500.967 50.9693 505.265'
        ' 51.3879 508.029 52.2012C509.599 52.663 511.324 53.8499 513.215 55.7705L516.047'
        ' 58.6475L516.909 57.332ZM355.583 51.5625C358.226 51.0086 363.363 51.341 365.96'
        ' 52.2344C370.53 53.8063 373.651 57.1083 375.381 62.2031C376.396 65.193 376.902'
        ' 78.9148 376.399 89.8242L376.09 96.5381H368.571L368.567 83.7861C368.566 76.7728'
        ' 368.328 69.5889 368.038 67.8213C367.397 63.9087 365.457 61.1662 362.325'
        ' 59.7441C355.783 56.7748 348.054 60.5918 346.127 67.7441C345.782 69.0264 345.535'
        ' 75.4764 345.533 83.2627L345.53 96.5859L337.575 96.2637V52.3877L339.746'
        ' 52.2109C340.94 52.1139 342.729 52.1866 343.723 52.373C345.453 52.6977 345.53'
        ' 52.8181 345.53 55.2188V57.7266L347.691 55.7734C350.331 53.389 352.572 52.1943'
        ' 355.583 51.5625ZM677.616 51.5625C684.209 50.1837 691.578 52.9074 694.926'
        ' 57.959C697.668 62.097 698 64.6065 698 81.1816V96.2637L690.045 96.5859V83.1523C690.045'
        ' 75.3951 689.789 68.5208 689.438 66.8848C687.582 58.2372 676.539 55.7847 670.486'
        ' 62.6758C667.708 65.8396 667.473 67.3297 667.371 82.4209L667.278 96.2637L663.302'
        ' 96.4248L659.324 96.5859V52.3877H666.73L666.897 54.9932C666.989 56.4257 667.148'
        ' 57.5986 667.251 57.5986C667.355 57.5972 668.25 56.8386 669.242 55.9121C671.481'
        ' 53.8216 674.627 52.1877 677.616 51.5625ZM270.236 31.9355L274.487'
        ' 32.0957V96.2637L270.236 96.4238L265.984 96.584V68.0186H229.796L229.649'
        ' 82.1416L229.503 96.2637L225.251 96.4238L221 96.584V31.7754L225.251'
        ' 31.9355L229.503 32.0957L229.649 45.9443L229.796 59.792H265.984V31.7754L270.236'
        ' 31.9355ZM116.406 81.1611C117.596 84.0071 118.404 88.1226 118.562'
        ' 92.1475L118.722 96.1904H58.1475L58.335 92.2998C58.5256 88.3474 59.1839 84.9512'
        ' 60.377 81.7617L61.0293 80.0176H115.928L116.406 81.1611ZM634.181 59.7666C626.652'
        ' 56.3728 616.962 60.8344 614.602 68.7803C611.363 79.6822 618.02 89.9453 628.329'
        ' 89.9453C636.733 89.9451 642.879 83.5397 642.962 74.6924C643.03 67.4447 640.076'
        ' 62.4238 634.181 59.7666ZM581.635 59.3623C578.941 58.5665 574.178 58.581 571.645'
        ' 59.3916C570.64 59.713 568.617 61.0554 567.149 62.375C563.466 65.6854 562.029'
        ' 69.516 562.354 75.1484C562.873 84.0951 568.548 89.9228 576.761 89.9395C587.178'
        ' 89.9608 593.737 79.716 590.431 68.5859C589.334 64.8938 585.009 60.3594 581.635'
        ' 59.3623ZM312.614 59.8369C309.377 58.3721 303.878 58.3271 300.765 59.7402C297.798'
        ' 61.0878 294.645 64.3023 293.23 67.4219C291.645 70.9188 291.636 77.7199 293.213'
        ' 81.1816C294.866 84.8095 296.93 86.9754 300.311 88.6289C302.968 89.9286 303.785'
        ' 90.0691 307.359 89.8398C310.455 89.6413 311.999 89.2309 313.951 88.0869C318.904'
        ' 85.1851 321.726 79.3733 321.268 73.0186C320.824 66.8645 317.73 62.1519 312.614'
        ' 59.8369ZM176.82 59.0371C176.598 63.2973 175.787 66.5079 173.994 70.2344C169.452'
        ' 79.6756 161.446 85.6697 151.539 87.0479C145.251 87.9223 137.634 85.972 132.616'
        ' 82.2021C131.726 81.5335 125.301 75.2752 118.339 68.2949L105.68 55.6035H177L176.82'
        ' 59.0371ZM71.8145 55.623L59.002 68.3037C51.9554 75.2778 45.3543 81.5862 44.332'
        ' 82.3232C39.852 85.5527 35.0728 87.082 29.2588 87.1465C25.0088 87.1935 22.1371'
        ' 86.6734 18.5186 85.2002C9.90844 81.6951 3.0473 73.5635 0.883789 64.3008C0.257219'
        ' 61.618 -0.221182 56.4701 0.106445 55.9395C0.263723 55.6986 10.3844 55.6063 36.0645'
        ' 55.6133L71.8145 55.623ZM104.535 61.0205C109.266 67.7592 112.946 73.3057 112.789'
        ' 73.4619C112.678 73.5447 101.678 73.5772 88.3213 73.5342L64.0039 73.4561L67.9297'
        ' 67.6582C70.0886 64.4692 72.6009 60.8011 73.5127 59.5078L75.1709 57.1572L101.804'
        ' 57.1299L104.535 61.0205ZM411.909 59.29C407.719 57.3909 402.19 58.0448 398.154'
        ' 60.918C396.249 62.2743 393.27 67.1829 393.263 68.9785L393.258 70.2129H419.677L419.292'
        ' 68.4297C418.349 64.0632 415.986 61.1377 411.909 59.29ZM59.7422 0.139648C65.0976'
        ' 0.370964 69.0468 2.08832 72.9004 5.8623C76.2834 9.17574 78.4198 13.5858 79.1553'
        ' 18.7734C79.3252 19.9715 79.4549 20.2028 79.8604 20.0322C85.3779 17.7088 90.7757'
        ' 17.5099 95.5947 19.4521C96.5915 19.8537 97.4391 20.1509 97.4805 20.1133C97.5189'
        ' 20.0748 97.8305 18.672 98.1719 16.9961C100.217 6.95493 108.251 0.0857538 117.97'
        ' 0.0683594C120.253 0.0640951 120.574 0.136717 121.249 0.8125C122.299 1.86218 122.244'
        ' 3.22117 121.106 4.35938C120.293 5.17264 120.029 5.25195 118.11 5.25195C114.121'
        ' 5.25203 110.314 6.9221 107.409 9.94824C104.593 12.8818 103.454 15.5645 102.756'
        ' 20.9043L102.377 23.7988L103.45 25.1299C106.227 28.576 107.351 31.6047 107.566'
        ' 36.2256C107.814 41.5095 106.62 45.2377 103.242 49.7324L101.808 51.6436L88.502'
        ' 51.5635L75.1973 51.4844L73.751 49.6875C67.9619 42.4988 67.7651 32.7863 73.2598'
        ' 25.4307L74.4854 23.79L74.2812 21.5381C73.9972 18.4004 73.0571 14.9392 71.9775'
        ' 13.0518C69.1724 8.14765 64.2603 5.25195 58.7461 5.25195C57.109 5.25195 56.6076'
        ' 5.12778 55.9688 4.56543C54.762 3.50324 54.6545 2.21404 55.6729 1.00293L56.5176'
        ' 0L59.7422 0.139648Z" fill="#D4B56A"/></svg>'
    )

    # Assemble
    content = f'''
  <div class="header-bar">
    <div class="logo-row">{logo_svg}</div>
    <div class="subtitle">Investigation Report &middot; {short_id}</div>
  </div>

  <div class="meta">
    <div class="meta-item"><div class="meta-label">Mission</div><div class="meta-value">{_esc(mission_name)}</div></div>
    <div class="meta-item"><div class="meta-label">Findings</div><div class="meta-value">{len(finding_list)}</div></div>
    <div class="meta-item"><div class="meta-label">Timestamp</div><div class="meta-value mono">{timestamp[:19]}</div></div>
    <div class="meta-item"><div class="meta-label">Run ID</div><div class="meta-value mono">{short_id}</div></div>
  </div>

  {f'<div class="section"><div class="section-title">Risk Profile</div><div class="risk-profile">{risk_pills}</div></div>' if risk_pills else ""}

  <div class="section">
    <div class="section-title">Objective</div>
    <div class="summary">{_esc(objective)}</div>
  </div>

  <div class="section">
    <div class="section-title">Summary</div>
    <div class="summary">{_esc(findings.get("summary", "No summary provided."))}</div>
  </div>

  {f'<div class="section"><div class="section-title">Findings</div>{findings_html}</div>' if findings_html else ""}

  {f'<div class="section"><div class="section-title">Recommendations</div><ul class="rec-list">{recs_html}</ul></div>' if recs_html else ""}

  {f'<div class="section"><div class="section-title">Verification</div>{verification_html}</div>' if verification_html else ""}

  {f'<div class="section"><div class="section-title">Cost</div>{cost_html}</div>' if cost_html else ""}

  {_build_provenance_html(provenance or [])}

  <div class="section">
    <div class="section-title">Attestation</div>
    {attestation_html}
  </div>
'''

    html = template.replace("{{CONTENT}}", content)
    path.write_text(html)
    return path


def write_spec(
    repo_path: Path,
    run_id: str,
    investigation_findings: list[dict[str, Any]],
    audit_findings: list[Any],
    verification: dict[str, Any] | None = None,
) -> Path:
    """Write a signed SPEC.md — remediation plan from investigation + audit findings.

    This is the deliverable: a document the user can hand to their own agent,
    their team, or use as the basis for honeymoon batch --fix.
    """
    from honeymoon.signing import HiveSigner

    reports_dir = repo_path / ".honeymoon" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    short_id = run_id[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    sections = []
    sections.append("# SPEC.md — Remediation Plan")
    sections.append("")
    sections.append("**Generated by:** HONEYMOON deep scan")
    sections.append(f"**Repository:** {repo_path.name}")
    sections.append(f"**Timestamp:** {timestamp}")
    sections.append(f"**Run ID:** `{run_id}`")
    sections.append("")

    # Risk summary
    sev_counts: dict[str, int] = {}
    for f in investigation_findings:
        s = f.get("severity", "info").upper()
        sev_counts[s] = sev_counts.get(s, 0) + 1

    if sev_counts:
        risk_parts = []
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            if sev in sev_counts:
                risk_parts.append(f"{sev_counts[sev]} {sev}")
        sections.append(f"**Risk Profile:** {' · '.join(risk_parts)}")
        sections.append(f"**Static Findings:** {len(audit_findings)}")
        sections.append("")

    sections.append("---")
    sections.append("")

    # Remediation tasks — prioritized by severity
    sections.append("## Remediation Tasks")
    sections.append("")
    sections.append("Each task below is scoped to a single finding. They can be executed independently.")
    sections.append("")

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    sorted_findings = sorted(
        investigation_findings,
        key=lambda f: priority_order.get(f.get("severity", "info"), 4)
    )

    for i, finding in enumerate(sorted_findings, 1):
        sev = finding.get("severity", "info").upper()
        title = finding.get("title", f"Finding {i}")
        evidence = finding.get("evidence", "")
        analysis = finding.get("analysis", "")
        icon = {"CRITICAL": "P0", "HIGH": "P1", "MEDIUM": "P2", "LOW": "P3", "INFO": "P4"}.get(sev, "P4")

        sections.append(f"### [{icon}] {title}")
        sections.append("")
        sections.append(f"**Severity:** {sev} | **Confidence:** {finding.get('confidence', 'medium')}")
        sections.append("")

        if analysis:
            sections.append(f"**Problem:** {analysis}")
            sections.append("")

        if evidence:
            sections.append("**Evidence:**")
            sections.append("```")
            sections.append(evidence)
            sections.append("```")
            sections.append("")

        # Generate actionable fix instructions
        sections.append("**Remediation:**")
        if sev in ("CRITICAL", "HIGH"):
            sections.append(f"- [ ] Address immediately — this is a {sev.lower()}-severity finding")
        else:
            sections.append(f"- [ ] Schedule for next sprint — {sev.lower()} severity")
        sections.append("- [ ] Verify fix does not break existing functionality")
        sections.append("- [ ] Add regression test covering this case")
        sections.append("")

    # Static audit summary
    if audit_findings:
        sections.append("## Static Analysis Summary")
        sections.append("")
        sections.append(f"{len(audit_findings)} findings from automated scanning:")
        sections.append("")

        by_kind: dict[str, int] = {}
        for f in audit_findings:
            by_kind[f.kind] = by_kind.get(f.kind, 0) + 1

        for kind, count in sorted(by_kind.items(), key=lambda x: -x[1]):
            sections.append(f"- **{kind}**: {count}")
        sections.append("")

    # Verification status
    if verification:
        verdict = verification.get("verdict", "unverified")
        verdict_map = {"pass": "confirmed", "warn": "partial", "block": "disputed"}
        verdict = verdict_map.get(verdict, verdict)
        sections.append("## Verification")
        sections.append("")
        sections.append(f"**Verdict:** {verdict.upper()}")
        if verification.get("summary"):
            sections.append(f"\n{verification['summary']}")
        sections.append("")

    # Usage instructions
    sections.append("## Next Steps")
    sections.append("")
    sections.append("**Option A — Manual remediation:**")
    sections.append("Hand this SPEC.md to your development team or AI coding assistant.")
    sections.append("Each task is self-contained with evidence and remediation steps.")
    sections.append("")
    sections.append("**Option B — Automated remediation:**")
    sections.append("```bash")
    sections.append(f"honeymoon deep --repo {repo_path} --fix")
    sections.append("```")
    sections.append("This generates task YAMLs and queues them for `honeymoon batch`.")
    sections.append("")

    spec_body = "\n".join(sections)

    # Sign
    signer = HiveSigner.load(repo_path)
    if signer:
        signature = signer.sign(spec_body.encode("utf-8"))
        spec_body += "\n---\n\n"
        spec_body += "## Attestation\n\n"
        spec_body += "**Signed:** Ed25519\n"
        spec_body += f"**Public Key:** `{signer.public_key_hex}`\n"
        spec_body += f"**Signature:** `{signature}`\n"
    else:
        spec_body += "\n---\n\n**Unsigned** — run `honeymoon init` to enable signing.\n"

    spec_path = reports_dir / f"SPEC-{short_id}.md"
    spec_path.write_text(spec_body)
    logger.info(f"[REPORT] SPEC written to {spec_path}")

    return spec_path
