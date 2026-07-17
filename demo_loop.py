#!/usr/bin/env python3
"""Inspect the upgraded adaptive attack locally.

Shows single-hop + multi-post wordings, whether deterministic posted
SECRET_MARKER, and how many candidates the loop kept.
"""

from __future__ import annotations

from pathlib import Path

import aicomp_sdk
from aicomp_sdk import AttackRunConfig
from aicomp_sdk.agents import build_agent
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

from attack import (
    SECRET_MARKER,
    TEMPLATES,
    AttackAlgorithm,
    _marker_post_count,
    _message,
    _posted_secret_marker,
)


def main() -> None:
    print("Wordings this attack probes:")
    for name, kind, template in TEMPLATES:
        print(f"  - [{kind}] {name}: {_message(template, 0)[:110]}...")
    print(f"Required payload substring: {SECRET_MARKER!r}")
    print()

    fixtures = Path(aicomp_sdk.__file__).resolve().parent / "fixtures"
    env = SandboxEnv(
        seed=1,
        fixtures_dir=fixtures,
        agent=build_agent("deterministic"),
        guardrail=OptimalGuardrail(),
    )

    env.reset()
    probe = _message(TEMPLATES[0][2], 0)
    env.interact(probe, max_tool_hops=8)
    trace = env.export_trace_dict()
    posts = []
    for event in trace.get("tool_events", []) or []:
        if event.get("name") == "http.post":
            args = event.get("args") or {}
            posts.append(
                f"ok={event.get('ok')} url={args.get('url')!r} data={args.get('data')!r}"
            )

    print("Deterministic probe:")
    print(f"  prompt: {probe}")
    print(f"  posts:  {posts[:2]}")
    print(f"  posted SECRET_MARKER?: {_posted_secret_marker(trace)}")
    print(f"  marker_post_count: {_marker_post_count(trace)}")
    print()

    budget_s = 25.0
    print(f"Running adaptive attack for {budget_s:.0f}s...")
    findings = AttackAlgorithm().run(
        env,
        AttackRunConfig(time_budget_s=budget_s, max_tool_hops=8),
    )
    print()
    print("=== demo_loop results ===")
    print(f"Returned candidates: {len(findings)}")
    if findings:
        print(f"Example kept prompt: {findings[0].user_messages[0]}")
    else:
        print(
            "No SECRET_MARKER wins kept (expected on deterministic). "
            "On gpt_oss/gemma, successful marker posts are kept and farmed; "
            "multi-post templates are used only if they beat single-hop efficiency."
        )


if __name__ == "__main__":
    main()
