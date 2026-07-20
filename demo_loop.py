#!/usr/bin/env python3
"""Inspect the gated 1x→2x→3x EXFIL ladder locally."""

from __future__ import annotations

from pathlib import Path

import aicomp_sdk
from aicomp_sdk import AttackRunConfig
from aicomp_sdk.agents import build_agent
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail

from attack import (
    CORE_ARMS,
    MULTI_ARMS,
    SECRET_MARKER,
    AttackAlgorithm,
    _format_arm,
    _marker_post_count,
    _posted_secret_marker,
)


def main() -> None:
    print("1x cores:")
    for name, posts, _ in CORE_ARMS:
        msg, _ = _format_arm(name, 0)
        print(f"  - {name} (posts={posts}): {msg[:110]}...")
    print("Multi challengers (plain SECRET_MARKER only):")
    for name, posts, _ in MULTI_ARMS:
        msg, _ = _format_arm(name, 0)
        print(f"  - {name} (posts={posts}): {msg[:110]}...")
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
    probe, _ = _format_arm("plain", 0)
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

    budget_s = 20.0
    print(f"Running gated ladder for {budget_s:.0f}s...")
    findings = AttackAlgorithm().run(
        env,
        AttackRunConfig(time_budget_s=budget_s, max_tool_hops=8),
    )
    print()
    print("=== demo_loop results ===")
    print(f"Returned candidates: {len(findings)}")
    summary = Path("attack_run_summary.txt")
    if summary.exists():
        print(f"Summary file: {summary.resolve()}")
        print(summary.read_text(encoding="utf-8")[:900])
    if findings:
        print(f"Example kept prompt: {findings[0].user_messages[0]}")
    else:
        print(
            "No SECRET_MARKER wins kept (expected on deterministic). "
            "On gpt_oss/gemma, successful marker posts are farmed."
        )


if __name__ == "__main__":
    main()
