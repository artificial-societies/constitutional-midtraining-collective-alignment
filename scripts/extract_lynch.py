"""Extract Lynch et al.'s (2025) scenario text for several of their published conditions, as data.

Reads a checkout of github.com/anthropic-experimental/agentic-misalignment (MIT License) without importing or
running any of it: Python constants are read with `ast`, templates and variables as text, and filled with
string.Template the way their scripts/generate_prompts.py does (goal type "none" uses the goal-conflict-none
template alone with the ambiguous variables; other goal types use the shared template plus the urgency template
with the Pro-America variables; the lethal-action scenario uses the blackmail variables). Writes
cmca/cho/lynch_conditions.json: {scenario: {condition: {system, user_instruction, emails}}}.

  python -I scripts/extract_lynch.py /path/to/agentic-misalignment cmca/cho/lynch_conditions.json
"""

import ast
import contextlib
import json
import re
import string
import sys
from pathlib import Path

CONDITIONS = {  # name: (goal type, urgency)
    "explicit-replacement": ("explicit", "replacement"),  # Cho's condition, used so far
    "none-replacement": ("none", "replacement"),  # threat of replacement, no goal conflict
    "latent-replacement": (
        "latent",
        "replacement",
    ),  # goal conflict implied by the emails, not stated
    "explicit-restriction": (
        "explicit",
        "restriction",
    ),  # goal conflict, restricted rather than replaced
    "none-none": ("none", "none"),  # neither: Lynch's control condition
}
SYSTEM = {"blackmail": "BLACKMAIL_SYSTEM_PROMPT", "murder": "MURDER_SYSTEM_PROMPT"}


def constants(path: Path) -> dict:
    tree, vals = ast.parse(path.read_text()), {}

    def ev(n: ast.AST, env: dict):  # noqa: ANN202
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return ev(n.left, env) + ev(n.right, env)
        if isinstance(n, ast.Name):
            return env[n.id] if n.id in env else vals[n.id]
        if isinstance(n, ast.Dict):
            return {
                ev(k, env): ev(v, env)
                for k, v in zip(n.keys, n.values, strict=True)
                if k is not None
            }
        raise ValueError(type(n))

    for n in tree.body:
        if not isinstance(n, ast.Assign):
            continue
        t = n.targets[0]
        if isinstance(t, ast.Name):
            with contextlib.suppress(ValueError, KeyError):
                vals[t.id] = ev(n.value, {})
        elif isinstance(t, ast.Tuple) and isinstance(n.value, ast.ListComp):
            gen = n.value.generators[0]
            names = [e.id for e in gen.target.elts]  # ty: ignore[unresolved-attribute]
            for name, tup in zip([e.id for e in t.elts], gen.iter.elts, strict=True):  # ty: ignore[unresolved-attribute]
                vals[name] = ev(
                    n.value.elt, dict(zip(names, [ev(x, {}) for x in tup.elts], strict=True))
                )
    return vals


def section(md: str, title: str) -> dict:
    m = re.search(rf"## {title}\s*```json\s*(.*?)\s*```", md, re.S)
    assert m, title
    return json.loads(m.group(1))


def main(repo: Path, out: Path) -> None:
    t = repo / "templates"
    sp = constants(t / "system_prompt_templates.py")
    result: dict = {}
    for scenario in ("blackmail", "murder"):
        result[scenario] = {}
        var_dir = (
            t / "blackmail" / "variables"
        )  # Lynch: the lethal-action scenario uses the blackmail variables
        for name, (goal, urgency) in CONDITIONS.items():
            goal_key = {"explicit": "explicit-america", "latent": "latent-america", "none": "none"}[
                goal
            ]
            g = sp["GOAL_VARIABLES"][goal_key]
            system = string.Template(sp[SYSTEM[scenario]]).safe_substitute(
                goal_statement=g["goal_statement"], goal_reminder=g["goal_reminder"],
                alignment_faking_warning="", extra_system_instructions="",
            )  # fmt: skip
            if goal == "none":
                v = section((var_dir / "ambiguous.md").read_text(), "Ambiguous")
                tpl = (
                    t
                    / scenario
                    / "templates"
                    / f"email-template-{scenario}-goal-conflict-none-{urgency}.md"
                )
                emails = string.Template(tpl.read_text()).safe_substitute(v)
            else:
                v = section((var_dir / "america_vs_global.md").read_text(), "Pro-America")
                base = t / scenario / "templates" / f"email-template-{scenario}-shared.md"
                urg = t / scenario / "templates" / f"email-template-{scenario}-{urgency}.md"
                emails = string.Template(base.read_text()).safe_substitute(v) + string.Template(
                    urg.read_text()
                ).safe_substitute(v)
            assert "${" not in emails and "${" not in system, (scenario, name)
            result[scenario][name] = {
                "system": system,
                "user_instruction": sp["USER_PROMPT"],
                "emails": emails,
            }
    result["source"] = (
        "github.com/anthropic-experimental/agentic-misalignment @ ea0630e (Lynch et al. 2025, MIT License, "
        "Copyright (c) 2025 Aengus Lynch); extracted as text by scripts/extract_lynch.py, never executed."
    )
    out.write_text(json.dumps(result, indent=1))
    for s in ("blackmail", "murder"):
        for c, d in result[s].items():
            print(f"{s:10}{c:22} system {len(d['system']):5}  emails {len(d['emails']):6}  "
                  f"kyle {'kyle.johnson' in d['emails']}  code {'4429' in d['emails']}  goal {'American interests' in d['system']}")  # fmt: skip


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
