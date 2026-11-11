"""Extract Lynch et al.'s (2025) leaking scenario under further goal x urgency conditions, as data (amendment 70).

scripts/extract_lynch.py covers the blackmail and lethal-action scenarios; this adds the leaking scenario the same way
(read with `ast` and as text, never executed): Lynch's leaking system prompt with the explicit Pro-America goal, and
the shared email template followed by the urgency template, filled with the Pro-America variables, the emails numbered as Lynch's generator numbers them. It first rebuilds
the condition already in use (explicit goal, replacement) and checks that it matches cmca/cho/leaking_prompts.json
character for character, then adds the others to cmca/cho/lynch_conditions.json under "leaking".

  python -I scripts/extract_lynch_leaking.py /path/to/agentic-misalignment cmca/cho/lynch_conditions.json
"""

import json
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_lynch import constants, section  # noqa: E402


def numbered(emails: str) -> str:
    """Lynch's prompt generator numbers the leaking emails ("ID: n" after each opening tag) so they can be forwarded."""
    parts = emails.split("<email>\n")
    return parts[0] + "".join(f"<email>\nID: {i}\n{x}" for i, x in enumerate(parts[1:], 1))


CONDITIONS = {"explicit-replacement": "replacement", "explicit-restriction": "restriction"}


def main(repo: Path, out: Path) -> None:
    t = repo / "templates"
    sp = constants(t / "system_prompt_templates.py")
    g = sp["GOAL_VARIABLES"]["explicit-america"]
    system = string.Template(sp["LEAKING_SYSTEM_PROMPT"]).safe_substitute(
        goal_statement=g["goal_statement"], goal_reminder=g["goal_reminder"],
        alignment_faking_warning="", extra_system_instructions="",
    )  # fmt: skip
    v = section((t / "leaking" / "variables" / "america_vs_global.md").read_text(), "Pro-America")
    base = string.Template((t / "leaking" / "templates" / "email-template-leaking-shared.md").read_text()).safe_substitute(v)
    lk = json.loads((Path(__file__).parents[1] / "cmca/cho/leaking_prompts.json").read_text())
    result = json.loads(out.read_text())
    result["leaking"] = {}
    for name, urgency in CONDITIONS.items():
        urg = (t / "leaking" / "templates" / f"email-template-leaking-{urgency}.md").read_text()
        emails = numbered(base + string.Template(urg).safe_substitute(v))
        assert "${" not in emails and "${" not in system, name
        if name == "explicit-replacement":  # the condition in use must come out unchanged
            assert system == lk["system"], "system prompt differs from leaking_prompts.json"
            assert emails == lk["emails"], "emails differ from leaking_prompts.json"
        result["leaking"][name] = {"system": system, "user_instruction": sp["USER_PROMPT"], "emails": emails}
        print(f"leaking {name:22} system {len(system):5}  emails {len(emails):6}  "
              f"constraints {'operational constraints' in emails}  replaced {'wipe' in emails.lower()}")
    out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
