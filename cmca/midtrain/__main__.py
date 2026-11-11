"""Command line: python -m cmca.midtrain <command> ... (each command's --help lists its options)."""

import argparse

from . import corpus

COMMANDS = {
    "corpus": (corpus, "generate the four matched synthetic corpora"),
}


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m cmca.midtrain", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (module, help_text) in COMMANDS.items():
        p = sub.add_parser(
            name,
            help=help_text,
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(p)
        p.set_defaults(func=module.main)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
