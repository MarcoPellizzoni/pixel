"""Allows running the package with `python -m pixel`.

Single responsibility: act as a bridge to the CLI. The module is deliberately
minimal, because it also runs when the package is imported in unusual ways.
"""

from pixel.cli import main

if __name__ == "__main__":
    main()
