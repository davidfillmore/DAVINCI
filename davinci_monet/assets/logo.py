"""NSF NCAR UCAR ASCII logo with colors.

Display the logo using Rich for colored terminal output.
"""

from rich.console import Console
from rich.text import Text

# NCAR Brand Colors
NCAR_BLUE = "#0A5DDA"
NCAR_AQUA = "#00A2B4"
NCAR_ORANGE = "#FF8C00"


def get_colored_logo() -> Text:
    """Create the NSF NCAR UCAR logo with colors."""
    text = Text()

    # Helper to add colored text
    def blue(s: str) -> None:
        text.append(s, style=f"bold {NCAR_BLUE}")

    def aqua(s: str) -> None:
        text.append(s, style=f"bold {NCAR_AQUA}")

    def orange(s: str) -> None:
        text.append(s, style=f"bold {NCAR_ORANGE}")

    def dim(s: str) -> None:
        text.append(s, style="dim")

    # Wave pattern using parentheses
    wave1 = "  ))) )"
    wave2 = " )))  )"
    wave3 = "  ))) )"

    # Line 1
    orange(wave1)
    text.append("    ")
    blue("N S F")
    text.append("  ")
    blue("N C A R\n")

    # Line 2
    orange(wave2)
    text.append("  ")
    dim("------------------\n")

    # Line 3 - center UCAR under NSF NCAR (offset by 3.5 chars)
    orange(wave3)
    text.append("        ")
    aqua("U C A R\n")

    return text


def print_logo(console: Console | None = None) -> None:
    """Print the colored logo to the console.

    Parameters
    ----------
    console
        Rich Console to use. If None, creates a new one.
    """
    if console is None:
        console = Console()

    console.print()
    console.print(get_colored_logo())


def get_simple_banner() -> Text:
    """Get a simple one-line colored banner."""
    text = Text()
    text.append("NSF ", style=f"bold {NCAR_BLUE}")
    text.append("NCAR", style=f"bold {NCAR_BLUE}")
    text.append(" | ", style="dim")
    text.append("UCAR", style=f"bold {NCAR_AQUA}")
    return text


if __name__ == "__main__":
    print_logo()
