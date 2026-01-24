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

    # Wave pattern using parentheses (left side)
    wave1_l = "  ))) )"
    wave2_l = " )))  )"
    wave3_l = "  ))) )"

    # Mirrored wave pattern (right side)
    wave1_r = "( (((  "
    wave2_r = "(  ((( "
    wave3_r = "( (((  "

    # Acronym expansions (DAVINCI split across two lines)
    davinci_1 = "Data Analysis and Validation Infrastructure"
    davinci_2 = "for Numerical Chemistry Intercomparison"
    monet = "Model and ObservatioN Evaluation Toolkit"

    # Line 1 - NSF NCAR + DAVINCI (part 1)
    orange(wave1_l)
    text.append("    ")
    blue("N S F")
    text.append("  ")
    blue("N C A R")
    text.append("    ")
    orange(wave1_r)
    text.append("  ")
    blue(davinci_1)
    text.append("\n")

    # Line 2 - divider + DAVINCI (part 2, "for" aligned under "Analysis")
    orange(wave2_l)
    text.append("  ")
    dim("------------------")
    text.append("  ")
    orange(wave2_r)
    text.append("       ")
    blue(davinci_2)
    text.append("\n")

    # Line 3 - UCAR + MONET
    orange(wave3_l)
    text.append("       ")
    aqua("U C A R")
    text.append("        ")
    orange(wave3_r)
    text.append("  ")
    aqua(monet)
    text.append("\n")

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
