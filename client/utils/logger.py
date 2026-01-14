# log_utils.py
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

console = Console()

# Stage-to-style mapping
STAGE_STYLES = {
    "agent": {"emoji": "", "color": "bright_cyan"},
    "loop": {"emoji": "", "color": "yellow"},
    "perception": {"emoji": "", "color": "bright_green"},
    "memory": {"emoji": "", "color": "magenta"},
    "plan": {"emoji": "", "color": "blue"},
    "parser": {"emoji": "", "color": "cyan"},
    "tool": {"emoji": "", "color": "bright_yellow"},
    "server": {"emoji": "", "color": "bright_magenta"},
    "default": {"emoji": "", "color": "white"},
}


def log(stage: str, msg: str, level: str = "INFO"):
    now = datetime.now().strftime("%H:%M:%S")
    key = stage.lower()
    style = STAGE_STYLES.get(key, STAGE_STYLES["default"])

    panel = Panel.fit(
        f"[bold white]{msg}[/bold white]",
        title=f"{style['emoji']} [bold {style['color']}]{stage.upper()}[/bold {style['color']}]",
        subtitle=f"[dim]{now}[/dim]",
        border_style=style['color']
    )

    if level == "INFO":
        console.print(panel)
    elif level == "WARNING":
        console.print(panel, style="bold yellow")
    elif level == "ERROR":
        console.print(panel, style="bold red")
    else:
        console.print(panel)