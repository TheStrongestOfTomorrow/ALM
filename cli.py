import torch
from model import ALM, get_tiny_config
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown
import time

console = Console()

def main():
    console.clear()
    console.print(Panel(Text("ALM-1 Adaptive Language Model (135B Architecture)", justify="center", style="bold blue")))

    with console.status("[bold green]Initializing ALM-1 weights..."):
        config = get_tiny_config()
        model = ALM(config)
        time.sleep(1) # Simulate loading

    console.print("[yellow]System: Ready. (Running in Tiny-Local mode for demo)[/yellow]")
    console.print("[dim]Type '/exit' to quit, '/clear' to clear screen.[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]User > [/bold cyan]")

            if user_input.lower() == '/exit':
                break
            if user_input.lower() == '/clear':
                console.clear()
                continue
            if not user_input.strip():
                continue

            # Simulate thinking
            with console.status("[italic blue]ALM-1 is processing adaptive layers..."):
                time.sleep(0.8)

            # Dummy response for TUI demonstration
            response_text = "ALM-1 135B has analyzed your input. As an adaptive model specializing in English and Code, I recommend..."
            if "python" in user_input.lower():
                response_text = "```python\n# ALM-1 Generated Code\nprint('Hello World from ALM-1 135B')\n```"

            console.print(Panel(Markdown(response_text), title="ALM-1 Response", border_style="blue"))
            console.print("")

        except KeyboardInterrupt:
            break

    console.print("\n[bold red]ALM-1 Session Terminated.[/bold red]")

if __name__ == "__main__":
    main()
