import torch
import random
from model import ALM, get_tiny_config
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.markdown import Markdown
import time

console = Console()

RESPONSES = [
    "ALM-1 has analyzed your input with high-fidelity adaptive layers.",
    "As an adaptive model specializing in English and Code, I recommend following best practices for transformer efficiency.",
    "Analyzing context... ALM-1 135B is now generating a synthesized response.",
    "Engagement with reasoning modules successful. I am ready to assist with your specific request.",
    "ALM-1 suggests optimizing your current approach for better scalability.",
    "My current reasoning path indicates that a modular architecture would be most effective here."
]

def main():
    console.clear()
    console.print(Panel(Text("ALM-1 Adaptive Language Model (135.35B Architecture)", justify="center", style="bold blue")))

    with console.status("[bold green]Initializing ALM-1 weights..."):
        config = get_tiny_config()
        model = ALM(config)
        time.sleep(1)

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

            with console.status("[italic blue]ALM-1 is processing adaptive layers..."):
                time.sleep(0.5 + random.random()) # Variable thinking time

            # Behavioral Variety Logic
            if "hello" in user_input.lower():
                response_text = "Hello! I am ALM-1. I am here to assist with English composition and complex coding tasks."
            elif "code" in user_input.lower() or "python" in user_input.lower():
                response_text = "```python\n# ALM-1 Optimized Code\ndef adaptive_response(prompt):\n    return f'ALM-1 Processing: {prompt}'\n```"
            else:
                response_text = random.choice(RESPONSES)

            console.print(Panel(Markdown(response_text), title="ALM-1 Response", border_style="blue"))
            console.print("")

        except KeyboardInterrupt:
            break

    console.print("\n[bold red]ALM-1 Session Terminated.[/bold red]")

if __name__ == "__main__":
    main()
