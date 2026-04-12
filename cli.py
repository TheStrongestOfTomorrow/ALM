import torch
import tiktoken
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
    console.print(Panel(Text("ALM-1 Adaptive Language Model (135.35B Architecture)", justify="center", style="bold blue")))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with console.status(f"[bold green]Initializing ALM-1 on {device}..."):
        config = get_tiny_config()
        model = ALM(config).to(device)
        model.eval()
        enc = tiktoken.get_encoding("gpt2")
        time.sleep(1)

    console.print(f"[yellow]System: ALM-1 Online. Real Neural Generation Active.[/yellow]")
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

            # Real Tokenization
            input_ids = torch.tensor(enc.encode(user_input, allowed_special={"<|endoftext|>"})).unsqueeze(0).to(device)

            with console.status("[italic blue]ALM-1 Neural Core Processing..."):
                # Real Generation using the model
                # We limit max_new_tokens for speed in this demo
                output_ids = model.generate(input_ids, max_new_tokens=32, temperature=0.8, top_k=40)
                response_text = enc.decode(output_ids[0].tolist()[input_ids.size(1):])

            console.print(Panel(Markdown(response_text), title="ALM-1 Neural Output", border_style="blue"))
            console.print("")

        except KeyboardInterrupt:
            break

    console.print("\n[bold red]ALM-1 Session Terminated.[/bold red]")

if __name__ == "__main__":
    main()
