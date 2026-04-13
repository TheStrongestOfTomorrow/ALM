"""
ALM Terminal Interface
Rich TUI for interacting with ALM in the terminal.
"""

import torch
import tiktoken
import time
import sys
import os

from model import ALM, get_alm_tiny_config

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.live import Live
    from rich.markdown import Markdown
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = tiktoken.get_encoding("gpt2")
    config = get_alm_tiny_config()

    # Load checkpoint if available
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints', 'alm_tiny_best.pt')
    if os.path.exists(checkpoint_path):
        model = ALM.load_checkpoint(checkpoint_path, device=device)
    else:
        model = ALM(config).to(device)

    model.eval()

    # Conversation history
    conversation = []
    max_history = 8  # Number of messages to keep for context

    if HAS_RICH:
        _rich_chat(model, enc, device, conversation, max_history)
    else:
        _plain_chat(model, enc, device, conversation, max_history)


def _build_prompt(conversation, enc, max_tokens=200):
    """Build a prompt from conversation history."""
    from model import ASSISTANT_TOKEN

    parts = []
    total_len = 0
    max_context = 256 - max_tokens  # Leave room for generation

    for msg in conversation[-max_history:]:
        role = msg['role']
        content = msg['content']
        if role == 'user':
            parts.append(content)
        elif role == 'assistant':
            parts.append(f"{ASSISTANT_TOKEN}{content}")
        total_len += len(enc.encode(parts[-1]))

    prompt = "\n".join(parts)
    if not prompt.endswith(ASSISTANT_TOKEN):
        prompt += f"\n{ASSISTANT_TOKEN}"

    # Truncate if too long
    tokens = enc.encode(prompt)
    if len(tokens) > max_context:
        tokens = tokens[-max_context:]
        prompt = enc.decode(tokens)

    return prompt


def _rich_chat(model, enc, device, conversation, max_history):
    """Rich TUI chat interface."""
    console = Console()
    console.clear()
    console.print(Panel(
        Text("ALM - Adaptive Learning Model\nMoE Transformer | Terminal Interface", justify="center", style="bold blue"),
        border_style="blue"
    ))

    console.print(f"[yellow]System: ALM Online on {device}[/yellow]")
    console.print("[dim]Type '/exit' to quit, '/clear' to clear, '/help' for commands[/dim]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]")

            if user_input.lower() == '/exit':
                break
            if user_input.lower() == '/clear':
                conversation.clear()
                console.clear()
                continue
            if user_input.lower() == '/help':
                console.print("[dim]/exit - Quit\n/clear - Clear conversation\n/help - Show help[/dim]")
                continue
            if not user_input.strip():
                continue

            conversation.append({"role": "user", "content": user_input})
            prompt = _build_prompt(conversation, enc)

            with console.status("[italic blue]ALM thinking..."):
                input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_k=40,
                    top_p=0.9,
                    repetition_penalty=1.3
                )
                response = enc.decode(output_ids[0].tolist()[input_ids.size(1):])
                response = response.replace("<|assistant|", "").replace("<|end|>", "").strip()

            conversation.append({"role": "assistant", "content": response})
            console.print(Panel(Markdown(response), title="ALM", border_style="blue"))
            console.print("")

        except KeyboardInterrupt:
            break

    console.print("\n[bold red]ALM session ended.[/bold red]")


def _plain_chat(model, enc, device, conversation, max_history):
    """Plain text chat interface (fallback without rich)."""
    print("\n" + "=" * 50)
    print("  ALM - Adaptive Learning Model")
    print("  MoE Transformer | Terminal Interface")
    print("=" * 50)
    print(f"System: ALM Online on {device}")
    print("Type '/exit' to quit, '/clear' to clear\n")

    while True:
        try:
            user_input = input("You > ")

            if user_input.lower() == '/exit':
                break
            if user_input.lower() == '/clear':
                conversation.clear()
                print("Conversation cleared.\n")
                continue
            if not user_input.strip():
                continue

            conversation.append({"role": "user", "content": user_input})
            prompt = _build_prompt(conversation, enc)

            print("ALM thinking...", end="", flush=True)
            input_ids = torch.tensor(enc.encode(prompt)).unsqueeze(0).to(device)
            output_ids = model.generate(
                input_ids,
                max_new_tokens=150,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.3
            )
            response = enc.decode(output_ids[0].tolist()[input_ids.size(1):])
            response = response.replace("<|assistant|", "").replace("<|end|>", "").strip()

            conversation.append({"role": "assistant", "content": response})
            print(f"\rALM: {response}\n")

        except KeyboardInterrupt:
            break

    print("\nALM session ended.")


if __name__ == "__main__":
    main()
