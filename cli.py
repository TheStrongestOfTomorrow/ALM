import torch
from model import ALM, get_tiny_config

def main():
    print("--- ALM-1 Adaptive Language Model CLI ---")
    print("Loading tiny model for demonstration purposes...")

    config = get_tiny_config()
    model = ALM(config)

    # In a real scenario, we would load weights here.
    # For this demonstration, we'll use a random model but show the interface.

    print("Model loaded. (Note: Using uninitialized weights for demo)")
    print("Type 'quit' to exit.")

    while True:
        user_input = input("\nALM > ")
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input.strip():
            continue

        # Basic tokenization (character-based for simplicity in this demo)
        # In production, we'd use a proper tokenizer (e.g., Tiktoken)
        chars = sorted(list(set(user_input)))
        stoi = { ch:i for i,ch in enumerate(chars) }
        # Map input to dummy tokens within vocab_size
        input_ids = torch.tensor([[hash(c) % config.vocab_size for c in user_input]], dtype=torch.long)

        print("\nALM is thinking...")

        # Generate some dummy output
        generated = model.generate(input_ids, max_new_tokens=20)

        # In a real model, we'd decode the tokens.
        # Here we just show that the generation loop works.
        print("ALM Response: [Adaptive Response Generated]")
        print("(In a fully trained model, this would be the English/Code response to your prompt.)")

if __name__ == "__main__":
    main()
