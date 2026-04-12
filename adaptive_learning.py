import torch
import torch.optim as optim

class AdaptiveTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)

    def learn_step(self, text, enc):
        """
        Performs an online learning step (adaptive learning) from user interaction.
        This is a real ML concept where the model updates its weights based on new data.
        """
        self.model.train()
        tokens = enc.encode(text)
        if len(tokens) < 2:
            return 0.0

        x = torch.tensor(tokens[:-1]).unsqueeze(0)
        y = torch.tensor(tokens[1:]).unsqueeze(0)

        logits = self.model(x)
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def demonstrate_adaptive_learning(model, enc):
    trainer = AdaptiveTrainer(model)
    text = "ALM-1 is a powerful adaptive language model."
    loss = trainer.learn_step(text, enc)
    print(f"Adaptive Learning Step Loss: {loss:.4f}")

if __name__ == "__main__":
    from model import ALM, get_tiny_config
    import tiktoken
    config = get_tiny_config()
    model = ALM(config)
    enc = tiktoken.get_encoding("gpt2")
    demonstrate_adaptive_learning(model, enc)
