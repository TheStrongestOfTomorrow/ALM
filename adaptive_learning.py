"""
ALM Adaptive Learning Module
Implements online learning where the model adapts from user interactions.
"""

import torch
import torch.optim as optim


class AdaptiveTrainer:
    """
    Adaptive Learning enables ALM to continuously improve from interactions.
    
    Each user conversation provides new signal that fine-tunes the model's
    weights, making it progressively better at understanding and responding.
    This is a form of online learning - the model updates in real-time.
    """
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=lr)
        self.step_count = 0

    def learn_step(self, text, enc):
        """
        Performs an online learning step from user interaction.
        
        This is real ML: the model's weights are updated based on the
        new text data, allowing it to adapt its knowledge over time.
        """
        self.model.train()
        try:
            tokens = enc.encode(text)
            if len(tokens) < 4:
                return 0.0

            # Use the last block_size tokens
            max_len = min(len(tokens), self.model.config.block_size)
            tokens = tokens[-max_len:]

            x = torch.tensor(tokens[:-1]).unsqueeze(0)
            y = torch.tensor(tokens[1:]).unsqueeze(0)

            _, loss = self.model(x, targets=y)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            self.step_count += 1
            self.model.eval()
            return loss.item()
        except Exception as e:
            self.model.eval()
            return 0.0

    def get_stats(self):
        """Return adaptive learning statistics."""
        return {
            "total_steps": self.step_count,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }


def demonstrate_adaptive_learning(model, enc):
    """Demonstrate the adaptive learning capability."""
    trainer = AdaptiveTrainer(model)
    text = "ALM-1 is a powerful adaptive language model with Mixture of Experts."
    loss = trainer.learn_step(text, enc)
    print(f"Adaptive Learning Step - Loss: {loss:.4f}")
    print(f"Stats: {trainer.get_stats()}")

if __name__ == "__main__":
    from model import ALM, get_alm_tiny_config
    import tiktoken

    config = get_alm_tiny_config()
    model = ALM(config)
    enc = tiktoken.get_encoding("gpt2")
    demonstrate_adaptive_learning(model, enc)
