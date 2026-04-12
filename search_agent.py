import torch

class SimpleVectorStore:
    def __init__(self, emb_dim=128):
        self.documents = []
        self.embeddings = []
        self.emb_dim = emb_dim

    def add_document(self, text):
        self.documents.append(text)
        # In a real system, we'd use a real embedding model
        # Here we simulate with a hash-based deterministic vector
        vec = torch.randn(1, self.emb_dim)
        self.embeddings.append(vec)

    def search(self, query, top_k=1):
        if not self.embeddings:
            return []
        # Simulate cosine similarity
        query_vec = torch.randn(1, self.emb_dim)
        similarities = [torch.cosine_similarity(query_vec, doc_vec).item() for doc_vec in self.embeddings]
        sorted_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
        return [self.documents[i] for i in sorted_indices[:top_k]]

class SearchAgent:
    def __init__(self):
        self.store = SimpleVectorStore()
        # Seed with some knowledge
        self.store.add_document("ALM-1 uses a 135.35B parameter architecture.")
        self.store.add_document("The model is optimized for English and Code.")
        self.store.add_document("Adaptive reasoning allows dynamic compute scaling.")

    def query(self, text):
        results = self.store.search(text)
        context = " ".join(results)
        return f"Context found: {context}"

if __name__ == "__main__":
    agent = SearchAgent()
    print(agent.query("How many parameters?"))
