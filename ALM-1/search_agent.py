"""
ALM Search Agent
RAG (Retrieval-Augmented Generation) module for grounding responses in knowledge.
Uses TF-IDF style matching for lightweight, efficient search without external dependencies.
"""

import math
import re
from collections import Counter


class TFIDFStore:
    """
    Lightweight vector store using TF-IDF for document retrieval.
    No external embeddings needed - works with pure Python.
    """
    def __init__(self):
        self.documents = []
        self.doc_tfidf = []
        self.idf = {}
        self.vocab = {}

    def _tokenize(self, text):
        """Simple word tokenization."""
        return re.findall(r'\w+', text.lower())

    def _compute_tf(self, tokens):
        """Compute term frequency."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {word: count / total for word, count in counts.items()}

    def add_document(self, text):
        """Add a document to the store."""
        doc_id = len(self.documents)
        self.documents.append(text)
        tokens = self._tokenize(text)

        # Update vocabulary
        for token in set(tokens):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        # Compute TF
        tf = self._compute_tf(tokens)
        self.doc_tfidf.append(tf)

        # Update IDF
        for word in set(tokens):
            self.idf[word] = self.idf.get(word, 0) + 1

    def _cosine_similarity(self, vec1, vec2):
        """Compute cosine similarity between two sparse vectors."""
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0

        dot = sum(vec1[k] * vec2[k] for k in common_keys)
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def search(self, query, top_k=3):
        """Search for relevant documents using TF-IDF similarity."""
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        query_tf = self._compute_tf(query_tokens)

        # Apply IDF weighting
        n_docs = len(self.documents)
        query_tfidf = {}
        for word, tf in query_tf.items():
            idf_val = math.log(n_docs / (self.idf.get(word, 0) + 1)) + 1
            query_tfidf[word] = tf * idf_val

        # Compute similarities
        similarities = []
        for doc_id, doc_tf in enumerate(self.doc_tfidf):
            doc_tfidf_weighted = {}
            for word, tf in doc_tf.items():
                idf_val = math.log(n_docs / (self.idf.get(word, 0) + 1)) + 1
                doc_tfidf_weighted[word] = tf * idf_val

            sim = self._cosine_similarity(query_tfidf, doc_tfidf_weighted)
            similarities.append((doc_id, sim))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [self.documents[doc_id] for doc_id, _ in similarities[:top_k]]


class SearchAgent:
    """
    RAG-powered search agent for ALM.
    Retrieves relevant context from a knowledge base to ground responses.
    """
    def __init__(self):
        self.store = TFIDFStore()
        # Seed with ALM knowledge
        self.store.add_document(
            "ALM (Adaptive Learning Model) is a next-generation transformer architecture "
            "featuring Mixture of Experts for adaptive computation. It dynamically routes "
            "tokens through specialized expert networks based on task complexity."
        )
        self.store.add_document(
            "ALM uses a Mixture of Experts (MoE) architecture where a gating network "
            "selects the most relevant experts for each input. This provides both "
            "specialization and efficiency - only relevant experts are activated."
        )
        self.store.add_document(
            "The Adaptive Learning feature allows ALM to continuously improve from "
            "interactions. Each conversation updates the model weights through online "
            "learning, making it progressively better at understanding users."
        )
        self.store.add_document(
            "ALM-Tiny has 7.8 million parameters with 4 layers, 4 attention heads, "
            "128 embedding dimensions, and 2 experts per MoE layer. The full ALM-1 "
            "specification targets 135.35 billion parameters."
        )
        self.store.add_document(
            "Python is a high-level programming language known for readability. "
            "It supports object-oriented, functional, and procedural paradigms. "
            "Common uses include web development, data science, and AI."
        )
        self.store.add_document(
            "Machine learning is a branch of AI where computers learn from data. "
            "Main types: supervised learning (labeled data), unsupervised learning "
            "(hidden patterns), and reinforcement learning (trial and reward)."
        )
        self.store.add_document(
            "GitHub is a platform for version control and collaboration using Git. "
            "Features include repositories, pull requests, issues, and GitHub Actions "
            "for CI/CD automation."
        )
        self.store.add_document(
            "Docker is a containerization platform. Containers package applications "
            "with their dependencies for consistent deployment. They share the host "
            "kernel, making them lightweight compared to virtual machines."
        )

    def query(self, text):
        """Search for relevant context given a query."""
        results = self.store.search(text, top_k=2)
        if results:
            return "Context: " + " ".join(results)
        return ""

    def add_knowledge(self, text):
        """Add new knowledge to the search store."""
        self.store.add_document(text)


if __name__ == "__main__":
    agent = SearchAgent()
    print("Testing Search Agent:")
    queries = [
        "How many parameters does ALM have?",
        "What is machine learning?",
        "Tell me about Python",
        "How does Docker work?",
    ]
    for q in queries:
        context = agent.query(q)
        print(f"\nQuery: {q}")
        print(f"Context: {context[:150]}...")
