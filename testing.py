from ollama import embed

# Single query embedding
query_embedding = embed(
    model="nomic-embed-text",
    input="The quick brown fox jumps over the lazy dog."
)

print(query_embedding)

