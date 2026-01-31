import ollama

single = ollama.embed(
  model='embeddinggemma',
  input='The quick brown fox jumps over the lazy dog.'
)
print(len(single['embeddings'][0]))  # vector length