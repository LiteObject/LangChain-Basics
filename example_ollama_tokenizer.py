from langchain_ollama import OllamaLLM, OllamaEmbeddings
from transformers import AutoTokenizer

# Initialize the Ollama model for text generation
llm = OllamaLLM(model="llama3.2:latest")  # Replace with a text generation model of your choice

# Initialize the Ollama embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Example text
text = "Hello, how are you?"

# Load the tokenizer for llama3.2
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Tokenize the text using the tokenizer
try:
    # Use the tokenizer from the LLM model
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    print(f"Number of tokens: {token_count}")
except Exception as e:
    print(f"Error during tokenization: {e}")

# Generate embeddings for the text
try:
    embedding = embeddings.embed_query(text)
    print(f"Embedding vector (first 5 elements): {embedding[:5]}")
    print(f"Embedding length: {len(embedding)}")
except Exception as e:
    print(f"Error during embedding generation: {e}")
