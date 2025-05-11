from transformers import GPT2Tokenizer

def count_tokens(text, tokenizer_name='gpt2') -> int:
    # Load the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize the input text
    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Print all the tokens
    print("Tokens:", tokens)
    
    # Return the number of tokens
    return len(tokens)

# Example usage
text = "Hello, how are you?"
token_count = count_tokens(text)
print(f"Number of tokens: {token_count}")