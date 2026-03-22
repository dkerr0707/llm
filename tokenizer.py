import tiktoken
import sys

tokenizer = tiktoken.get_encoding("gpt2")
print(len(tokenizer._mergeable_ranks))

text = sys.argv[1]
print(text)

tokens = tokenizer.encode(text)
print(tokens)

for i in tokens:
    print(tokenizer.decode([i]))

words = tokenizer.decode(tokens)
print(words)

