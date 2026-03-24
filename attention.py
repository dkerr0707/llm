import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your     (x^1)
     [0.55, 0.87, 0.66], # journey  (x^2)
     [0.57, 0.85, 0.64], # starts   (x^3)
     [0.22, 0.58, 0.33], # with     (x^4)
     [0.77, 0.25, 0.10], # one      (x^5)
     [0.05, 0.80, 0.55]] # step     (x^6)
)
print("\nInputs\n", inputs)
print("\nInputs transpose\n", inputs.T)

attn_scores = inputs @ inputs.T
print("\nAttention scores\n", attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print("\nAttention weights\n", attn_weights)

print("\nSum\n", attn_weights.sum(dim=-1))

context_vectors = attn_weights @ inputs
print("\nContext Vectors\n", context_vectors)
