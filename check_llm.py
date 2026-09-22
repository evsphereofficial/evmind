import torch
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M", torch_dtype=torch.float32)
print("Total params:", sum(p.numel() for p in model.parameters()))
for i, layer in enumerate(model.transformer.h):
    w1 = list(layer.mlp.c_fc.weight.shape)
    w2 = list(layer.mlp.c_proj.weight.shape)
    print(f"Layer {i}: c_fc={w1} c_proj={w2} neuron_params={w1[0]*(w1[1]+w2[0])}")
    print(f"  intermediate={w1[0]} hidden={w1[1]}")
print("hidden_size:", model.config.hidden_size)
print("num_layers:", model.config.num_layers)
print("n_head:", model.config.num_heads)
total_neurons = sum(layer.mlp.c_fc.weight.shape[0] for layer in model.transformer.h)
print(f"Total FFN neurons: {total_neurons}")
print(f"Params per neuron (c_fc row + c_proj col): {model.config.hidden_size * 2}")
print(f"Total FFN params: {total_neurons * model.config.hidden_size * 2}")
