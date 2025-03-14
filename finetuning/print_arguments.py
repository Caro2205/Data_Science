import torch

folder = "ModernBERT-sentiment"

training_args = torch.load(f"{folder}/training_args.bin", weights_only=False)
print(training_args)