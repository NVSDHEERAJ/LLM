import torch
import torch.nn.functional as F


def perplexity(logits, labels):
    perplexity = torch.exp(F.cross_entropy(logits, labels).sum() / labels.shape[1])

    return {"perplexity": perplexity}


def accuracy(predictions, labels):
    return {"accuracy": (predictions == labels).sum() / len(labels)}
