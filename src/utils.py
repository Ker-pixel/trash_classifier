import torch

def accuracy(pred, labels):
    return (pred == labels).sum().item() / len(labels)

def to_binary(labels, trash_idx):
    return (labels != trash_idx).long()