import torch

# Define the Softmax function from scratch
def softmax(logits):
    exp_logits = torch.exp(logits - torch.max(logits))
    sum_exp_logits = exp_logits.sum(dim=0, keepdim=True)
    return exp_logits / sum_exp_logits


# Define the cross-entropy loss function from scratch
def cross_entropy_loss(y_pred, y_true):
    y_pred_clipped = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -torch.sum(y_true * torch.log(y_pred_clipped))

logits = torch.tensor([2.0, 5.0, 3.0])
y_pred = torch.tensor([0, 1, 0])

print("logits: ", logits)
print("softmax(logits): ", softmax(logits))
print("cross_entropy_loss(y_pred, y_true):", cross_entropy_loss(y_pred, logits))
