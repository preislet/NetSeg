import torch

def channel_precision(y_true, y_pred, channel):
    y_pred_label = y_pred.argmax(dim=1) == channel
    y_true_label = y_true[:, channel, :, :] == 1

    tp = torch.sum((y_pred_label & y_true_label).float())
    pred_pos = torch.sum(y_pred_label.float())

    return tp / (pred_pos + 1e-8)


def channel_recall(y_true, y_pred, channel):
    y_pred_label = y_pred.argmax(dim=1) == channel
    y_true_label = y_true[:, channel, :, :] == 1

    tp = torch.sum((y_pred_label & y_true_label).float())
    true_pos = torch.sum(y_true_label.float())

    return tp / (true_pos + 1e-8)

def categorical_accuracy(y_true, y_pred):
    """
    Accuracy over all pixels (multiclass)
    y_true: (N, C, H, W) one-hot
    y_pred: (N, C, H, W) logits or probabilities
    """
    pred_class = y_pred.argmax(dim=1)
    true_class = y_true.argmax(dim=1)
    correct = (pred_class == true_class).float()
    return correct.mean()
