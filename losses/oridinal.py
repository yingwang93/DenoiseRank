import torch
from torch.nn import BCELoss
import torch.distributed as dist



def with_ordinals(y, n, dev, padded_value_indicator=-1):
    """
    Helper function for ordinal loss, transforming input labels to ordinal values.
    :param y: labels, shape [batch_size, slate_length]
    :param n: number of ordinals
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: ordinals, shape [batch_size, slate_length, n]
    """
    one_to_n = torch.arange(start=1, end=n + 1, dtype=torch.float, device=dev)
    unsqueezed = y.unsqueeze(2).repeat(1, 1, n)
    mask = unsqueezed == padded_value_indicator
    ordinals = (unsqueezed >= one_to_n).type(torch.float)
    ordinals[mask] = padded_value_indicator
    return ordinals


def ordinal(y_pred, y_true, n=4, padded_value_indicator=-1):
    """
    Ordinal loss.
    :param y_pred: predictions from the model, shape [batch_size, slate_length, n]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param n: number of ordinal values, int
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    device = y_pred.device
    y_pred = y_pred.clone()
    y_true = with_ordinals(y_true.clone(), n, dev=device)

    mask = y_true == padded_value_indicator
    valid_mask = y_true != padded_value_indicator

    ls = BCELoss(reduction='none')(y_pred, y_true)
    ls[mask] = 0.0

    document_loss = torch.sum(ls, dim=2)
    sum_valid = torch.sum(valid_mask, dim=2).type(torch.float32) > torch.tensor(0.0, dtype=torch.float32, device=device)

    loss_output = torch.sum(document_loss) / torch.sum(sum_valid)

    return loss_output
