from contextlib import contextmanager
from timeit import default_timer
import torch
import torch.nn.functional as F

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def pad(tensor, bos_id, eos_id):
    '''Adds a padding symbol at the beginning and at the end of a tensor.

    Arguments:
    tensor -- The tensor to pad.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''
    tensor = F.pad(input=tensor, pad=(1,0), mode='constant', value=bos_id)
    tensor = F.pad(input=tensor, pad=(0,1), mode='constant', value=eos_id)
    return tensor

def collate(batch, bos_id, eos_id):
    '''Generates padded tensors for the dataloader.

    Arguments:
    batch -- The original data.
    bos_id -- The value of the padding for the beginning.
    eos_id -- The value of the padding for the end.'''

    a_emb, b_emb, c_emb, d_emb = [], [], [], []

    for a, b, c, d in batch:
        a_emb.append(pad(a, bos_id, eos_id))
        b_emb.append(pad(b, bos_id, eos_id))
        c_emb.append(pad(c, bos_id, eos_id))
        d_emb.append(pad(d, bos_id, eos_id))

    # make a tensor of all As, af all Bs, of all Cs and of all Ds
    a_emb = torch.stack(a_emb)
    b_emb = torch.stack(b_emb)
    c_emb = torch.stack(c_emb)
    d_emb = torch.stack(d_emb)

    return a_emb, b_emb, c_emb, d_emb

def get_accuracy_classification(y_true, y_pred):
    '''Computes the accuracy for a batch of data of the classification task.

    Arguments:
    y_true -- The tensor of expected values.
    y_pred -- The tensor of predicted values.'''
    assert y_true.size() == y_pred.size()
    y_pred = y_pred > 0.5
    if y_pred.ndim > 1:
        return (y_true == y_pred).sum().item() / y_true.size(0)
    else:
        return (y_true == y_pred).sum().item()
