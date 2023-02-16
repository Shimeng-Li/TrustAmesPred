from collections.abc import Mapping, Iterable, Sequence
import torch.nn as nn

def narrow_row(x, start, stop):
    """Narrow down the tensor along the first dimension.
    Parameters
    ----------
    x : Tensor
        The input tensor.
    start : int
        The start index (inclusive).
    stop : int
        The stop index (exclusive).
    Returns
    -------
    Tensor
        The narrowed tensor
    Notes
    -----
    The returned tensor could be a view of the original tensor.
    """
    pass

def expand_as_pair(input_, g=None):
    """Return a pair of same element if the input is not a pair.
    If the graph is a block, obtain the feature of destination nodes from the source nodes.
    Parameters
    ----------
    input_ : Tensor, dict[str, Tensor], or their pairs
        The input features
    g : DGLHeteroGraph or DGLGraph or None
        The graph.
        If None, skip checking if the graph is a block.
    Returns
    -------
    tuple[Tensor, Tensor] or tuple[dict[str, Tensor], dict[str, Tensor]]
        The features for input and output nodes
    """
    if isinstance(input_, tuple):
        return input_
    elif g is not None and g.is_block:
        if isinstance(input_, Mapping):
            input_dst = {k: narrow_row(v, 0, g.number_of_dst_nodes(k)) for k, v in input_.items()}
        else:
            input_dst = narrow_row(input_, 0, g.number_of_dst_nodes())
        return input_, input_dst
    else:
        return input_, input_

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x