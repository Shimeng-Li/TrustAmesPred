import torch
import torch.nn as nn
import dgl
import dgl.backend as F

def segment_reduce(seglen, value, reducer='sum'):
    """
    Segment reduction operator.
    It aggregates the value tensor along the first dimension by segments.
    The first argument ``seglen`` stores the length of each segment. Its
    summation must be equal to the first dimension of the 'value' tensor.
    Zero-length segments are allowed.
    ----------
    seglen : Segment lengths.
    value : Value to aggregate.
    reducer : Aggregation method. Can be 'sum', 'max', 'min', 'mean'.
    """
    offsets = F.cumsum(F.cat([F.zeros((1,), F.dtype(seglen), F.context(seglen)), seglen], 0), 0)
    if reducer == 'mean':
        rst = F.segment_reduce('sum', value, offsets)
        rst_shape = F.shape(rst)
        z = F.astype(F.clamp(seglen, 1, len(value)), F.dtype(rst))
        z_shape = (rst_shape[0],) + (1,) * (len(rst_shape) - 1)
        return rst / F.reshape(z, z_shape)
    elif reducer in ['min', 'sum', 'max']:
        rst = F.segment_reduce(reducer, value, offsets)
        if reducer in ['min', 'max']:
            rst = F.replace_inf_with_zero(rst)
        return rst

def segment_softmax(seglen, value):
    """
    Performa softmax on each segment.
    The first argument 'seglen' stores the length of each segment.
    Its summation must be equal to the first dimension of the 'value' tensor.
    Zero-length segments are allowed.
    ----------
    seglen : Segment lengths.
    value : Value to aggregate.
    """
    value_max = segment_reduce(seglen, value, reducer='max')
    value = F.exp(value - F.repeat(value_max, seglen, dim=0))
    value_sum = segment_reduce(seglen, value, reducer='sum')
    return value / F.repeat(value_sum, seglen, dim=0)

def segment_mm(a, b, seglen_a):
    """
    Performs matrix multiplication according to segments.
    Suppose 'seglen_a == [10, 5, 0, 3]', the operator will perform four matrix multiplications:
        a[0:10] @ b[0], a[10:15] @ b[1],
        a[15:15] @ b[2], a[15:18] @ b[3]
    ----------
    a : The left operand, 2-D tensor of shape (N, D1)
    b : The right operand, 3-D tensor of shape (R, D1, D2)
    seglen_a : An integer tensor of shape (R,). Each element is the length of segments of input 'a'.
               The summation of all elements must be equal to 'N'.
    -------
    The output dense matrix of shape ``(N, D2)``
    """
    return F.segment_mm(a, b, seglen_a)

def readout_nodes(graph, feat, weight=None, *, op='sum', ntype=None):
    """
    Generate a graph-level representation by aggregating node features
    The function is commonly used as a *readout* function on a batch of graphs to generate graph-level representation.
    Thus, the result tensor shape depends on the batch size of the input graph.
    Given a graph of batch size 'B', and a feature size of 'D', the result shape will be '(B, D)',
    with each row being the aggregated node features of each graph.
    ----------
    graph : Input graph.
    feat : Node feature name.
    weight : Node weight name.
             None means aggregating without weights.
             Otherwise, multiply each node feature by node feature :attr:`weight` before aggregation.
             The weight feature shape must be compatible with an element-wise multiplication with the feature tensor.
    op : Readout operator. Can be 'sum', 'max', 'min', 'mean'.
    ntype : Node type. Can be omitted if there is only one node type in the graph.
    -------
    Result tensor.
    """
    x = graph.nodes[ntype].data[feat]
    if weight is not None:
        x = x * graph.nodes[ntype].data[weight]
    return segment_reduce(graph.batch_num_nodes(ntype), x, reducer=op)

def readout_edges(graph, feat, weight=None, *, op='sum', etype=None):
    """
    Sum the edge feature :attr:`feat` in :attr:`graph`, optionally multiplies it by a edge :attr:`weight`.
    The function is commonly used as a *readout* function on a batch of graphs to generate graph-level representation.
    Thus, the result tensor shape depends on the batch size of the input graph.
    Given a graph of batch size 'B', and a feature size of 'D', the result shape will be '(B, D)',
    with each row being the aggregated edge features of each graph.
    ----------
    graph : The input graph.
    feat : The edge feature name.
    weight : The edge weight feature name.
             If None, no weighting will be performed,
             otherwise, weight each edge feature with field 'feat' for summation.
             The weight feature shape must be compatible with an element-wise multiplication with the feature tensor.
    op : Readout operator. Can be 'sum', 'max', 'min', 'mean'.
    etype : The type names of the edges. The allowed type name formats are:
            * (str, str, str) for source node type, edge type and destination node type.
            * or one 'str' edge type name if the name can uniquely identify a triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
    -------
    Result tensor.
    """
    x = graph.edges[etype].data[feat]
    if weight is not None:
        x = x * graph.edges[etype].data[weight]
    return segment_reduce(graph.batch_num_edges(etype), x, reducer=op)

def sum_nodes(graph, feat, weight=None, *, ntype=None):
    """
    Syntax sugar for 'dgl.readout_nodes(graph, feat, weight, ntype=ntype, op='sum')'.
    """
    return readout_nodes(graph, feat, weight, ntype=ntype, op='sum')

def sum_edges(graph, feat, weight=None, *, etype=None):
    """
    Syntax sugar for 'dgl.readout_edges(graph, feat, weight, etype=etype, op='sum')'.
    """
    return readout_edges(graph, feat, weight, etype=etype, op='sum')

def mean_nodes(graph, feat, weight=None, *, ntype=None):
    """
    Syntax sugar for 'dgl.readout_nodes(graph, feat, weight, ntype=ntype, op='mean')'.
    """
    return readout_nodes(graph, feat, weight, ntype=ntype, op='mean')

def mean_edges(graph, feat, weight=None, *, etype=None):
    """
    Syntax sugar for 'dgl.readout_edges(graph, feat, weight, etype=etype, op='mean')'.
    """
    return readout_edges(graph, feat, weight, etype=etype, op='mean')

def max_nodes(graph, feat, weight=None, *, ntype=None):
    """
    Syntax sugar for 'dgl.readout_nodes(graph, feat, weight, ntype=ntype, op='max')'.
    """
    return readout_nodes(graph, feat, weight, ntype=ntype, op='max')

def max_edges(graph, feat, weight=None, *, etype=None):
    """
    Syntax sugar for 'dgl.readout_edges(graph, feat, weight, etype=etype, op='max')'.
    """
    return readout_edges(graph, feat, weight, etype=etype, op='max')

def softmax_nodes(graph, feat, *, ntype=None):
    """
    Perform graph-wise softmax on the node features.
    For each node: v\in\mathcal{V} and its feature: x_v,
    calculate its normalized feature as follows: z_v = exp(x_v)/sum_{u\in\mathcal{V}}\exp(x_u)}
    If the graph is a batch of multiple graphs, each graph computes softmax independently.
    The result tensor has the same shape as the original node feature.
    ----------
    graph : The input graph.
    feat : The node feature name.
    ntype : The node type name. Can be omitted if there is only one node type in the graph.
    """
    x = graph.nodes[ntype].data[feat]
    return segment_softmax(graph.batch_num_nodes(ntype), x)

def softmax_edges(graph, feat, *, etype=None):
    """
    Perform graph-wise softmax on the edge features.
    For each edge: e\in\mathcal{E} and its feature: ’x_e‘, calculate its normalized feature as follows:
    z_e = exp(x_e)/{\sum_{e\in\mathcal{E}}\exp(x_{e})}
    If the graph is a batch of multiple graphs, each graph computes softmax independently.
    The result tensor has the same shape as the original edge feature.
    ----------
    graph : The input graph.
    feat : The edge feature name.
    etype : The type names of the edges. The allowed type name formats are:
            * (str, str, str) for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a triplet format in the graph.
            Can be omitted if the graph has only one type of edges.
    """
    x = graph.edges[etype].data[feat]
    return segment_softmax(graph.batch_num_edges(etype), x)

def broadcast_nodes(graph, graph_feat, *, ntype=None):
    """
    Generate a node feature equal to the graph-level feature 'graph_feat'.
    The operation is similar to 'numpy.repeat' (or 'torch.repeat_interleave').
    It is commonly used to normalize node features by a global vector.
    ----------
    graph : The graph.
    graph_feat : The feature to broadcast. Tensor shape is (*) for single graph, and (B, *) for batched graph.
    ntype : Node type. Can be omitted if there is only one node type.
    """
    if len(F.shape(graph_feat)) == 1:
        graph_feat = F.unsqueeze(graph_feat, dim=0)
    return F.repeat(graph_feat, graph.batch_num_nodes(ntype), dim=0)

def broadcast_edges(graph, graph_feat, *, etype=None):
    """
    Generate an edge feature equal to the graph-level feature 'graph_feat'.
    The operation is similar to 'numpy.repeat' (or 'torch.repeat_interleave').
    It is commonly used to normalize edge features by a global vector.
    ----------
    graph : The graph.
    graph_feat : The feature to broadcast. Tensor shape is (*) for single graph, and (B, *) for batched graph.
    etype : Edge type. Can be omitted if there is only one edge type in the graph.
    -------
    The edge features tensor with shape (M, *), where 'M' is the number of edges.
    """
    if len(F.shape(graph_feat)) == 1:
        graph_feat = F.unsqueeze(graph_feat, dim=0)
    return F.repeat(graph_feat, graph.batch_num_edges(etype), dim=0)


class WeightAndSum(nn.Module):
    """
    Compute importance weights for atoms and perform a weighted sum.
    """
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(nn.Linear(in_feats, 1),
                                            nn.Sigmoid())

    def forward(self, g, feats):
        """
        Compute molecule representations out of atom representations
        """
        with g.local_scope():
            g.ndata['h'] = feats
            g.ndata['w'] = self.atom_weighting(g.ndata['h'])
            h_g_sum = sum_nodes(g, 'h', 'w')

        return h_g_sum

class WeightedSumAndMax(nn.Module):
    """
    Apply weighted sum and max pooling to the node representations and concatenate the results.
    """
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()

        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):
        """
        Readout
        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization
        Returns
        -------
        h_g : FloatTensor of shape (B, 2 * M1)
            * B is the number of graphs in the batch
            * M1 is the input node feature size, which must match
              in_feats in initialization
        """
        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g


