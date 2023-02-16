import torch
import torch.nn as nn
import torch.nn.functional as F
from gat_model_utils import expand_as_pair, Identity
from readout_utils import WeightedSumAndMax
import dgl.function as func
import dgl.ops as ops

class gat_conv(nn.Module):
    def __init__(self,
                 in_feats,  # Input feature size, i.e, the number of dimensions of 'h_i^{(l)}'
                 out_feats, # Output feature size, i.e, the number of dimensions of 'h_i^{(l+1)}'
                 num_heads, # Number of heads in multi-head attention
                 feat_drop = 0, # Dropout rate on feature.
                 attn_drop = 0, # Dropout rate on attention weight
                 negative_slope = 0.2,  # LeakyReLU angle of negative slope
                 residual = False,  # If True, use residual connection
                 activation = None, # If not None, applies an activation function to the update node features
                 bias = True):  # If True, learns a bias term.
        super(gat_conv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self._in_src_feats = nn.Linear(self._in_src_feats, out_feats * num_heads, bias = False)
            self.fc_dst = nn.Linear(self._in_dst_feats, out_feats * num_heads, bias = False)
        else:
            self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias = False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size = (1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size = (1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(size = (num_heads * out_feats, )))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(self._in_dst_feats, num_heads * out_feats, bias = False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        The fc weights 'W^{(l)}' are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain = gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain = gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain = gain)
        nn.init.xavier_normal_(self.attn_l, gain = gain)
        nn.init.xavier_normal_(self.attn_r, gain = gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain = gain)

    def forward(self,   # Compute graph attention network layer
                graph,
                feat,   # If a torch.Tensor is given, the input feature is the number of nodes;
                get_attention = False): # Whether to return the attention values.
        with graph.local_scope():
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(*dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(*src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]
            # Note: The GAT paper uses "first concatenation then linear projection" to compute attention scores, while
            #       the implementation is "first projection then addition", the two approaches are mathematically
            #       equivalent:
            #       The weight vector mentioned in the paper was decomposed into [a_l || a_r], then
            #       a^T [Wh_i | | Wh_j] = a_l Wh_i + a_r Wh_j
            el = (feat_src * self.attn_l).sum(dim = -1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim = -1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # Compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(func.u_add_v('el', 'er', 'e'))
            e = self.leakyrelu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(ops.edge_softmax(graph, e))
            # message passing
            graph.update_all(func.u_mul_e('ft', 'a', 'm'),
                             func.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(*dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(*((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class gat_layer(nn.Module):
    def __init__(self,
                 in_feats,  # Number of input node features
                 out_feats, # Number of output node features
                 num_heads, # Number of attention heads
                 feat_drop, # Dropout applied to the input features
                 attn_drop, # Dropout applied to attention values of edges
                 alpha=0.2, # LeakyReLU angle of negative slope
                 residual=True, # Whether to perform skip connection, default to True.
                 agg_mode='flatten',    # The way to aggregate multi-head attention results.
                 activation=None,   # Activation function applied to the aggregated multi-head results, default to None.
                 bias=True):    # Whether to use bias in the GAT layer.
        super(gat_layer, self).__init__()

        self.gat_conv = gat_conv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop,
                                negative_slope=alpha, residual=residual, bias=bias)
        assert agg_mode in ['flatten', 'mean']
        # 'flatten' for concatenating all-head results
        # 'mean' for averaging all head results.
        self.agg_mode = agg_mode
        self.activation = activation

    def reset_parameters(self):
        self.gat_conv.reset_parameters()    # Reinitialize model parameters.

    def forward(self,   # Update node representations.
                bg, # DGLGraph for a batch of graphs.
                feats): # FloatTensor of shape (N, M1)
                        # N is the total number of nodes in the batch of graphs
                        # M1 is the input node feature size, which equals in_feats in initialization
        feats = self.gat_conv(bg, feats)
        if self.agg_mode == 'flatten':
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)
            # FloatTensor of shape (N, M2)
            # N is the total number of nodes in the batch of graphs
            # M2 is the output node representation size.
            # If self.agg_mode == 'mean', M2 equals out_feats in initialization
            # If initialization == 'flatten', M2 =  out_feats * num_heads

        return feats

class gat_model(nn.Module):
    """
    in_feats : int
               Number of input node features
    hidden_feats : list of int
                   hidden_feats[i] gives the output size of an attention head in the i-th GAT layer.
                   len(hidden_feats) equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
                num_heads[i] gives the number of attention heads in the i-th GAT layer.
                len(num_heads) equals the number of GAT layers. By default, we use 4 attention heads for each GAT layer.
    feat_drops : list of float
                 feat_drops[i] gives the dropout applied to the input features in the i-th GAT layer.
                 len(feat_drops) equals the number of GAT layers. By default, this will be zero for all GAT layers.
    attn_drops : list of float
                 attn_drops[i] gives the dropout applied to attention values of edges in the i-th GAT layer.
                 len(attn_drops) equals the number of GAT layers. By default, this will be zero for all GAT layers.
    alphas : list of float
             Hyperparameters in LeakyReLU, which are the slopes for negative values.
             alphas[i] gives the slope for negative value in the i-th GAT layer.
             len(alphas) equals the number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
                residual[i] decides if residual connection is to be used for the i-th GAT layer.
                len(residual) equals the number of GAT layers.
                By default, residual connection is performed for each GAT layer.
    agg_modes : list of str
                The way to aggregate multi-head attention results for each GAT layer.
                'flatten' for concatenating all-head results
                'mean' for averaging all-head results.
                agg_modes[i] gives the way to aggregate multi-head attention results for the i-th GAT layer.
                len(agg_modes) equals the number of GAT layers.
                By default, we flatten all-head results for each GAT layer.
    activations : list of activation function or None
                  activations[i] gives the activation function applied to the aggregated multi-head results for the i-th GAT layer.
                  len(activations) equals the number of GAT layers.
                  By default, no activation is applied for each GAT layer.
    biases : list of bool
             biases[i] gives whether to use bias for the i-th GAT layer.
             len(activations) equals the number of GAT layers.
             By default, we use bias for all GAT layers.
    """
    def __init__(self,
                 in_feats,
                 hidden_feats=None,
                 num_heads=None,
                 feat_drops=None,
                 attn_drops=None,
                 alphas=None,
                 residuals=None,
                 agg_modes=None,
                 activations=None,
                 biases=None):
        super(gat_model, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0. for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0. for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ['flatten' for _ in range(n_layers - 1)]
            agg_modes.append('mean')
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        if biases is None:
            biases = [True for _ in range(n_layers)]
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(gat_layer(in_feats,
                                             hidden_feats[i],
                                             num_heads[i],
                                             feat_drops[i],
                                             attn_drops[i],
                                             alphas[i],
                                             residuals[i],
                                             agg_modes[i],
                                             activations[i],
                                             biases[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def reset_parameters(self):
        """Reinitialize model parameters."""
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, g, feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which equals in_feats in initialization
        Returns
        -------
        feats : FloatTensor of shape (N, M2)
            * N is the total number of nodes in the batch of graphs
            * M2 is the output node representation size, which equals
              hidden_sizes[-1] if agg_modes[-1] == 'mean' and
              hidden_sizes[-1] * num_heads[-1] otherwise.
        """
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)
        return feats

class gat_classifier(nn.Module):
    """
    After updating node representations,
    we perform a weighted sum with learnable weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.
    For classification tasks, the output will be logits, i.e. values before sigmoid or softmax.
    ----------
    in_feats : Number of input node features
    hidden_feats : hidden_feats[i] gives the output size of an attention head in the i-th GAT layer.
                   len(hidden_feats)`` equals the number of GAT layers.
                   default = [32, 32].
      num_heads : list of int
          ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
          ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
          for each GAT layer.
      feat_drops : list of float
          ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
          ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
          all GAT layers.
      attn_drops : list of float
          ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
          layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
          for all GAT layers.
      alphas : list of float
          Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
          gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
          number of GAT layers. By default, this will be 0.2 for all GAT layers.
      residuals : list of bool
          ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
          ``len(residual)`` equals the number of GAT layers. By default, residual connection
          is performed for each GAT layer.
      agg_modes : list of str
          The way to aggregate multi-head attention results for each GAT layer, which can be either
          'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
          ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
          GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
          multi-head results for intermediate GAT layers and compute mean of multi-head results
          for the last GAT layer.
      activations : list of activation function or None
          ``activations[i]`` gives the activation function applied to the aggregated multi-head
          results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
          By default, ELU is applied for intermediate GAT layers and no activation is applied
          for the last GAT layer.
      biases : list of bool
          ``biases[i]`` gives whether to add bias for the i-th GAT layer. ``len(activations)``
          equals the number of GAT layers. By default, bias is added for all GAT layers.
      classifier_hidden_feats : int
          (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
          in the classifier. Default to 128.
      classifier_dropout : float
          (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
          Default to 0.
      n_tasks : int
          Number of tasks, which is also the output size. Default to 1.
      predictor_hidden_feats : int
          Size for hidden representations in the output MLP predictor. Default to 128.
      predictor_dropout : float
          The probability for dropout in the output MLP predictor. Default to 0.
      """

    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 biases=None, n_tasks=1,
                 predictor_hidden_feats=128, predictor_dropout=0.):
        super(gat_classifier, self).__init__()

        self.gnn = gat_model(in_feats=in_feats,
                             hidden_feats=hidden_feats,
                             num_heads=num_heads,
                             feat_drops=feat_drops,
                             attn_drops=attn_drops,
                             alphas=alphas,
                             residuals=residuals,
                             agg_modes=agg_modes,
                             activations=activations,
                             biases=biases)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats, n_tasks, predictor_dropout)

    def forward(self, bg, feats):
        """Graph-level regression/soft classification.
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
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.predict(graph_feats)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data)

class MLPPredictor(nn.Module):
    """
    Two-layer MLP for regression or soft classification over multiple tasks from graph representations.
    For classification tasks, the output will be logits, i.e. values before sigmoid or softmax.
    Parameters
    ----------
    in_feats : int
        Number of input graph features
    hidden_feats : int
        Number of graph features in hidden layers
    n_tasks : int
        Number of tasks, which is also the output size.
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPPredictor, self).__init__()

        self.predict = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(in_feats, hidden_feats),
                                     nn.ReLU(),
                                     nn.BatchNorm1d(hidden_feats),
                                     nn.Linear(hidden_feats, n_tasks))

    def forward(self, feats):
        """Make prediction.
        Parameters
        ----------
        feats : FloatTensor of shape (B, M3)
            * B is the number of graphs in a batch
            * M3 is the input graph feature size, must match in_feats in initialization
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(feats)

class gat_feat(nn.Module):
    """
    After updating node representations,
    we perform a weighted sum with learnable weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.
    For classification tasks, the output will be logits, i.e. values before sigmoid or softmax.
    ----------
    in_feats : Number of input node features
    hidden_feats : hidden_feats[i] gives the output size of an attention head in the i-th GAT layer.
                   len(hidden_feats)`` equals the number of GAT layers.
                   default = [32, 32].
      num_heads : list of int
          ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
          ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
          for each GAT layer.
      feat_drops : list of float
          ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
          ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
          all GAT layers.
      attn_drops : list of float
          ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
          layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
          for all GAT layers.
      alphas : list of float
          Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
          gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
          number of GAT layers. By default, this will be 0.2 for all GAT layers.
      residuals : list of bool
          ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
          ``len(residual)`` equals the number of GAT layers. By default, residual connection
          is performed for each GAT layer.
      agg_modes : list of str
          The way to aggregate multi-head attention results for each GAT layer, which can be either
          'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
          ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
          GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
          multi-head results for intermediate GAT layers and compute mean of multi-head results
          for the last GAT layer.
      activations : list of activation function or None
          ``activations[i]`` gives the activation function applied to the aggregated multi-head
          results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
          By default, ELU is applied for intermediate GAT layers and no activation is applied
          for the last GAT layer.
      biases : list of bool
          ``biases[i]`` gives whether to add bias for the i-th GAT layer. ``len(activations)``
          equals the number of GAT layers. By default, bias is added for all GAT layers.
      classifier_hidden_feats : int
          (Deprecated, see ``predictor_hidden_feats``) Size of hidden graph representations
          in the classifier. Default to 128.
      classifier_dropout : float
          (Deprecated, see ``predictor_dropout``) The probability for dropout in the classifier.
          Default to 0.
      n_tasks : int
          Number of tasks, which is also the output size. Default to 1.
      predictor_hidden_feats : int
          Size for hidden representations in the output MLP predictor. Default to 128.
      predictor_dropout : float
          The probability for dropout in the output MLP predictor. Default to 0.
      """

    def __init__(self,
                 in_feats,
                 hidden_feats=None,
                 num_heads=None,
                 feat_drops=None,
                 attn_drops=None,
                 alphas=None,
                 residuals=None,
                 agg_modes=None,
                 activations=None,
                 biases=None):
        super(gat_feat, self).__init__()

        self.gnn = gat_model(in_feats=in_feats,
                             hidden_feats=hidden_feats,
                             num_heads=num_heads,
                             feat_drops=feat_drops,
                             attn_drops=attn_drops,
                             alphas=alphas,
                             residuals=residuals,
                             agg_modes=agg_modes,
                             activations=activations,
                             biases=biases)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)

    def forward(self, bg, feats):
        """Graph-level regression/soft classification.
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
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return graph_feats
