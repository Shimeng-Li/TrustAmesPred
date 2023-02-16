import dgl.backend as F
import numpy as np
import os
import pandas as pd
import torch
from dgl.data.utils import save_graphs, load_graphs, split_dataset, Subset
from rdkit.Chem import rdmolfiles, rdmolops
from utils import pmap
import dgl
from collections import defaultdict
from functools import partial
from itertools import accumulate, chain
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
import itertools

class MoleculeCSVDataset(object):
    """
    MoleculeCSVDataset
    This is a general class for loading molecular data from :class:`pandas.DataFrame`.
    In data pre-processing, we construct a binary mask indicating the existence of labels.
    All molecules are converted into DGLGraphs. After the first-time construction, the
    DGLGraphs can be saved for reloading so that we do not need to reconstruct them every time.
    """

    def __init__(self,
                 df,
                 smiles_to_graph,
                 node_featurizer,
                 edge_featurizer,
                 smiles_column,
                 cache_file_path,
                 task_names=None,
                 load=False,
                 log_every=1000,
                 init_mask=True,
                 n_jobs=1,
                 error_log=None):
        self.df = df
        self.smiles = self.df[smiles_column].tolist()
        if task_names is None:
            self.task_names = self.df.columns.drop([smiles_column]).tolist()
        else:
            self.task_names = task_names
        self.n_tasks = len(self.task_names)
        self.cache_file_path = cache_file_path
        self._pre_process(smiles_to_graph, node_featurizer, edge_featurizer,
                          load, log_every, init_mask, n_jobs, error_log)

        # Only useful for binary classification tasks
        self._task_pos_weights = None

    def _pre_process(self, smiles_to_graph, node_featurizer, edge_featurizer,
                     load, log_every, init_mask, n_jobs, error_log):
        """Pre-process the dataset
        * Convert molecules from smiles format into DGLGraphs and featurize their atoms
        * Set missing labels to be 0 and use a binary masking matrix to mask them
        """
        if os.path.exists(self.cache_file_path) and load:
            # DGLGraphs have been constructed before, reload them
            print('Loading previously saved dgl graphs...')
            self.graphs, label_dict = load_graphs(self.cache_file_path)
            self.labels = label_dict['labels']
            if init_mask:
                self.mask = label_dict['mask']
            self.valid_ids = label_dict['valid_ids'].tolist()
        else:
            print('Processing dgl graphs from scratch...')
            if n_jobs > 1:
                self.graphs = pmap(smiles_to_graph,
                                   self.smiles,
                                   node_featurizer=node_featurizer,
                                   edge_featurizer=edge_featurizer,
                                   n_jobs=n_jobs)
            else:
                self.graphs = []
                for i, s in enumerate(self.smiles):
                    if (i + 1) % log_every == 0:
                        print('Processing molecule {:d}/{:d}'.format(i + 1, len(self)))
                    self.graphs.append(smiles_to_graph(s, node_featurizer=node_featurizer,
                                                       edge_featurizer=edge_featurizer))

            # Keep only valid molecules
            self.valid_ids = []
            graphs = []
            failed_mols = []
            for i, g in enumerate(self.graphs):
                if g is not None:
                    self.valid_ids.append(i)
                    graphs.append(g)
                else:
                    failed_mols.append((i, self.smiles[i]))

            if error_log is not None:
                if len(failed_mols) > 0:
                    failed_ids, failed_smis = map(list, zip(*failed_mols))
                else:
                    failed_ids, failed_smis = [], []
                df = pd.DataFrame({'raw_id': failed_ids, 'smiles': failed_smis})
                df.to_csv(error_log, index=False)

            self.graphs = graphs
            _label_values = self.df[self.task_names].values
            # np.nan_to_num will also turn inf into a very large number
            self.labels = F.zerocopy_from_numpy(
                np.nan_to_num(_label_values).astype(np.float32))[self.valid_ids]
            valid_ids = torch.tensor(self.valid_ids)
            if init_mask:
                self.mask = F.zerocopy_from_numpy(
                    (~np.isnan(_label_values)).astype(np.float32))[self.valid_ids]
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'mask': self.mask,
                                    'valid_ids': valid_ids})
            else:
                self.mask = None
                save_graphs(self.cache_file_path, self.graphs,
                            labels={'labels': self.labels, 'valid_ids': valid_ids})

        self.smiles = [self.smiles[i] for i in self.valid_ids]

    def __getitem__(self, item):
        """
        Get datapoint with index
        """
        if self.mask is not None:
            return self.smiles[item], self.graphs[item], self.labels[item], self.mask[item]
        else:
            return self.smiles[item], self.graphs[item], self.labels[item]

    def __len__(self):
        """
        Size for the dataset
        """
        return len(self.smiles)

    def task_pos_weights(self, indices):
        """
        Get weights for positive samples on each task
        This should only be used when all tasks are binary classification.
        It's quite common that the number of positive samples and the number of
        negative samples are significantly different for binary classification.
        To compensate for the class imbalance issue, we can weight each datapoint
        in loss computation.
        In particular, for each task we will set the weight of negative samples
        to be 1 and the weight of positive samples to be the number of negative
        samples divided by the number of positive samples.
        """
        task_pos_weights = torch.ones(self.labels.shape[1])
        num_pos = F.sum(self.labels[indices], dim=0)
        num_indices = F.sum(self.mask[indices], dim=0)
        task_pos_weights[num_pos > 0] = ((num_indices - num_pos) / num_pos)[num_pos > 0]

        return task_pos_weights

def mol_to_graph(mol, graph_constructor, node_featurizer, edge_featurizer,
                 canonical_atom_order, explicit_hydrogens=False, num_virtual_nodes=0):
    """
    Convert an RDKit molecule object into a DGLGraph and featurize for it.
    This function can be used to construct any arbitrary ``DGLGraph`` from an
    RDKit molecule instance.
    """
    if mol is None:
        print('Invalid mol found')
        return None

    # Whether to have hydrogen atoms as explicit nodes
    if explicit_hydrogens:
        mol = Chem.AddHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    g = graph_constructor(mol)

    if node_featurizer is not None:
        g.ndata.update(node_featurizer(mol))

    if edge_featurizer is not None:
        g.edata.update(edge_featurizer(mol))

    if num_virtual_nodes > 0:
        num_real_nodes = g.num_nodes()
        real_nodes = list(range(num_real_nodes))
        g.add_nodes(num_virtual_nodes)

        # Change Topology
        virtual_src = []
        virtual_dst = []
        for count in range(num_virtual_nodes):
            virtual_node = num_real_nodes + count
            virtual_node_copy = [virtual_node] * num_real_nodes
            virtual_src.extend(real_nodes)
            virtual_src.extend(virtual_node_copy)
            virtual_dst.extend(virtual_node_copy)
            virtual_dst.extend(real_nodes)
        g.add_edges(virtual_src, virtual_dst)

        for nk, nv in g.ndata.items():
            nv = torch.cat([nv, torch.zeros(g.num_nodes(), 1)], dim=1)
            nv[-num_virtual_nodes:, -1] = 1
            g.ndata[nk] = nv

        for ek, ev in g.edata.items():
            ev = torch.cat([ev, torch.zeros(g.num_edges(), 1)], dim=1)
            ev[-num_virtual_nodes * num_real_nodes * 2:, -1] = 1
            g.edata[ek] = ev

    return g

def construct_bigraph_from_mol(mol, add_self_loop=False):
    """Construct a bi-directed DGLGraph with topology only for the molecule.
    The **i** th atom in the molecule, i.e. ``mol.GetAtomWithIdx(i)``, corresponds to the
    **i** th node in the returned DGLGraph.
    The **i** th bond in the molecule, i.e. ``mol.GetBondWithIdx(i)``, corresponds to the
    **(2i)**-th and **(2i+1)**-th edges in the returned DGLGraph. The **(2i)**-th and
    **(2i+1)**-th edges will be separately from **u** to **v** and **v** to **u**, where
    **u** is ``bond.GetBeginAtomIdx()`` and **v** is ``bond.GetEndAtomIdx()``.
    If self loops are added, the last **n** edges will separately be self loops for
    atoms ``0, 1, ..., n-1``.
    """
    g = dgl.graph(([], []), idtype=torch.int32)

    # Add nodes
    num_atoms = mol.GetNumAtoms()
    g.add_nodes(num_atoms)

    # Add edges
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    if add_self_loop:
        nodes = g.nodes().tolist()
        src_list.extend(nodes)
        dst_list.extend(nodes)

    g.add_edges(torch.IntTensor(src_list), torch.IntTensor(dst_list))

    return g

def mol_to_bigraph(mol, add_self_loop=False,
                   node_featurizer=None,
                   edge_featurizer=None,
                   canonical_atom_order=True,
                   explicit_hydrogens=False,
                   num_virtual_nodes=0):
    return mol_to_graph(mol, partial(construct_bigraph_from_mol, add_self_loop=add_self_loop),
                        node_featurizer, edge_featurizer,
                        canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

def smiles_to_bigraph(smiles, add_self_loop=False,
                      node_featurizer=None,
                      edge_featurizer=None,
                      canonical_atom_order=True,
                      explicit_hydrogens=False,
                      num_virtual_nodes=0):
    mol = Chem.MolFromSmiles(smiles)
    return mol_to_bigraph(mol, add_self_loop, node_featurizer, edge_featurizer, canonical_atom_order, explicit_hydrogens, num_virtual_nodes)

def one_hot_encoding(x, allowable_set, encode_unknown=False):
    """
    One-hot encoding.
    """
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))

def collate_molgraphs(data):
    """
    Batching a list of datapoints for dataloader.
    """
    if len(data[0]) == 3:
        smiles, graphs, labels = map(list, zip(*data))
    else:
        smiles, graphs, labels, masks = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    if len(data[0]) == 3:
        masks = torch.ones(labels.shape)
    else:
        masks = torch.stack(masks, dim=0)

    return smiles, bg, labels, masks

def atom_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the type of an atom.
    """
    if allowable_set is None:
        allowable_set = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn',
                         'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au',
                         'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb']
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)

def atomic_number_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the atomic number of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(1, 101))
    return one_hot_encoding(atom.GetAtomicNum(), allowable_set, encode_unknown)

def atomic_number(atom):
    """
    Get the atomic number for an atom.
    """
    return [atom.GetAtomicNum()]

def atom_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the degree of an atom.
    Note that the result will be different depending on whether the Hs are explicitly modeled in the graph.
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)

def atom_degree(atom):
    """
    Get the degree of an atom.
    """
    return [atom.GetDegree()]

def atom_total_degree_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the degree of an atom including Hs.
    """
    if allowable_set is None:
        allowable_set = list(range(6))
    return one_hot_encoding(atom.GetTotalDegree(), allowable_set, encode_unknown)

def atom_total_degree(atom):
    """
    The degree of an atom including Hs.
    """
    return [atom.GetTotalDegree()]

def atom_explicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the explicit valence of an aotm.
    """
    if allowable_set is None:
        allowable_set = list(range(1, 7))
    return one_hot_encoding(atom.GetExplicitValence(), allowable_set, encode_unknown)

def atom_explicit_valence(atom):
    """
    Get the explicit valence of an atom.
    """
    return [atom.GetExplicitValence()]

def atom_implicit_valence_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the implicit valence of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)

def atom_implicit_valence(atom):
    """
    Get the implicit valence of an atom.
    """
    return [atom.GetImplicitValence()]

def atom_hybridization_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the hybridization of an atom.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.HybridizationType.SP,
                         Chem.rdchem.HybridizationType.SP2,
                         Chem.rdchem.HybridizationType.SP3,
                         Chem.rdchem.HybridizationType.SP3D,
                         Chem.rdchem.HybridizationType.SP3D2]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)

def atom_total_num_H_one_hot(atom, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the total number of Hs of an atom.
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)

def atom_total_num_H(atom):
    """Get the total number of Hs of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one int only.
    See Also
    --------
    atom_total_num_H_one_hot
    """
    return [atom.GetTotalNumHs()]

def atom_formal_charge_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the formal charge of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Formal charges to consider. Default: ``-2`` - ``2``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    atom_formal_charge
    """
    if allowable_set is None:
        allowable_set = list(range(-2, 3))
    return one_hot_encoding(atom.GetFormalCharge(), allowable_set, encode_unknown)

def atom_formal_charge(atom):
    """Get formal charge for an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one int only.
    See Also
    --------
    atom_formal_charge_one_hot
    """
    return [atom.GetFormalCharge()]

def atom_partial_charge(atom):
    """Get Gasteiger partial charge for an atom.
    For using this function, you must have called ``AllChem.ComputeGasteigerCharges(mol)``
    to compute Gasteiger charges.
    Occasionally, we can get nan or infinity Gasteiger charges, in which case we will set
    the result to be 0.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one float only.
    """
    gasteiger_charge = atom.GetProp('_GasteigerCharge')
    if gasteiger_charge in ['-nan', 'nan', '-inf', 'inf']:
        gasteiger_charge = 0
    return [float(gasteiger_charge)]

def atom_num_radical_electrons_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the number of radical electrons of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of int
        Number of radical electrons to consider. Default: ``0`` - ``4``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    atom_num_radical_electrons
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetNumRadicalElectrons(), allowable_set, encode_unknown)

def atom_num_radical_electrons(atom):
    """Get the number of radical electrons for an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one int only.
    See Also
    --------
    atom_num_radical_electrons_one_hot
    """
    return [atom.GetNumRadicalElectrons()]

def atom_is_aromatic_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is aromatic.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    atom_is_aromatic
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.GetIsAromatic(), allowable_set, encode_unknown)

def atom_is_aromatic(atom):
    """Get whether the atom is aromatic.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one bool only.
    See Also
    --------
    atom_is_aromatic_one_hot
    """
    return [atom.GetIsAromatic()]

def atom_is_in_ring_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for whether the atom is in ring.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of bool
        Conditions to consider. Default: ``False`` and ``True``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List of boolean values where at most one value is True.
    See Also
    --------
    one_hot_encoding
    atom_is_in_ring
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(atom.IsInRing(), allowable_set, encode_unknown)

def atom_is_in_ring(atom):
    """Get whether the atom is in ring.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one bool only.
    See Also
    --------
    atom_is_in_ring_one_hot
    """
    return [atom.IsInRing()]

def atom_chiral_tag_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chiral tag of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of rdkit.Chem.rdchem.ChiralType
        Chiral tags to consider. Default: ``rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW``,
        ``rdkit.Chem.rdchem.ChiralType.CHI_OTHER``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List containing one bool only.
    See Also
    --------
    one_hot_encoding
    atom_chirality_type_one_hot
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                         Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                         Chem.rdchem.ChiralType.CHI_OTHER]
    return one_hot_encoding(atom.GetChiralTag(), allowable_set, encode_unknown)

def atom_chirality_type_one_hot(atom, allowable_set=None, encode_unknown=False):
    """One hot encoding for the chirality type of an atom.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    allowable_set : list of str
        Chirality types to consider. Default: ``R``, ``S``.
    encode_unknown : bool
        If True, map inputs not in the allowable set to the
        additional last element. (Default: False)
    Returns
    -------
    list
        List containing one bool only.
    See Also
    --------
    one_hot_encoding
    atom_chiral_tag_one_hot
    """
    if not atom.HasProp('_CIPCode'):
        return [False, False]

    if allowable_set is None:
        allowable_set = ['R', 'S']
    return one_hot_encoding(atom.GetProp('_CIPCode'), allowable_set, encode_unknown)

def atom_mass(atom, coef=0.01):
    """Get the mass of an atom and scale it.
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    coef : float
        The mass will be multiplied by ``coef``.
    Returns
    -------
    list
        List containing one float only.
    """
    return [atom.GetMass() * coef]

def atom_is_chiral_center(atom):
    """Get whether the atom is chiral center
    Parameters
    ----------
    atom : rdkit.Chem.rdchem.Atom
        RDKit atom instance.
    Returns
    -------
    list
        List containing one bool only.
    """
    return [atom.HasProp('_ChiralityPossible')]


class BaseAtomFeaturizer(object):
    """
    An abstract class for atom featurizers.
    Loop over all atoms in a molecule and featurize them with the 'featurizer_funcs'.
    We assume the resulting DGLGraph will not contain any virtual nodes and a node i in the graph corresponds to exactly atom i in the molecule.
    """
    def __init__(self, featurizer_funcs, feat_sizes=None):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes

    def feat_size(self, feat_name=None):
        """
        Get the feature size for 'feat_name'.
        When there is only one feature, users do not need to provide 'feat_name'.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, \
                'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        if feat_name not in self._feat_sizes:
            atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol):
        """
        Featurize all atoms in a molecule.
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        return processed_features

class ConcatFeaturizer(object):
    """
    Concatenate the evaluation results of multiple functions as a single feature.
    """
    def __init__(self, func_list):
        self.func_list = func_list

    def __call__(self, x):
        """
        Featurize the input data.
        """
        return list(itertools.chain.from_iterable([func(x) for func in self.func_list]))

class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    """
    A default featurizer for atoms.
    The atom features include:
    * One hot encoding of the atom type.
      The supported atom types include
      [C, N, O, S, F, Si, P, Cl, Br, Mg, Na, Ca, Fe, As, Al, I, B, V, K, Tl, Yb, Sb, Sn, Ag, Pd, Co, Se, Ti, Zn, H, Li,
      Ge, Cu, Au, Ni, Cd, In, Mn, Zr, Cr, Pt, Hg, Pb]
    * One hot encoding of the atom degree.
      The supported possibilities include '0 - 10'.
    * One hot encoding of the number of implicit Hs on the atom.
      The supported possibilities include ``0 - 6``.
    * Formal charge of the atom.
    * Number of radical electrons of the atom.
    * One hot encoding of the atom hybridization.
      The supported possibilities include 'SP', SP2, SP3, SP3D, SP3D2.
    * Whether the atom is aromatic.
    * One hot encoding of the number of total Hs on the atom.
      The supported possibilities include 0 - 4.
    """
    def __init__(self, atom_data_field='h'):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer([atom_type_one_hot,
                                                                 atom_degree_one_hot,
                                                                 atom_implicit_valence_one_hot,
                                                                 atom_formal_charge,
                                                                 atom_num_radical_electrons,
                                                                 atom_hybridization_one_hot,
                                                                 atom_is_aromatic,
                                                                 atom_total_num_H_one_hot])})

class AttentiveFPAtomFeaturizer(BaseAtomFeaturizer):
    """
    The atom featurizer used in AttentiveFP.
    * One hot encoding of the atom type.
      The supported atom types include
      [B, C, N, O, F, Si, P, S, Cl, As, Se, Br, Te, I, At, and other]
    * One hot encoding of the atom degree**.
      The supported possibilities include 0 - 5.
    * Formal charge of the atom.
    * Number of radical electrons of the atom.
    * One hot encoding of the atom hybridization.
      The supported possibilities include SP, SP2, SP3, SP3D, SP3D2, and other.
    * Whether the atom is aromatic.
    * One hot encoding of the number of total Hs on the atom.
      The supported possibilities include 0 - 4.
    * Whether the atom is chiral center
    * One hot encoding of the atom chirality type.
      The supported possibilities include R, and S.
    """
    def __init__(self, atom_data_field='h'):
        super(AttentiveFPAtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer([partial(atom_type_one_hot,
                                                                         allowable_set=['B', 'C', 'N', 'O', 'F', 'Si',
                                                                                        'P', 'S', 'Cl', 'As', 'Se',
                                                                                        'Br', 'Te', 'I', 'At'],
                                                                         encode_unknown=True),
                                                                 partial(atom_degree_one_hot,
                                                                         allowable_set=list(range(6))),
                                                                 atom_formal_charge,
                                                                 atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot, encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 atom_total_num_H_one_hot,
                                                                 atom_is_chiral_center,
                                                                 atom_chirality_type_one_hot])})

class AtomFeaturizer(BaseAtomFeaturizer):
    """
    A default featurizer for atoms.
    The atom features include:
    * One hot encoding of the atom type.
      The supported atom types include
      [C, N, O, S, F, Si, P, Cl, Br, Mg, Na, Ca, Fe, As, Al, I, B, V, K, Tl, Yb, Sb, Sn, Ag, Pd, Co, Se, Ti, Zn, H, Li,
      Ge, Cu, Au, Ni, Cd, In, Mn, Zr, Cr, Pt, Hg, Pb]
    * One hot encoding of the atom degree.
      The supported possibilities include 0 - 10.
    * One hot encoding of the number of implicit Hs on the atom.
      The supported possibilities include 0 - 6.
    * Formal charge of the atom.
    * Number of radical electrons of the atom.
    * One hot encoding of the atom hybridization.
      The supported possibilities include SP, SP2, SP3, SP3D, SP3D2.
    * Whether the atom is aromatic.
    * One hot encoding of the number of total Hs on the atom.
      The supported possibilities include 0 - 4.
    """
    def __init__(self, atom_data_field='h'):
        super(AtomFeaturizer, self).__init__(
            featurizer_funcs={atom_data_field: ConcatFeaturizer([atom_type_one_hot,
                                                                 atom_degree_one_hot,
                                                                 atom_implicit_valence_one_hot,
                                                                 atom_formal_charge,
                                                                 atom_num_radical_electrons,
                                                                 partial(atom_hybridization_one_hot, encode_unknown=True),
                                                                 atom_is_aromatic,
                                                                 atom_total_num_H_one_hot,
                                                                 atom_is_chiral_center,
                                                                 atom_chirality_type_one_hot])})


def bond_type_one_hot(bond, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the type of a bond.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondType.SINGLE,
                         Chem.rdchem.BondType.DOUBLE,
                         Chem.rdchem.BondType.TRIPLE,
                         Chem.rdchem.BondType.AROMATIC]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)

def bond_is_conjugated_one_hot(bond, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for whether the bond is conjugated.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.GetIsConjugated(), allowable_set, encode_unknown)

def bond_is_conjugated(bond):
    """
    Get whether the bond is conjugated.
    """
    return [bond.GetIsConjugated()]

def bond_is_in_ring_one_hot(bond, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for whether the bond is in a ring of any size.
    """
    if allowable_set is None:
        allowable_set = [False, True]
    return one_hot_encoding(bond.IsInRing(), allowable_set, encode_unknown)

def bond_is_in_ring(bond):
    """
    Get whether the bond is in a ring of any size.
    """
    return [bond.IsInRing()]

def bond_stereo_one_hot(bond, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the stereo configuration of a bond.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondStereo.STEREONONE,
                         Chem.rdchem.BondStereo.STEREOANY,
                         Chem.rdchem.BondStereo.STEREOZ,
                         Chem.rdchem.BondStereo.STEREOE,
                         Chem.rdchem.BondStereo.STEREOCIS,
                         Chem.rdchem.BondStereo.STEREOTRANS]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

def bond_direction_one_hot(bond, allowable_set=None, encode_unknown=False):
    """
    One hot encoding for the direction of a bond.
    """
    if allowable_set is None:
        allowable_set = [Chem.rdchem.BondDir.NONE,
                         Chem.rdchem.BondDir.ENDUPRIGHT,
                         Chem.rdchem.BondDir.ENDDOWNRIGHT]
    return one_hot_encoding(bond.GetBondDir(), allowable_set, encode_unknown)


class BaseBondFeaturizer(object):
    """
    An abstract class for bond featurizers.
    Loop over all bonds in a molecule and featurize them with the 'featurizer_funcs'.
    We assume the constructed 'DGLGraph' is a bi-directed graph where the **i** th bond in the molecule,
    i.e. 'mol.GetBondWithIdx(i)', corresponds to the **(2i)**-th and **(2i+1)**-th edges in the DGLGraph.
    **We assume the resulting DGLGraph will be created with function 'smiles_to_bigraph' without self loops.**
    """
    def __init__(self, featurizer_funcs, feat_sizes=None, self_loop=False):
        self.featurizer_funcs = featurizer_funcs
        if feat_sizes is None:
            feat_sizes = dict()
        self._feat_sizes = feat_sizes
        self._self_loop = self_loop

    def feat_size(self, feat_name=None):
        """
        Get the feature size for 'feat_name'.
        When there is only one feature, users do not need to provide 'feat_name'.
        """
        if feat_name is None:
            assert len(self.featurizer_funcs) == 1, 'feat_name should be provided if there are more than one features'
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            return ValueError('Expect feat_name to be in {}, got {}'.format(
                list(self.featurizer_funcs.keys()), feat_name))

        mol = Chem.MolFromSmiles('CCO')
        feats = self(mol)

        return feats[feat_name].shape[1]

    def __call__(self, mol):
        """
        Featurize all bonds in a molecule.
        """
        num_bonds = mol.GetNumBonds()
        bond_features = defaultdict(list)

        # Compute features for each bond
        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                bond_features[feat_name].extend([feat, feat.copy()])

        # Stack the features and convert them to float arrays
        processed_features = dict()
        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list)
            processed_features[feat_name] = F.zerocopy_from_numpy(feat.astype(np.float32))

        if self._self_loop and num_bonds > 0:
            num_atoms = mol.GetNumAtoms()
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.cat([feats, torch.zeros(feats.shape[0], 1)], dim=1)
                self_loop_feats = torch.zeros(num_atoms, feats.shape[1])
                self_loop_feats[:, -1] = 1
                feats = torch.cat([feats, self_loop_feats], dim=0)
                processed_features[feat_name] = feats

        if self._self_loop and num_bonds == 0:
            num_atoms = mol.GetNumAtoms()
            toy_mol = Chem.MolFromSmiles('CO')
            processed_features = self(toy_mol)
            for feat_name in processed_features:
                feats = processed_features[feat_name]
                feats = torch.zeros(num_atoms, feats.shape[1])
                feats[:, -1] = 1
                processed_features[feat_name] = feats

        return processed_features

class CanonicalBondFeaturizer(BaseBondFeaturizer):
    """
    A default featurizer for bonds.
    The bond features include:
    * One hot encoding of the bond type.
      The supported bond types include 'SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC'.
    * Whether the bond is conjugated.
    * Whether the bond is in a ring of any size.
    * One hot encoding of the stereo configuration of a bond.
      The supported bond stereo configurations include
      'STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'.
    """

    def __init__(self, bond_data_field='e', self_loop=False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer([bond_type_one_hot,
                                                                 bond_is_conjugated,
                                                                 bond_is_in_ring,
                                                                 bond_stereo_one_hot])},
            self_loop=self_loop)

class AttentiveFPBondFeaturizer(BaseBondFeaturizer):
    """
    The bond featurizer used in AttentiveFP
    The bond features include:
    * One hot encoding of the bond type.
      The supported bond types include SINGLE, DOUBLE, TRIPLE, AROMATIC.
    * Whether the bond is conjugated.
    * Whether the bond is in a ring of any size.
    * One hot encoding of the stereo configuration of a bond.
      The supported bond stereo configurations include STEREONONE, STEREOANY, STEREOZ, STEREOE.
    """
    def __init__(self, bond_data_field='e', self_loop=False):
        super(AttentiveFPBondFeaturizer, self).__init__(
            featurizer_funcs={bond_data_field: ConcatFeaturizer([bond_type_one_hot,
                                                                 bond_is_conjugated,
                                                                 bond_is_in_ring,
                                                                 partial(bond_stereo_one_hot,
                                                                         allowable_set=[Chem.rdchem.BondStereo.STEREONONE,
                                                                                        Chem.rdchem.BondStereo.STEREOANY,
                                                                                        Chem.rdchem.BondStereo.STEREOZ,
                                                                                        Chem.rdchem.BondStereo.STEREOE])])},
            self_loop=self_loop)

def train_val_split_dataset(dataset, frac_list = None, shuffle = False, random_state = None):
    if frac_list is None:
        frac_list = [0.8, 0.2]
    frac_list = np.asarray(frac_list)
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    return [Subset(dataset, indices[offset - length:offset]) for offset, length in zip(accumulate(lengths), lengths)]

def base_k_fold_split(split_method, dataset, k, log):
    """Split dataset for k-fold cross validation.
    Parameters
    ----------
    split_method : callable
        Arbitrary method for splitting the dataset
        into training, validation and test subsets.
    dataset
        We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
        gives the ith datapoint.
    k : int
        Number of folds to use and should be no smaller than 2.
    log : bool
        Whether to print a message at the start of preparing each fold.
    Returns
    -------
    all_folds : list of 2-tuples
        Each element of the list represents a fold and is a 2-tuple (train_set, val_set),
        which are all :class:`Subset` instances.
    """
    assert k >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k)
    all_folds = []
    frac_per_part = 1. / k
    for i in range(k):
        if log:
            print('Processing fold {:d}/{:d}'.format(i + 1, k))
        # We are reusing the code for train-validation-test split.
        train_set1, val_set, train_set2 = split_method(dataset,
                                                       frac_train=i * frac_per_part,
                                                       frac_val=frac_per_part,
                                                       frac_test=1. - (i + 1) * frac_per_part)
        # For cross validation, each fold consists of only a train subset and
        # a validation subset.
        train_set = Subset(dataset, np.concatenate([train_set1.indices, train_set2.indices]).astype(np.int64))
        all_folds.append((train_set, val_set))
    return all_folds

def indices_split(dataset, frac_train, frac_val, frac_test, indices):
    """
    Reorder datapoints based on the specified indices and then take consecutive chunks as subsets.
    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
        gives the ith datapoint.
    frac_train : float
        Fraction of data to use for training.
    frac_val : float
        Fraction of data to use for validation.
    frac_test : float
        Fraction of data to use for test.
    indices : list or ndarray
        Indices specifying the order of datapoints.
    Returns
    -------
    list of length 3
        Subsets for training, validation and test, which are all :class:`Subset` instances.
    """
    frac_list = np.array([frac_train, frac_val, frac_test])
    assert np.allclose(np.sum(frac_list), 1.), \
        'Expect frac_list sum to 1, got {:.4f}'.format(np.sum(frac_list))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])

    return [Subset(dataset, list(indices[offset - length:offset]))
            for offset, length in zip(accumulate(lengths), lengths)]

def count_and_log(message, i, total, log_every_n):
    """Print a message to reflect the progress of processing once a while.
    Parameters
    ----------
    message : str
        Message to print.
    i : int
        Current index.
    total : int
        Total count.
    log_every_n : None or int
        Molecule related computation can take a long time for a large dataset and we want
        to learn the progress of processing. This can be done by printing a message whenever
        a batch of ``log_every_n`` molecules have been processed. If None, no messages will
        be printed.
    """
    if (log_every_n is not None) and ((i + 1) % log_every_n == 0):
        print('{} {:d}/{:d}'.format(message, i + 1, total))

def prepare_mols(dataset, mols, sanitize, log_every_n=1000):
    """Prepare RDKit molecule instances.
    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
        gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
        ith datapoint.
    mols : None or list of rdkit.Chem.rdchem.Mol
        None or pre-computed RDKit molecule instances. If not None, we expect a
        one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
        ``mols[i]`` corresponds to ``dataset.smiles[i]``.
    sanitize : bool
        This argument only comes into effect when ``mols`` is None and decides whether
        sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
    log_every_n : None or int
        Molecule related computation can take a long time for a large dataset and we want
        to learn the progress of processing. This can be done by printing a message whenever
        a batch of ``log_every_n`` molecules have been processed. If None, no messages will
        be printed. Default to 1000.
    Returns
    -------
    mols : list of rdkit.Chem.rdchem.Mol
        RDkit molecule instances where there is a one-on-one correspondence between
        ``dataset.smiles`` and ``mols``, i.e. ``mols[i]`` corresponds to ``dataset.smiles[i]``.
    """
    if mols is not None:
        # Sanity check
        assert len(mols) == len(dataset), \
            'Expect mols to be of the same size as that of the dataset, ' \
            'got {:d} and {:d}'.format(len(mols), len(dataset))
    else:
        if log_every_n is not None:
            print('Start initializing RDKit molecule instances...')
        mols = []
        for i, s in enumerate(dataset.smiles):
            count_and_log('Creating RDKit molecule instance',
                          i, len(dataset.smiles), log_every_n)
            mols.append(Chem.MolFromSmiles(s, sanitize=sanitize))

    return mols

class RandomSplitter(object):
    """
    Randomly reorder datasets and then split them.
    The dataset is split with permutation and the splitting is hence random.
    """

    @staticmethod
    def train_val_split(dataset, frac_train=0.8, frac_val=0.2, random_state=None):
        """

        :rtype: object
        """
        return train_val_split_dataset(dataset, frac_list = [frac_train, frac_val], shuffle=True, random_state=random_state)

    @staticmethod
    def k_fold_split(dataset, k=5, random_state=None, log=True):
        """Randomly permute the dataset and then split it
        for k-fold cross validation by taking consecutive chunks.
        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset and ``dataset[i]``
            gives the ith datapoint.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        random_state : None, int or array_like, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array
            (or other sequence) of such integers, or None (the default).
            If seed is None, then RandomState will try to read data from /dev/urandom
            (or the Windows analogue) if available or seed from the clock otherwise.
        log : bool
            Whether to print a message at the start of preparing each fold. Default to True.
        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple ``(train_set, val_set)``.
            ``train_set`` and ``val_set`` also have ``len(dataset)`` and ``dataset[i]`` behaviors.
        """
        # Permute the dataset only once so that each datapoint
        # will appear once in exactly one fold.
        indices = np.random.RandomState(seed=random_state).permutation(len(dataset))

        return base_k_fold_split(partial(indices_split, indices=indices), dataset, k, log)

# pylint: disable=W0702
class ScaffoldSplitter(object):
    """Group molecules based on their Bemis-Murcko scaffolds and then split the groups.
    Group molecules so that all molecules in a group have a same scaffold (see reference).
    The dataset is then split at the level of groups.
    References
    ----------
    Bemis, G. W.; Murcko, M. A. “The Properties of Known Drugs.
        1. Molecular Frameworks.” J. Med. Chem. 39:2887-93 (1996).
    """

    @staticmethod
    def get_ordered_scaffold_sets(molecules, log_every_n, scaffold_func):
        """Group molecules based on their Bemis-Murcko scaffolds and
        order these groups based on their sizes.
        The order is decided by comparing the size of groups, where groups with a larger size
        are placed before the ones with a smaller size.
        Parameters
        ----------
        molecules : list of rdkit.Chem.rdchem.Mol
            Pre-computed RDKit molecule instances. We expect a one-on-one
            correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed.
        scaffold_func : str
            The function to use for computing scaffolds, which can be 'murcko_decompose' for
            using rdkit.Chem.AllChem.MurckoDecompose or 'scaffold_smiles' for using
            rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles.
        Returns
        -------
        scaffold_sets : list
            Each element of the list is a list of int,
            representing the indices of compounds with a same scaffold.
        """
        assert scaffold_func in ['decompose', 'smiles'], \
            "Expect scaffold_func to be 'decompose' or 'smiles', " \
            "got '{}'".format(scaffold_func)

        if log_every_n is not None:
            print('Start computing Bemis-Murcko scaffolds.')
        scaffolds = defaultdict(list)
        for i, mol in enumerate(molecules):
            count_and_log('Computing Bemis-Murcko for compound',
                          i, len(molecules), log_every_n)
            # For mols that have not been sanitized, we need to compute their ring information
            try:
                FastFindRings(mol)
                if scaffold_func == 'decompose':
                    mol_scaffold = Chem.MolToSmiles(AllChem.MurckoDecompose(mol))
                if scaffold_func == 'smiles':
                    mol_scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                        mol=mol, includeChirality=False)
                # Group molecules that have the same scaffold
                scaffolds[mol_scaffold].append(i)
            except:
                print('Failed to compute the scaffold for molecule {:d} '
                      'and it will be excluded.'.format(i + 1))

        # Order groups of molecules by first comparing the size of groups
        # and then the index of the first compound in the group.
        scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
        scaffold_sets = [
            scaffold_set for (scaffold, scaffold_set) in sorted(
                scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
        ]

        return scaffold_sets

    @staticmethod
    def train_val_test_split(dataset,
                             mols=None,
                             sanitize=True,
                             frac_train=0.8,
                             frac_val=0.2,
                             log_every_n=1000,
                             scaffold_func='decompose'):
        """Split the dataset into training, validation and test set based on molecular scaffolds.
        This spliting method ensures that molecules with a same scaffold will be collectively
        in only one of the training, validation or test set. As a result, the fraction
        of dataset to use for training and validation tend to be smaller than ``frac_train``
        and ``frac_val``, while the fraction of dataset to use for test tends to be larger
        than ``frac_test``.
        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to True.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.
        scaffold_func : str
            The function to use for computing scaffolds, which can be 'decompose' for
            using rdkit.Chem.AllChem.MurckoDecompose or 'smiles' for using
            rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles.
        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which also have ``len(dataset)`` and
            ``dataset[i]`` behaviors
        """
        # Perform sanity check first as molecule related computation can take a long time.
        # train_val_test_sanity_check(frac_train, frac_val, frac_test)
        molecules = prepare_mols(dataset, mols, sanitize)
        scaffold_sets = ScaffoldSplitter.get_ordered_scaffold_sets(
            molecules, log_every_n, scaffold_func)
        train_indices, val_indices, test_indices = [], [], []
        train_cutoff = int(frac_train * len(molecules))
        val_cutoff = int((frac_train + frac_val) * len(molecules))
        for group_indices in scaffold_sets:
            if len(train_indices) + len(group_indices) > train_cutoff:
                if len(train_indices) + len(val_indices) + len(group_indices) > val_cutoff:
                    test_indices.extend(group_indices)
                else:
                    val_indices.extend(group_indices)
            else:
                train_indices.extend(group_indices)
        return [Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices)]

    @staticmethod
    def train_val_split(dataset,
                        mols=None,
                        sanitize=True,
                        frac_train=0.8,
                        frac_val=0.2,
                        log_every_n=1000,
                        scaffold_func='decompose'):
        """Split the dataset into training, validation and test set based on molecular scaffolds.
        This spliting method ensures that molecules with a same scaffold will be collectively
        in only one of the training, validation or test set. As a result, the fraction
        of dataset to use for training and validation tend to be smaller than ``frac_train``
        and ``frac_val``, while the fraction of dataset to use for test tends to be larger
        than ``frac_test``.
        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to True.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.
        scaffold_func : str
            The function to use for computing scaffolds, which can be 'decompose' for
            using rdkit.Chem.AllChem.MurckoDecompose or 'smiles' for using
            rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles.
        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which also have ``len(dataset)`` and
            ``dataset[i]`` behaviors
        """
        # Perform sanity check first as molecule related computation can take a long time.
        # train_val_test_sanity_check(frac_train, frac_val, frac_test)
        molecules = prepare_mols(dataset, mols, sanitize)
        scaffold_sets = ScaffoldSplitter.get_ordered_scaffold_sets(molecules, log_every_n, scaffold_func)
        train_indices, val_indices = [], []
        train_cutoff = int(frac_train * len(molecules))
        val_cutoff = int((frac_train + frac_val) * len(molecules))
        for group_indices in scaffold_sets:
            if len(train_indices) + len(group_indices) > train_cutoff:
                val_indices.extend(group_indices)
            else:
                train_indices.extend(group_indices)
        return [Subset(dataset, train_indices), Subset(dataset, val_indices)]

    @staticmethod
    def k_fold_split(dataset, mols=None, sanitize=True, k=5, log_every_n=1000, scaffold_func='decompose'):
        """Group molecules based on their scaffolds and sort groups based on their sizes.
        The groups are then split for k-fold cross validation.
        Same as usual k-fold splitting methods, each molecule will appear only once
        in the validation set among all folds. In addition, this method ensures that
        molecules with a same scaffold will be collectively in either the training
        set or the validation set for each fold.
        Note that the folds can be highly imbalanced depending on the
        scaffold distribution in the dataset.
        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        mols : None or list of rdkit.Chem.rdchem.Mol
            None or pre-computed RDKit molecule instances. If not None, we expect a
            one-on-one correspondence between ``dataset.smiles`` and ``mols``, i.e.
            ``mols[i]`` corresponds to ``dataset.smiles[i]``. Default to None.
        sanitize : bool
            This argument only comes into effect when ``mols`` is None and decides whether
            sanitization is performed in initializing RDKit molecule instances. See
            https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
            Default to True.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        log_every_n : None or int
            Molecule related computation can take a long time for a large dataset and we want
            to learn the progress of processing. This can be done by printing a message whenever
            a batch of ``log_every_n`` molecules have been processed. If None, no messages will
            be printed. Default to 1000.
        scaffold_func : str
            The function to use for computing scaffolds, which can be 'decompose' for
            using rdkit.Chem.AllChem.MurckoDecompose or 'smiles' for using
            rdkit.Chem.Scaffolds.MurckoScaffold.MurckoScaffoldSmiles.
        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple ``(train_set, val_set)``.
            ``train_set`` and ``val_set`` also have ``len(dataset)`` and ``dataset[i]`` behaviors.
        """
        assert k >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k)

        molecules = prepare_mols(dataset, mols, sanitize)
        scaffold_sets = ScaffoldSplitter.get_ordered_scaffold_sets(
            molecules, log_every_n, scaffold_func)

        # k buckets that form a relatively balanced partition of the dataset
        index_buckets = [[] for _ in range(k)]
        for group_indices in scaffold_sets:
            bucket_chosen = int(np.argmin([len(bucket) for bucket in index_buckets]))
            index_buckets[bucket_chosen].extend(group_indices)

        all_folds = []
        for i in range(k):
            if log_every_n is not None:
                print('Processing fold {:d}/{:d}'.format(i + 1, k))
            train_indices = list(chain.from_iterable(index_buckets[:i] + index_buckets[i + 1:]))
            val_indices = index_buckets[i]
            all_folds.append((Subset(dataset, train_indices), Subset(dataset, val_indices)))

        return all_folds


class SingleTaskStratifiedSplitter(object):
    """Splits the dataset by stratification on a single task.
    We sort the molecules based on their label values for a task and then repeatedly
    take buckets of datapoints to augment the training, validation and test subsets.
    """

    @staticmethod
    def train_val_test_split(dataset, labels, task_id, frac_train=0.8, frac_val=0.1,
                             frac_test=0.1, bucket_size=10, random_state=None):
        """Split the dataset into training, validation and test subsets as stated above.
        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        labels : tensor of shape (N, T)
            Dataset labels all tasks. N for the number of datapoints and T for the number
            of tasks.
        task_id : int
            Index for the task.
        frac_train : float
            Fraction of data to use for training. By default, we set this to be 0.8, i.e.
            80% of the dataset is used for training.
        frac_val : float
            Fraction of data to use for validation. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for validation.
        frac_test : float
            Fraction of data to use for test. By default, we set this to be 0.1, i.e.
            10% of the dataset is used for test.
        bucket_size : int
            Size of bucket of datapoints. Default to 10.
        random_state : None, int or array_like, optional
            Random seed used to initialize the pseudo-random number generator.
            Can be any integer between 0 and 2**32 - 1 inclusive, an array
            (or other sequence) of such integers, or None (the default).
            If seed is None, then RandomState will try to read data from /dev/urandom
            (or the Windows analogue) if available or seed from the clock otherwise.
        Returns
        -------
        list of length 3
            Subsets for training, validation and test, which also have ``len(dataset)``
            and ``dataset[i]`` behaviors
        """
        # train_val_test_sanity_check(frac_train, frac_val, frac_test)

        if random_state is not None:
            np.random.seed(random_state)

        if not isinstance(labels, np.ndarray):
            labels = F.asnumpy(labels)
        task_labels = labels[:, task_id]
        sorted_indices = np.argsort(task_labels)

        train_bucket_cutoff = int(np.round(frac_train * bucket_size))
        val_bucket_cutoff = int(np.round(frac_val * bucket_size)) + train_bucket_cutoff

        train_indices, val_indices, test_indices = [], [], []

        while sorted_indices.shape[0] >= bucket_size:
            current_batch, sorted_indices = np.split(sorted_indices, [bucket_size])
            shuffled = np.random.permutation(range(bucket_size))
            train_indices.extend(
                current_batch[shuffled[:train_bucket_cutoff]].tolist())
            val_indices.extend(
                current_batch[shuffled[train_bucket_cutoff:val_bucket_cutoff]].tolist())
            test_indices.extend(
                current_batch[shuffled[val_bucket_cutoff:]].tolist())

        # Place rest samples in the training set.
        train_indices.extend(sorted_indices.tolist())

        return [Subset(dataset, train_indices),
                Subset(dataset, val_indices),
                Subset(dataset, test_indices)]

    @staticmethod
    def k_fold_split(dataset, labels, task_id, k=5, log=True):
        """Sort molecules based on their label values for a task and then split them
        for k-fold cross validation by taking consecutive chunks.
        Parameters
        ----------
        dataset
            We assume ``len(dataset)`` gives the size for the dataset, ``dataset[i]``
            gives the ith datapoint and ``dataset.smiles[i]`` gives the SMILES for the
            ith datapoint.
        labels : tensor of shape (N, T)
            Dataset labels all tasks. N for the number of datapoints and T for the number
            of tasks.
        task_id : int
            Index for the task.
        k : int
            Number of folds to use and should be no smaller than 2. Default to be 5.
        log : bool
            Whether to print a message at the start of preparing each fold.
        Returns
        -------
        list of 2-tuples
            Each element of the list represents a fold and is a 2-tuple ``(train_set, val_set)``.
            ``train_set`` and ``val_set`` also have ``len(dataset)`` and ``dataset[i]`` behaviors.
        """
        if not isinstance(labels, np.ndarray):
            labels = F.asnumpy(labels)
        task_labels = labels[:, task_id]
        sorted_indices = np.argsort(task_labels).tolist()

        return base_k_fold_split(partial(indices_split, indices=sorted_indices), dataset, k, log)

