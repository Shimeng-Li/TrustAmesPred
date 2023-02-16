import pandas as pd
import torch
from torch.utils.data import DataLoader
from gat_model_build import gat_classifier
from data_utils import MoleculeCSVDataset, smiles_to_bigraph, collate_molgraphs
from data_utils import AttentiveFPBondFeaturizer, AtomFeaturizer
from utils import mkdir_p, predict
import json
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, confusion_matrix
import numpy as np

data_args = {'ext_file': '../dataset/ames/data_v2/external_baseline.csv',
             'ext_cache_file_path': './data_transform/external_baseline.bin',
             'task_name': ['ames'],
             'num_workers': 6,
             "batch_size": 512,
             'random_state': 0,
             'node_featurizer': AtomFeaturizer(),
             'edge_featurizer': AttentiveFPBondFeaturizer(),
             'load': False}

model_args = {'param_path': './gat_result/param_opt/param_opt_v3/test_opt',
              'result_path': './gat_result/metric_calc/test',
              'model_path': './gat_result/gat_final_model/final_model_v3',
              'device': 'cuda:0',
              "residual": True}

with open ('{}/best_param.json'.format(model_args['param_path']), 'r') as file:
    param_config = json.load(file)

model_args.update(param_config)

mkdir_p(model_args['result_path'])
ext_filed_df = pd.read_csv(data_args['ext_file'])

ames_test_dataset = MoleculeCSVDataset(df = ext_filed_df,
                                       smiles_to_graph = smiles_to_bigraph,
                                       node_featurizer = data_args['node_featurizer'],
                                       edge_featurizer = data_args['edge_featurizer'],
                                       smiles_column = 'smiles',
                                       task_names = data_args['task_name'],
                                       log_every = 1000,
                                       cache_file_path = data_args['ext_cache_file_path'],
                                       n_jobs = data_args['num_workers'],
                                       load=data_args['load'])

test_loader = DataLoader(ames_test_dataset,
                         batch_size=model_args['batch_size'],
                         shuffle=True,
                         collate_fn=collate_molgraphs,
                         num_workers=data_args['num_workers'])

model_args['in_node_feats'] = data_args['node_featurizer'].feat_size()
model_args['n_tasks'] = ames_test_dataset.n_tasks

# model
model = gat_classifier(in_feats=model_args['in_node_feats'],
                       hidden_feats=[model_args['gnn_hidden_feats']] * model_args['num_gnn_layers'],
                       num_heads=[model_args['num_heads']] * model_args['num_gnn_layers'],
                       feat_drops=[model_args['feat_dropout']] * model_args['num_gnn_layers'],
                       attn_drops=[model_args['attn_dropout']] * model_args['num_gnn_layers'],
                       alphas=[model_args['alpha']] * model_args['num_gnn_layers'],
                       residuals=[model_args['residual']] * model_args['num_gnn_layers'],
                       predictor_hidden_feats=model_args['predictor_hidden_feats'],
                       predictor_dropout=model_args['predictor_dropout'],
                       n_tasks=model_args['n_tasks'])

model.to(model_args['device'])
model.load_state_dict(torch.load(model_args['model_path'] + '/model.pth')['model_state_dict'])
model.eval()
with torch.no_grad():
    for batch_id, batch_data in enumerate(test_loader):
        smiles, bg, labels, masks = batch_data
        labels = labels.to(model_args['device'])
        logits = predict(model_args, model, bg)

auc = roc_auc_score(labels.long().cpu().numpy(), logits.detach().cpu().numpy())
tn, fp, fn, tp = confusion_matrix(labels.long().cpu().numpy(), np.round(torch.sigmoid(logits).detach().cpu().numpy())).ravel()
acc = accuracy_score(labels.long().cpu().numpy(), np.round(torch.sigmoid(logits).detach().cpu().numpy()))
sen = tp / (tp + fn)
spc = tn / (tn + fp)