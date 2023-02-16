import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from gat_model_build import gat_classifier
from data_utils import MoleculeCSVDataset, smiles_to_bigraph, RandomSplitter, collate_molgraphs
from data_utils import AttentiveFPBondFeaturizer, AtomFeaturizer
from utils import mkdir_p, Metrics, predict, EarlyStopping
import numpy as np
import json
# import matplotlib.pyplot as plt

data_args = {'train_file': '../dataset/ames/data_v2/train_baseline.csv',
             'ext_file': '../dataset/ames/data_v2/external_baseline.csv',
             'test_cache_file_path': './data_transform/bootstrap_ext.bin',
             'task_name': ['ames'],
             'num_workers': 20,
             'random_state': 0,
             'node_featurizer': AtomFeaturizer(),
             'edge_featurizer': AttentiveFPBondFeaturizer(),
             'load': True,
             'sample_times': 10}

model_args = {'param_path': './gat_result/param_opt/param_opt_v3/test_opt',
              'result_path': './gat_result/uncertainty_estimation/bootstrap',
              'metric': 'roc_auc_score',
              'device': 'cuda:0',
              'num_epochs': 500,
              "alpha": 0.5,
              "feat_dropout": 0.4,
              "attn_dropout": 0.4,
              "predictor_dropout": 0.4,
              "gnn_hidden_feats": 256,
              "lr": 0.001,
              "num_gnn_layers": 3,
              "num_heads": 8,
              "patience": 50,
              "predictor_hidden_feats": 256,
              "residual": True,
              "weight_decay": 0.0001}

with open ('{}/best_param.json'.format(model_args['param_path']), 'r') as file:
    param_config = json.load(file)

model_args.update(param_config)
mkdir_p(model_args['result_path'])
train_file_df = pd.read_csv(data_args['train_file'])
ext_filed_df = pd.read_csv(data_args['ext_file'])

ames_test_dataset = MoleculeCSVDataset(df = ext_filed_df,
                                       smiles_to_graph = smiles_to_bigraph,
                                       node_featurizer = data_args['node_featurizer'],
                                       edge_featurizer = None,
                                       smiles_column = 'smiles',
                                       task_names = data_args['task_name'],
                                       log_every = 1000,
                                       cache_file_path = data_args['test_cache_file_path'],
                                       n_jobs = data_args['num_workers'])

test_loader = DataLoader(ames_test_dataset,
                         batch_size=model_args['batch_size'],
                         shuffle=True,
                         collate_fn=collate_molgraphs,
                         num_workers=data_args['num_workers'])

data_args['n_tasks'] = ames_test_dataset.n_tasks
model_args['in_node_feats'] = data_args['node_featurizer'].feat_size()
model_args['n_tasks'] = data_args['n_tasks']

splitter = RandomSplitter()
all_sample_list = []
train_set_list = []
val_set_list = []
test_score_list = []
for sample_i in range(data_args['sample_times']):
    train_cache_path = './data_transform/bootstrap/train_'+str(sample_i+1)+'.bin'
    new_train_df = train_file_df.sample(n=len(train_file_df),replace=True)
    all_sample_list.append(new_train_df)
    new_train_dataset = MoleculeCSVDataset(df=train_file_df,
                                            smiles_to_graph=smiles_to_bigraph,
                                            node_featurizer=data_args['node_featurizer'],
                                            edge_featurizer=data_args['edge_featurizer'],
                                            smiles_column='smiles',
                                            task_names=data_args['task_name'],
                                            log_every=1000,
                                            cache_file_path=train_cache_path,
                                            n_jobs=data_args['num_workers'],
                                            load=data_args['load'])
    train_set, val_set = splitter.train_val_split(new_train_dataset,
                                                  frac_train=0.8,
                                                  frac_val=0.2,
                                                  random_state=model_args['random_state'])
    train_loader = DataLoader(train_set,
                              batch_size=model_args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs,
                              num_workers=data_args['num_workers'])
    val_loader = DataLoader(val_set,
                            batch_size=model_args['batch_size'],
                            shuffle=True,
                            collate_fn=collate_molgraphs,
                            num_workers=data_args['num_workers'])
    dir_name = model_args['result_path'] + '/model_' + str(sample_i + 1)
    mkdir_p(dir_name)
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
    loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = Adam(model.parameters(), lr = model_args['lr'], weight_decay=model_args['weight_decay'])
    stopper = EarlyStopping(patience=model_args['patience'],
                            filename=dir_name+'/model.pth',
                            metric=model_args['metric'])
    train_score_list = []
    val_score_list = []
    train_loss_list = []
    for epoch in range(model_args['num_epochs']):
        model.train()
        train_meter = Metrics()
        for batch_id, batch_data in enumerate(train_loader):
            smiles, bg, labels, masks = batch_data
            labels, masks = labels.to(model_args['device']), masks.to(model_args['device'])
            logits = predict(model_args, model, bg)
            # Mask non-existing labels
            loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_meter.update(logits, labels, masks)
        train_score = np.mean(train_meter.compute_metric(model_args['metric']))
        print('epoch {:d}/{:d}, training {} {:.4f}, training loss {:.4f}'.format(epoch + 1, model_args['num_epochs'],
                                                                                 model_args['metric'], train_score,
                                                                                 loss.item()))
        train_score_list.append(train_score)
        train_loss_list.append(loss.item())
        model.eval()
        eval_meter = Metrics()
        with torch.no_grad():
            for batch_id, batch_data in enumerate(val_loader):
                smiles, bg, labels, masks = batch_data
                labels = labels.to(model_args['device'])
                logits = predict(model_args, model, bg)
                eval_meter.update(logits, labels, masks)
        val_score = np.mean(eval_meter.compute_metric(model_args['metric']))
        early_stop = stopper.step(val_score, model)
        val_score_list.append(val_score)
        print('epoch {:d}/{:d}, validation AUC {:.4f}, best validation AUC {:.4f}'.format(epoch + 1,
                                                                                          model_args['num_epochs'],
                                                                                          val_score,
                                                                                          max(val_score_list)))
        if early_stop:
            break
    np.save('%s/train_loss.npy' % (dir_name),  np.array(train_loss_list, dtype=object))
    np.save('%s/train_auc.npy' % (dir_name),  np.array(train_score_list, dtype=object))
    np.save('%s/val_auc.npy' % (dir_name),  np.array(val_score_list, dtype=object))
    model.load_state_dict(torch.load(dir_name + '/model.pth')['model_state_dict'])
    model.eval()
    eval_meter = Metrics()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(test_loader):
            smiles, bg, labels, masks = batch_data
            labels = labels.to(model_args['device'])
            logits = predict(model_args, model, bg)
            eval_meter.update(logits, labels, masks)
    test_score = np.mean(eval_meter.compute_metric(model_args['metric']))
    print('test {} {:.4f}'.format(model_args['metric'], test_score))
    test_score_list.append(test_score)

np.save('%s/test_auc.npy' % (model_args['result_path']),  np.array(test_score_list, dtype=object))
