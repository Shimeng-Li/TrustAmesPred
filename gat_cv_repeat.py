import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from gat_model_build import gat_classifier
from data_utils import MoleculeCSVDataset, smiles_to_bigraph, RandomSplitter, ScaffoldSplitter, collate_molgraphs
from data_utils import AttentiveFPBondFeaturizer, AtomFeaturizer
from utils import mkdir_p, Metrics, predict, EarlyStopping
import numpy as np
import json
# import matplotlib.pyplot as plt

data_args = {'train_file': '../dataset/ames/data_v2/train_baseline.csv',
             'ext_file': '../dataset/ames/data_v2/external_baseline.csv',
             'train_cache_file_path': './data_transform/train_cv_cache.bin',
             'test_cache_file_path': './data_transform/external_cv_cache.bin',
             'task_name': ['ames'],
             'num_workers': 1,
             "batch_size": 512,
             'random_state': 0,
             'node_featurizer': AtomFeaturizer(),
             'edge_featurizer': AttentiveFPBondFeaturizer(),
             'split': 'random',
             'load': True,
             'fold': 5,
             'repeat': 10}

model_args = {'param_path': './gat_result/param_opt/param_opt_v3/test_opt',
              'cv_result_path': './gat_result/result_summary/model_performance/cv',
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

mkdir_p(model_args['cv_result_path'])
train_file_df = pd.read_csv(data_args['train_file'])
ext_filed_df = pd.read_csv(data_args['ext_file'])

ames_train_dataset = MoleculeCSVDataset(df = train_file_df,
                                        smiles_to_graph = smiles_to_bigraph,
                                        node_featurizer = data_args['node_featurizer'],
                                        edge_featurizer = data_args['edge_featurizer'],
                                        smiles_column = 'smiles',
                                        task_names = data_args['task_name'],
                                        log_every = 1000,
                                        cache_file_path = data_args['train_cache_file_path'],
                                        n_jobs = data_args['num_workers'],
                                        load = data_args['load'])

ames_test_dataset = MoleculeCSVDataset(df = ext_filed_df,
                                       smiles_to_graph = smiles_to_bigraph,
                                       node_featurizer = data_args['node_featurizer'],
                                       edge_featurizer = data_args['edge_featurizer'],
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

data_args['n_tasks'] = ames_train_dataset.n_tasks
if data_args['split'] == 'random':
    splitter = RandomSplitter()
elif data_args['split'] == 'scaffold':
    splitter = ScaffoldSplitter()

fold_dataset = splitter.k_fold_split(ames_train_dataset, k = data_args['fold'])

model_args['in_node_feats'] = data_args['node_featurizer'].feat_size()
model_args['n_tasks'] = data_args['n_tasks']

# 5-fold CV
all_train_fold = []
all_val_fold = []
all_loss_fold = []
all_test_score = []
for fold_id in range(data_args['fold']):
    train_loader = DataLoader(fold_dataset[fold_id][0],
                              batch_size=model_args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs,
                              num_workers=data_args['num_workers'])
    val_loader = DataLoader(fold_dataset[fold_id][1],
                              batch_size=model_args['batch_size'],
                              shuffle=True,
                              collate_fn=collate_molgraphs,
                              num_workers=data_args['num_workers'])
    model_save_dir = model_args['cv_result_path'] + '/fold_' + str(fold_id + 1)
    mkdir_p(model_save_dir)
    fold_train_score = []
    fold_val_score = []
    fold_test_score = []
    fold_train_loss = []
    for repeat_id in range(data_args['repeat']):
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
        optimizer = Adam(model.parameters(), lr=model_args['lr'], weight_decay=model_args['weight_decay'])
        # stopper = EarlyStopping(patience=model_args['patience'],
        stopper = EarlyStopping(patience=model_args['patience'],
                                filename=model_save_dir+'/model.pth',
                                metric=model_args['metric'])
        repeat_train_score = []
        repeat_val_score = []
        repeat_train_loss = []
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
            # print('fold {:d}/{:d}, epoch {:d}/{:d}, training {} {:.4f}, training loss {:.4f}'.format(fold_id + 1, len(fold_dataset),
            #                                                                                           epoch + 1, model_args['num_epochs'],
            #                                                                                           model_args['metric'], train_score,
            #                                                                                           loss.item()))
            repeat_train_score.append(train_score)
            repeat_train_loss.append(loss.item())
            # Validation and early stop
            # val_score = run_an_eval_epoch(args, model, val_loader)
            model.eval()
            eval_meter = Metrics()
            with torch.no_grad():
                for batch_id, batch_data in enumerate(val_loader):
                    smiles, bg, labels, masks = batch_data
                    labels = labels.to(model_args['device'])
                    logits = predict(model_args, model, bg)
                    eval_meter.update(logits, labels, masks)
            val_score = np.mean(eval_meter.compute_metric(model_args['metric']))
            early_stop = stopper.cv_step(val_score, model)
            repeat_val_score.append(val_score)
            print('fold {:d}/{:d}, repeat {:d}/{:d}, epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(fold_id + 1, data_args['fold'],
                                                                                                                              repeat_id + 1, data_args['repeat'],
                                                                                                                              epoch + 1, model_args['num_epochs'],
                                                                                                                              model_args['metric'], val_score,
                                                                                                                              model_args['metric'], max(repeat_val_score)))
            if early_stop:
                break
            model.load_state_dict(torch.load(model_save_dir+'/model.pth')['model_state_dict'])
            model.eval()
            eval_meter = Metrics()
            with torch.no_grad():
                for batch_id, batch_data in enumerate(test_loader):
                    smiles, bg, labels, masks = batch_data
                    labels = labels.to(model_args['device'])
                    logits = predict(model_args, model, bg)
                    eval_meter.update(logits, labels, masks)
            test_score = np.mean(eval_meter.compute_metric(model_args['metric']))
            print('fold {:d}/{:d}, repeat {:d}/{:d}, best validation {} {:.4f}, test {} {:.4f}'.format(fold_id + 1, data_args['fold'],
                                                                                                       repeat_id + 1, data_args['repeat'],
                                                                                                       model_args['metric'], max(repeat_val_score),
                                                                                                       model_args['metric'], test_score))
        fold_train_loss.append(repeat_train_loss)
        fold_train_score.append(repeat_train_score)
        fold_val_score.append(repeat_val_score)
        fold_test_score.append(test_score)
    np.save('%s/train_loss.npy' % (model_save_dir),  np.array(fold_train_loss, dtype=object))
    np.save('%s/train_auc.npy' % (model_save_dir),  np.array(fold_train_score, dtype=object))
    np.save('%s/val_auc.npy' % (model_save_dir),  np.array(fold_val_score, dtype=object))
    np.save('%s/test_auc.npy' % (model_save_dir),  np.array(fold_test_score, dtype=object))
    all_loss_fold.append(fold_train_loss)
    all_train_fold.append(fold_train_score)
    all_val_fold.append(fold_val_score)
    all_test_score.append(fold_test_score)

np.save('%s/all_train_loss.npy' % (model_args['cv_result_path']),  np.array(all_loss_fold, dtype=object))
np.save('%s/all_train_auc.npy' % (model_args['cv_result_path']),  np.array(all_train_fold, dtype=object))
np.save('%s/all_val_auc.npy' % (model_args['cv_result_path']),  np.array(all_val_fold, dtype=object))
np.save('%s/all_test_auc.npy' % (model_args['cv_result_path']),  np.array(all_test_score, dtype=object))
