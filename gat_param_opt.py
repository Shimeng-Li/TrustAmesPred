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
from hyperopt import hp, tpe, fmin, Trials
import json

data_args = {'train_file': '../dataset/ames/data_v2/train_baseline.csv',
             'ext_file': '../dataset/ames/data_v2/external_baseline.csv',
             'train_cache_file_path': './data_transform/train_baseline.bin',
             'test_cache_file_path': './data_transform/external_baseline.bin',
             'task_name': ['ames'],
             'num_workers': 20,
             "batch_size": 512,
             'random_state': 0,
             'node_featurizer': AtomFeaturizer(),
             'edge_featurizer': AttentiveFPBondFeaturizer(),
             'load': False}

model_args = {'metric': 'roc_auc_score',
              'result_path': './gat_result/param_opt/param_opt_v3/test_opt',
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

mkdir_p(model_args['result_path'])

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

data_args['n_tasks'] = ames_train_dataset.n_tasks
splitter = RandomSplitter()

def val_opt(param_opt):
    try:
        random_state, batch_size, alpha, feat_dropout, attn_dropout, predictor_dropout, gnn_hidden_feats,\
        predictor_hidden_feats, num_gnn_layers, num_heads, lr, weight_decay = param_opt
        print(param_opt)
        train_set, val_set = splitter.train_val_split(ames_train_dataset,
                                                      frac_train = 0.8,
                                                      frac_val = 0.2,
                                                      random_state=random_state)
        model_args['in_node_feats'] = data_args['node_featurizer'].feat_size()
        model_args['n_tasks'] = data_args['n_tasks']
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_molgraphs,
                                  num_workers=data_args['num_workers'])
        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_molgraphs,
                                num_workers=data_args['num_workers'])
        model = gat_classifier(in_feats=model_args['in_node_feats'],
                               hidden_feats=[gnn_hidden_feats] * num_gnn_layers,
                               num_heads=[num_heads] * num_gnn_layers,
                               feat_drops=[feat_dropout] * num_gnn_layers,
                               attn_drops=[attn_dropout] * num_gnn_layers,
                               alphas=[alpha] * num_gnn_layers,
                               residuals=[model_args['residual']] * num_gnn_layers,
                               predictor_hidden_feats=predictor_hidden_feats,
                               predictor_dropout=predictor_dropout,
                               n_tasks=model_args['n_tasks'])
        model.to(model_args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        stopper = EarlyStopping(patience=model_args['patience'],
                                filename=model_args['result_path']+'/model.pth',
                                metric=model_args['metric'])
        fold_train_score = []
        fold_val_score = []
        fold_train_loss = []
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
            fold_train_score.append(train_score)
            fold_train_loss.append(loss.item())
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
            fold_val_score.append(val_score)
            print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(epoch + 1, model_args['num_epochs'],
                                                                                            model_args['metric'], val_score,
                                                                                            model_args['metric'], max(fold_val_score)))
            if early_stop:
                break
    except:
        print('Error Occurred')
    return -max(fold_val_score)


test_score = 0
def test_opt(param_opt):
    try:
        random_state, batch_size, alpha, feat_dropout, attn_dropout, predictor_dropout, gnn_hidden_feats,\
        predictor_hidden_feats, num_gnn_layers, num_heads, lr, weight_decay = param_opt
        print(param_opt)
        train_set, val_set = splitter.train_val_split(ames_train_dataset,
                                                      frac_train = 0.8,
                                                      frac_val = 0.2,
                                                      random_state=random_state)
        model_args['in_node_feats'] = data_args['node_featurizer'].feat_size()
        model_args['n_tasks'] = data_args['n_tasks']
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=collate_molgraphs,
                                  num_workers=data_args['num_workers'])
        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=True,
                                collate_fn=collate_molgraphs,
                                num_workers=data_args['num_workers'])
        test_loader = DataLoader(ames_test_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 collate_fn=collate_molgraphs,
                                 num_workers=data_args['num_workers'])
        model = gat_classifier(in_feats=model_args['in_node_feats'],
                               hidden_feats=[gnn_hidden_feats] * num_gnn_layers,
                               num_heads=[num_heads] * num_gnn_layers,
                               feat_drops=[feat_dropout] * num_gnn_layers,
                               attn_drops=[attn_dropout] * num_gnn_layers,
                               alphas=[alpha] * num_gnn_layers,
                               residuals=[model_args['residual']] * num_gnn_layers,
                               predictor_hidden_feats=predictor_hidden_feats,
                               predictor_dropout=predictor_dropout,
                               n_tasks=model_args['n_tasks'])
        model.to(model_args['device'])
        loss_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        stopper = EarlyStopping(patience=model_args['patience'],
                                filename=model_args['result_path']+'/model.pth',
                                metric=model_args['metric'])
        train_score_list = []
        val_score_list = []
        train_loss_list = []
        global test_score
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
            model.load_state_dict(torch.load(model_args['result_path'] + '/model.pth')['model_state_dict'])
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
            print('epoch {:d}/{:d}, validation {} {:.4f}, best validation {} {:.4f}'.format(epoch + 1, model_args['num_epochs'],
                                                                                            model_args['metric'], val_score,
                                                                                            model_args['metric'], max(val_score_list)))
            if early_stop:
                break
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
    except:
        print('Error Occurred')
    return -test_score

random_state_list = [seed for seed in range(1000)]
batch_size_list = [128, 256, 512, 1024]
gnn_hidden_feats_list = [32, 64, 128, 256]
predictor_hidden_feats_list = [32, 64, 128, 256]
num_gnn_layers_list = [layer_num + 1 for layer_num in range(10)]
num_heads_list = [(heads_num+1) * 2 for heads_num in range(10)]
param_space = [hp.choice('random_state', random_state_list),
               hp.choice('batch_size', batch_size_list),
               hp.uniform('alpha', 0, 1),
               hp.uniform('feat_dropout', 0, 1),
               hp.uniform('attn_dropout', 0, 1),
               hp.uniform('predictor_dropout', 0, 1),
               hp.choice('gnn_hidden_feats', gnn_hidden_feats_list),
               hp.choice('predictor_hidden_feats', predictor_hidden_feats_list),
               hp.choice('num_gnn_layers', num_gnn_layers_list),
               hp.choice('num_heads', num_heads_list),
               hp.loguniform('lr', -10, 0),
               hp.loguniform('weight_decay', -10, 0)]

# import hyperopt.pyll.stochastic
# print(hyperopt.pyll.stochastic.sample(param_space))
trials = Trials()
best = fmin(fn = test_opt,
            space = param_space,
            algo = tpe.suggest,
            max_evals = 200,
            trials = trials)

print('best:', best)
# best_param_dict = json.dumps(best)
best_param_dict = best
best_param_dict['batch_size'] = batch_size_list[best['batch_size']]
best_param_dict['gnn_hidden_feats'] = gnn_hidden_feats_list[best['gnn_hidden_feats']]
best_param_dict['num_gnn_layers'] = num_gnn_layers_list[best['num_gnn_layers']]
best_param_dict['num_heads'] = num_heads_list[best['num_heads']]
best_param_dict['predictor_hidden_feats'] = predictor_hidden_feats_list[best['predictor_hidden_feats']]
best_param_dict['random_state'] = random_state_list[best['random_state']]

with open('%s/best_param.json' % model_args['result_path'], 'w') as f:
    json.dump(best_param_dict, f)

with open('%s/param_opt_record.txt' % model_args['result_path'], 'w') as f:
    for trial in trials:
        f.write(str(trial)+'\n')

import matplotlib.pyplot as plt

# Parameters Optimization (time to loss)
f, ax = plt.subplots(1)
xs = [t['tid'] for t in trials.trials]
ys = [-t['result']['loss'] for t in trials.trials]
ax.set_xlim(0, 200)
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$AUC$ $vs$ $param$ ', fontsize=18)
ax.set_xlabel('$param$', fontsize=16)
ax.set_ylabel('$AUC$', fontsize=16)
plt.savefig('%s/param_auc.tiff' % model_args['result_path'])
