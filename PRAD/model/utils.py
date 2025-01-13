import sys
import os
import torch
import random
import math

from sklearn.utils import shuffle
from sklearn.metrics import f1_score

import torch.nn as nn
import numpy as np
import torch.autograd as autograd
torch.autograd.set_detect_anomaly(True)

def get_gnn_embeddings(gnn_model, dataCenter, ds):
    print('Loading embeddings from trained GraphSAGE model.')
    features = np.zeros((len(getattr(dataCenter, ds + '_labels')), gnn_model.out_size))
    nodes = np.arange(len(getattr(dataCenter, ds + '_labels'))).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)
    embs = []
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)
        # if ((index+1)*b_sz) % 10000 == 0:
        #     print(f'Dealed Nodes [{(index+1)*b_sz}/{len(nodes)}]')

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    print('Embeddings loaded.')
    return embs.detach()

def apply_model(dataCenter, ds, graphSage, unsupervised_loss, b_sz, unsup_loss, device, learn_method):
    train_nodes = getattr(dataCenter, ds + '_train')
    labels = getattr(dataCenter, ds + '_labels')
    feat_data = getattr(dataCenter, ds + '_feats')

    if unsup_loss == 'margin':
        num_neg = 6
    elif unsup_loss == 'normal':
        num_neg = 100
    else:
        print("unsup_loss can be only 'margin' or 'normal'.")
        sys.exit(1)

    train_nodes = shuffle(train_nodes)
    models = [graphSage]
    params = []
    for model in models:
        for param in model.parameters():
            if param.requires_grad:
                params.append(param)

    optimizer = torch.optim.SGD(params, lr=0.001, weight_decay=0.001)
    loss_fn = nn.MSELoss()

    batches = math.ceil(len(train_nodes) / b_sz)
    visited_nodes = set()
    embed_result = [[] for l in range(len(labels))]
    loss_result = [[] for l in range(batches)]
    for index in range(batches):
        nodes_batch = train_nodes[index * b_sz:(index + 1) * b_sz]  # batch训练的节点
        # extend nodes batch for unspervised learning
        # no conflicts with supervised learning
        nodes_batch = np.asarray(list(unsupervised_loss.extend_nodes(nodes_batch, num_neg=num_neg)))
        visited_nodes |= set(nodes_batch)
        visited_labels = labels[list(visited_nodes)]
        batch_label = labels[list(nodes_batch)]

        true_batch = 0
        loss_total = 0
        if sum(visited_labels) != 0:
            # feed nodes batch to the graphSAGE
            # returning the nodes embeddings。 得到GraphSAGE后的ebmedding向量
            embs_batch = graphSage(nodes_batch)  # 跳到models的GraphSge

            # the loss between true data and emb_data
            true_feature = torch.FloatTensor(feat_data[list(nodes_batch)][np.where(batch_label == 1)]).to(device)
            embed_feature = embs_batch[np.where(batch_label == 1)].to(device)
            BN_t = nn.BatchNorm1d(true_feature.shape[1]).to(device)
            BN_e = nn.BatchNorm1d(embed_feature.shape[1]).to(device)
            true_feature_bn = BN_t(true_feature)
            embed_feature_bn = BN_e(embed_feature)

            RMSE_loss = torch.abs(loss_fn(true_feature_bn, embed_feature_bn)+1e-8)

            print('Step: {}/{}, RMSE_loss:{}'.format(index+1, batches, RMSE_loss.item()))

            optimizer.zero_grad()
            RMSE_loss.backward(retain_graph=True)
            for model in models:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=3, norm_type='inf')  # 梯度的二范数和不超过5（平方和开根号）
            optimizer.step()

            true_batch += 1
            loss_total += RMSE_loss.item()
            loss_mean = loss_total / true_batch
            true_node = np.array(list(nodes_batch))[np.where(batch_label == 1)]
            loss_result[index] = RMSE_loss.item()
            for n,v in enumerate(np.array(list(nodes_batch))):
                embed_result[v] = embs_batch[n]
            for t_n, t_v in enumerate(true_node):
                embed_result[t_v] = true_feature[t_n]

        continue

    return graphSage, loss_result, embed_result

def valid_model(dataCenter, ds, graphSage, device, learn_method):
    valid_node = getattr(dataCenter, ds + '_valid')
    labels = getattr(dataCenter, ds + '_labels')
    feat_data = getattr(dataCenter, ds + '_feats')
    loss_fn = nn.MSELoss()
    val_label = labels[list(valid_node)]
    val_embs = graphSage(valid_node)
    true_feature = torch.FloatTensor(feat_data[list(valid_node)][np.where(val_label == 1)]).to(device)
    true_embs = val_embs[np.where(val_label == 1)].to(device)

    BN_t = nn.BatchNorm1d(true_feature.shape[1]).to(device)
    BN_e = nn.BatchNorm1d(true_embs.shape[1]).to(device)
    true_feature_bn = BN_t(true_feature)
    embed_feature_bn = BN_e(true_embs)

    RMSE_loss = torch.abs(loss_fn(true_feature_bn, embed_feature_bn))
    print('RMSE_loss:{}'.format(RMSE_loss.item()))

    loss_result = RMSE_loss.item()

    return loss_result, true_embs
