import os
import json
#from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math, copy, time
from torch.autograd import Variable
from scipy import stats
import pandas as pd 
from sklearn.model_selection import KFold
import pickle
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import os.path
# from Bio import SeqIO
import string
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from APEX_models import AMP_model
#from propy.AAComposition import CalculateAADipeptideComposition
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy import stats
from utils import *
from scipy import sparse
import sys
from optparse import OptionParser
import copy
from sklearn.metrics import r2_score


max_len = 52 # maximun peptide length

word2idx, idx2word = make_vocab()
#emb = PC6('./fpScales.csv', word2idx)
#emb = FP_scale('./fpScales.csv', word2idx)
emb, AAindex_dict = AAindex('aaindex1.csv', word2idx)  # emb 从这来的，应该有这个就够了
vocab_size = len(word2idx)
emb_size = np.shape(emb)[1]


def load_inhouse(path):

    f = open(path, 'rb')
    inhouse_pkl = pickle.load(f)
    f.close()

    Seq_val, _, MIC_val = inhouse_pkl['val_set']
    Seq_tune, _, MIC_tune = inhouse_pkl['tune_set']

    print ('Tune data size', len(Seq_tune), np.shape(MIC_tune))
    print ('Val data size', len(Seq_val), np.shape(MIC_val))


    MIC_tune[np.where(np.isinf(MIC_tune))] = 512
    MIC_val[np.where(np.isinf(MIC_val))] = 512


    Y_mask_tune = np.ones((np.shape(MIC_tune)[0], np.shape(MIC_tune)[1]))
    Y_mask_tune[np.where(MIC_tune == -1000.0)] = 0
    MIC_tune[np.where(MIC_tune==-1000.0)] = 1000
    MIC_tune =  -np.log10(MIC_tune/float(1000000))


    Y_mask_val = np.ones((np.shape(MIC_val)[0], np.shape(MIC_val)[1]))
    Y_mask_val[np.where(MIC_val == -1000.0)] = 0
    MIC_val[np.where(MIC_val==-1000.0)] = 1000
    MIC_val =  -np.log10(MIC_val/float(1000000))

    num_p = np.shape(MIC_tune)[1]
    KNN_o, KNN_median = inhouse_pkl['knn']

    KFold_list = inhouse_pkl['kfold_list']
    eval_list = inhouse_pkl['eval_list']


    return Seq_tune, MIC_tune, Y_mask_tune, Seq_val, MIC_val, Y_mask_val, num_p, KNN_o, KNN_median, inhouse_pkl


Seq_tune, MIC_tune, Y_mask_tune, Seq_val, MIC_val, Y_mask_val, num_p, KNN_o, KNN_median, inhouse_pkl = load_inhouse('inhouse_pathogen_pkl')

"""
def load_DBAASP(path, inhouse_seq):
    inhouse_set = set()
    for i in inhouse_seq:
        inhouse_set.add(i)

    f = open(path, 'rb')
    DBAASP_pkl = pickle.load(f)
    f.close()

    X, Y = DBAASP_pkl
    X_trim = []
    Y_trim = []
    for i in range(len(X)):
        if X[i] in inhouse_set:
            #print ('Exclude redundant peptide')
            continue
        X_trim.append(X[i])
        Y_trim.append(Y[i])

    return X_trim, Y_trim

DBAASP_X, DBAASP_Y = load_DBAASP('./DBAASP.pkl', Seq_tune+Seq_val)
Y_DBAASP = np.array(DBAASP_Y).reshape(-1, 1)
X_DBAASP = onehot_encoding(DBAASP_X, max_len, word2idx) 

"""


X_tune = onehot_encoding(Seq_tune, max_len, word2idx)
X_val = onehot_encoding(Seq_val, max_len, word2idx)



def load_public_AMPs(path_pos, path_neg, Seq_tune, Seq_val):
    public_AMPs = []
    public_nonAMPs = []

    f = open(path_pos)
    lines = f.readlines()
    f.close()
    for line in lines:
        parsed = line.strip('\n').strip('\r').split()
        if parsed[-1] not in Seq_tune and parsed[-1] not in Seq_val:
            public_AMPs.append(parsed[-1])

    f = open(path_neg)
    lines = f.readlines()
    f.close()
    for line in lines:
        parsed = line.strip('\n').strip('\r')
        if parsed not in Seq_tune and parsed not in Seq_val:
            public_nonAMPs.append(parsed)

    return public_AMPs, public_nonAMPs


public_AMPs, public_nonAMPs = load_public_AMPs('./public_positive_AMP_data.tsv',
    './public_negative_AMP_data.tsv', Seq_tune, Seq_val)


X_public_pos = onehot_encoding(public_AMPs, max_len, word2idx) 
X_public_neg = onehot_encoding(public_nonAMPs, max_len, word2idx) 
X_DBAASP = np.vstack([X_public_pos, X_public_neg])
Y_DBAASP = np.vstack([np.ones(len(X_public_pos)).reshape(-1, 1), np.zeros(len(X_public_neg)).reshape(-1, 1)])


#print (np.shape(X_tune), np.shape(MIC_tune), np.shape(Y_mask_tune))
#print (np.shape(X_val), np.shape(MIC_val), np.shape(Y_mask_val))


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


def init_weights(m):
    try:
        torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
    except:
        pass
 

def train_model(train_data, test_data, val_data, knn, num_rnn_layers, dim_h, dim_latent, num_fc_layers, l2_reg, multiask_reg, dbaasp_reg, batch_size=128, num_epoch=1000):

    model = AMP_model(emb, emb_size, num_rnn_layers, dim_h, dim_latent, num_fc_layers, num_p)
    # Applying it to our net
    #model.apply(init_weights)
    model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=l2_reg, amsgrad=False)
    #optimizer = optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-6) 

    min_loss = 1000000000
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    X_train, Y_train, Y_train_mask = train_data

    train_len = len(X_train)
    shuffle_index = np.arange(train_len)
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)
    np.random.shuffle(shuffle_index)

    DBAASP_len = len(X_DBAASP)
    DBAASP_shuffle_index = np.arange(DBAASP_len)
    np.random.shuffle(DBAASP_shuffle_index)
    np.random.shuffle(DBAASP_shuffle_index)
    np.random.shuffle(DBAASP_shuffle_index)
    np.random.shuffle(DBAASP_shuffle_index)
    np.random.shuffle(DBAASP_shuffle_index)



    BCEloss = nn.BCEWithLogitsLoss()

    knn = torch.FloatTensor(knn).cuda()

    for epoch in range(num_epoch):  
        #np.random.seed(100)
        model.train()
        np.random.shuffle(shuffle_index)
        np.random.shuffle(DBAASP_shuffle_index)

        for i in range(int(train_len/batch_size)):
            X_batch = X_train[shuffle_index[i*batch_size:(i+1)*batch_size]]
            X_batch = torch.LongTensor(X_batch).cuda()
            label = torch.FloatTensor(Y_train[shuffle_index[i*batch_size:(i+1)*batch_size]]).cuda()
            mask = torch.FloatTensor(Y_train_mask[shuffle_index[i*batch_size:(i+1)*batch_size]]).cuda()

            X_DBAASP_batch = X_DBAASP[DBAASP_shuffle_index[i*batch_size:(i+1)*batch_size]]
            X_DBAASP_batch = torch.LongTensor(X_DBAASP_batch).cuda()
            Y_DBAASP_batch = torch.FloatTensor(Y_DBAASP[DBAASP_shuffle_index[i*batch_size:(i+1)*batch_size]]).cuda()


            optimizer.zero_grad()
            Y_pred = model(X_batch)

            Y_pred_DBAASP = model.clf_forward(X_DBAASP_batch)
            loss_clf = BCEloss(Y_pred_DBAASP, Y_DBAASP_batch)

            loss = torch.mean(torch.sum(mask * (label - Y_pred)**2, axis=0) / (torch.sum(mask, axis=0)+1e-12))
            #loss = torch.sum(mask * (label - Y_pred)**2) / torch.sum(mask)


            llw = model.fc4.weight#.unsqueeze(0)
            pdist = 1 - sim_matrix(llw, llw, eps=1e-12)#torch.cdist(llw, llw).squeeze(-1)
            mcl = torch.sum(pdist * knn) / float(2.0)
            #print (mcl)
            loss += float(multiask_reg) * mcl

            loss += float(dbaasp_reg) * loss_clf

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()

        if (epoch+1) % 100 == 0:


            train_all_r2, train_all_p, train_all_sp, train_wise_r2, train_wise_p, train_wise_sp, _ = test_model(model, train_data)
            test_all_r2, test_all_p, test_all_sp, test_wise_r2, test_wise_p, test_wise_sp, test_data_pack = test_model(model, test_data)
            val_all_r2, val_all_p, val_all_sp, val_wise_r2, val_wise_p, val_wise_sp, val_data_pack = test_model(model, val_data)

            print ('Epoch', epoch+1)
            print ('train, test and val all r2', train_all_r2, test_all_r2, val_all_r2)
            print ('train, test and val all pearsons', train_all_p, test_all_p, val_all_p)
            print ('train, test and val all spearmans', train_all_sp, test_all_sp, val_all_sp)
            print ('train, test and val avg r2', np.mean(train_wise_r2), np.mean(test_wise_r2), np.mean(val_wise_r2))
            print ('train, test and val avg pearsons', np.mean(train_wise_p), np.mean(test_wise_p), np.mean(val_wise_p))
            print ('train, test and val avg spearmans', np.mean(train_wise_sp), np.mean(test_wise_sp), np.mean(val_wise_sp))
            print ('====================')

    return [test_all_p, test_all_sp, test_wise_p, test_wise_sp, test_data_pack], [val_all_p, val_all_sp, val_wise_p, val_wise_sp, val_data_pack], model


# def test_model(model, data, batch_size=128):
#     model.eval()
#     output_list = []
#     label_list = []
#
#     X_, Y_, Y_mask_ = data
#
#     data_len = len(X_)
#
#     X_matrix = []
#     Y_matrix = []
#     Y_mask_matrix = []
#     Y_pred_matrix = []
#
#     for i in range(int(math.ceil(data_len/float(batch_size)))):
#
#         X_batch = X_[i*batch_size:(i+1)*batch_size]
#         Y_batch = Y_[i*batch_size:(i+1)*batch_size]
#         Y_mask_batch = Y_mask_[i*batch_size:(i+1)*batch_size]
#
#         X_batch = torch.LongTensor(X_batch).cuda()
#         Y_pred_batch = model(X_batch)
#
#         Y_pred_batch = Y_pred_batch.cpu().detach().numpy()
#
#         if i == 0:
#             #X_matrix = X_seq.cpu().detach().numpy()
#             Y_matrix = Y_batch
#             Y_mask_matrix = Y_mask_batch
#             Y_pred_matrix = Y_pred_batch
#         else:
#             #X_matrix = np.vstack([X_matrix, X_seq.cpu().detach().numpy()])
#             Y_matrix = np.vstack([Y_matrix, Y_batch])
#             Y_mask_matrix = np.vstack([Y_mask_matrix, Y_mask_batch])
#             Y_pred_matrix = np.vstack([Y_pred_matrix, Y_pred_batch])
#
#     kept_list = []
#     reverse_list = []
#     for i in inhouse_pkl['eval_list']:
#         kept_list.append(i[1])
#         reverse_list.append(i[0])
#
#     Y_matrix = Y_matrix[:, kept_list]
#     Y_mask_matrix = Y_mask_matrix[:, kept_list]
#     Y_pred_matrix = Y_pred_matrix[:, kept_list]
#
#     label_list = np.array(Y_matrix[np.where(Y_mask_matrix==1)]).reshape(-1)
#     output_list = np.array(Y_pred_matrix[np.where(Y_mask_matrix==1)]).reshape(-1)
#
#     all_r2 = r2_score(label_list, output_list)
#     all_p = stats.pearsonr(label_list, output_list)[0]
#     all_sp = stats.spearmanr(label_list, output_list)[0]
#
#
#     counter = 0
#
#     bacterium_wise_r2 = []
#     bacterium_wise_p = []
#     bacterium_wise_sp = []
#     for i in range(np.shape(Y_matrix)[1]):
#         mask_col = Y_mask_matrix[:, i]
#         label_col = Y_matrix[:, i]
#         pred_col = Y_pred_matrix[:, i]
#         label_list = label_col[np.where(mask_col==1)].reshape(-1)
#         output_list = pred_col[np.where(mask_col==1)].reshape(-1)
#         a_r2 = r2_score(label_list, output_list)
#         a_p = stats.pearsonr(label_list, output_list)[0]
#         a_sp = stats.spearmanr(label_list, output_list)[0]
#         bacterium_wise_r2.append(a_r2)
#         bacterium_wise_p.append(a_p)
#         bacterium_wise_sp.append(a_sp)
#         #print (reverse_list[counter], a_r2)
#         counter += 1
#
#     #return p, sp, [X_matrix, Y_matrix, Y_mask_matrix, Y_pred_matrix]
#     return all_r2, all_p, all_sp, bacterium_wise_r2, bacterium_wise_p, bacterium_wise_sp, [Y_matrix, Y_mask_matrix, Y_pred_matrix]



def CV(tune_data, val_data, knn, inhouse_pkl, num_rnn_layers, dim_h, dim_latent, num_fc_layers, l2_reg, multiask_reg, dbaasp_reg, rep_num=1):

    X, Y, Y_mask = tune_data

    all_pearson_list = []
    all_spearman_list = []
    avg_pearson_list = []
    avg_spearman_list = []
    data_pack = []
    for a_rep in range(rep_num):
        KFold_list = inhouse_pkl['kfold_list'][rep_num+18] #I randomly choose 18 as a shift.
        fold_counter = 0
        for a_split in KFold_list:
            fold_counter += 1
            train_index, test_index = a_split

            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            Y_train_mask, Y_test_mask = Y_mask[train_index], Y_mask[test_index]


            
            
            test_results, val_results, model = train_model([X_train, Y_train, Y_train_mask], [X_test, Y_test, Y_test_mask], val_data, knn, 
                                                            num_rnn_layers, dim_h, dim_latent, num_fc_layers, l2_reg, multiask_reg, dbaasp_reg,
                                                            batch_size=128, num_epoch=5000)
            all_pearson_list.append(test_results[0])
            all_spearman_list.append(test_results[1])
            avg_pearson_list.append(np.mean(test_results[2]))
            avg_spearman_list.append(np.mean(test_results[3]))
            data_pack.append(test_results[4])



    #print ('All Pearson:', np.mean(all_pearson_list))
    #print ('All Spearman:', np.mean(all_spearman_list))

    #print ('Avg Pearson:', np.mean(avg_pearson_list))
    #print ('Avg Spearman:', np.mean(avg_spearman_list))



    return all_pearson_list, all_spearman_list, avg_pearson_list, avg_spearman_list, data_pack


#all_pearson_list, all_spearman_list, avg_pearson_list, avg_spearman_list, data_pack = \
#                        CV(tune_data=[X_tune, MIC_tune, Y_mask_tune], val_data=[X_val, MIC_val, Y_mask_val], 
#                        knn=KNN_median, inhouse_pkl=inhouse_pkl, num_rnn_layers=3, 
#                        dim_h=256, dim_latent=1024, num_fc_layers=3, l2_reg=1e-6, multiask_reg=0.001, dbaasp_reg=0.1, rep_num=1)



parser = OptionParser()
parser.add_option("-n","--n", default=3, help="num_rnn_layers")
parser.add_option("-i","--i", default=128, help="rnn_dim")
#parser.add_option("-d","--d", default=512, help="dim_latent")
#parser.add_option("-r","--r", default=1e-6, help="l2_reg")
parser.add_option("-m","--m", default=1.0, help="a_dbaasp_reg")
#parser.add_option("-b","--b", default=1.0, help="a_multiask_reg")


(opts, args) = parser.parse_args()

a_dbaasp_reg = float(opts.m)
#a_l2_reg = float(opts.r)
#a_dim_latent = int(opts.d)
a_dim_h = int(opts.i)
a_num_rnn_layers = int(opts.n)
#a_multiask_reg = float(opts.b)

#for multiask_reg in [0.0, 0.001, 0.01, 0.1]:
#for dbaasp_reg in [1.0, 0.1]:



for num_rnn_layers in [a_num_rnn_layers]:
    for dim_h in [a_dim_h]:
        for dim_latent in [256, 512, 1024, 2048]:
            for l2_reg in [1e-5, 1e-6]:
                for multiask_reg in [0.01, 0.001, 0.0]:
                    for dbaasp_reg in [a_dbaasp_reg]:


                        all_pearson_list, all_spearman_list, avg_pearson_list, avg_spearman_list, data_pack = \
                            CV(tune_data=[X_tune, MIC_tune, Y_mask_tune], val_data=[X_val, MIC_val, Y_mask_val], 
                            knn=KNN_median, inhouse_pkl=inhouse_pkl, num_rnn_layers=num_rnn_layers, 
                            dim_h=dim_h, dim_latent=dim_latent, num_fc_layers=3, l2_reg=l2_reg, multiask_reg=multiask_reg, dbaasp_reg=dbaasp_reg, rep_num=1)
                        
                        hyperparameter_dict = {}

                        print ('Hyperparameter:', num_rnn_layers, dim_h, dim_latent, l2_reg, multiask_reg, dbaasp_reg)
                        print ('All Pearson:', np.mean(all_pearson_list))
                        print ('All Spearman:', np.mean(all_spearman_list))
                        print ('Avg Pearson:', np.mean(avg_pearson_list))
                        print ('Avg Spearman:', np.mean(avg_spearman_list))
                        print ('------------------------------------')

                        key_name = str(num_rnn_layers)+'&'+str(dim_h)+'&'+str(dim_latent)+'&'+str(l2_reg)+'&'+str(multiask_reg)+'&'+str(dbaasp_reg)
                        hyperparameter_dict[key_name] = [all_pearson_list, all_spearman_list, avg_pearson_list, avg_spearman_list, data_pack]

                        f = open('./cv_tmp/hyperparameter_search_on_cv_'+'_multiask_reg_'+str(multiask_reg)+'_dbaasp_reg_'+str(dbaasp_reg)+'_l2_'+str(l2_reg)+'_latent_'+str(dim_latent)+'_input_'+str(dim_h)+'_rnn_'+str(num_rnn_layers), 'wb')
                        pickle.dump(hyperparameter_dict, f)
                        f.close()



#ensemble_num = 32
#for ensemble_id in range(ensemble_num):
#    _,_, model,_ = train_model(X, Y, Y_mask, X, Y, Y_mask, laplacian, batch_size=128, num_epoch=3000)
#    torch.save(model, './pp_20211203/trained_AMP_model_'+str(ensemble_id))
