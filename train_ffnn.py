import pandas as pd
import numpy as np
import re
import os

import csv
import sys

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.multiprocessing as mp

#scikit-learn modules
from sklearn import ensemble, linear_model, metrics
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import sklearn.model_selection

ko_matrix = pd.read_csv('taxid_ko_matrix_all_full.csv').drop('Unnamed: 0',1)
func_matrix = pd.read_csv('groups_to_records_all_full.txt',sep='\t', comment='#')

ko_matrix.columns = [re.sub('ko:','',i) for i in ko_matrix.columns]

#get lists of the KOs and functions for later use
ko_list = [i for i in list(ko_matrix.columns) if i.startswith('K')]
func_list = func_matrix.drop('record',1).columns



#merge the two files together based on the column 'species_taxid'#merge
merged = pd.merge(ko_matrix, func_matrix, left_on='species_taxid', right_on='record')

merged.head()

#function = 'manganese_oxidation'

dict_lr = {}
dict_nn = {}
dict_rf = {}
dict_svm = {}



for function in func_list:

    if function not in list(used_functions):

        print(function)

        kos = merged[ko_list]
        to_predict = merged[function]

        kos = np.array(kos,dtype=np.float64)
        sc = StandardScaler()
        k_fold = StratifiedKFold(n_splits=5)

        # Run Linear Regression
        clf_lr = linear_model.LogisticRegression(penalty='l1', solver='liblinear',C=0.05,
                                              class_weight=None)

        # try is used in case there is no presence of a function in the test set for any fold
        try:
            roc_auc_lr = np.mean([metrics.roc_auc_score(to_predict[test],clf_lr.fit(sc.fit_transform(kos[train]), to_predict[train]).predict(sc.fit_transform(kos[test])))
                for train, test in k_fold.split(kos,to_predict)])

        except:
            print ("Exception: ", sys.exc_info()[0])



        print('Test AUROC lr: ', roc_auc_lr)
        dict_lr[function] = roc_auc_lr
        print("Lr mean AUROC: ", np.mean(list(dict_lr.values())))
        dict_lr_df = pd.DataFrame(list(dict_lr.items()), columns=['function', 'Mean 5Fold AUROC'])
        dict_lr_df.to_csv('lr_auroc.csv',index=False)

        # Run Random Forest
        clf_rf = ensemble.RandomForestClassifier()

        try:
            roc_auc_rf = np.mean([metrics.roc_auc_score(to_predict[test],clf_rf.fit(sc.fit_transform(kos[train]), to_predict[train]).predict(sc.fit_transform(kos[test])))
                for train, test in k_fold.split(kos,to_predict)])

        except :
            print ("Exception: ", sys.exc_info()[0])



        print('Test AUROC rf: ', roc_auc_rf)
        dict_rf[function]=roc_auc_rf
        print("Rf mean AUROC: ", np.mean(list(dict_rf.values())))
        dict_rf_df = pd.DataFrame(list(dict_rf.items()), columns=['function', 'Mean 5Fold AUROC'])
        dict_rf_df.to_csv('rf_auroc.csv',index=False)


        # Run SVM
        clf_svm = svm.SVC(kernel='linear', probability=True, C=.0005, class_weight=None)

        try:
            roc_auc_svm = np.mean([metrics.roc_auc_score(to_predict[test],clf_svm.fit(sc.fit_transform(kos[train]), to_predict[train]).predict(sc.fit_transform(kos[test])))
                for train, test in k_fold.split(kos,to_predict)])

        except ValueError:
            print("Error in function %s" % (function))


        print('Test AUROC svm: ', roc_auc_svm)
        dict_svm[function]=roc_auc_svm
        print("Svm mean AUROC: ", np.mean(list(dict_svm.values())))
        dict_svm_df = pd.DataFrame(list(dict_svm.items()), columns=['function', 'Mean 5Fold AUROC'])
        dict_svm_df.to_csv('svm_auroc.csv',index=False)


        ### Train a ffnn

        # Use GPU if present
        if torch.cuda.is_available():
          device = 'cuda:0'
        else:
          device = 'cpu'


        def heaviside(vec,cut_off):
            return np.array([1.0 if el >= cut_off else 0.0 for el in vec])

        class Net(nn.Module):

            def __init__(self,num_features):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(num_features, 500).to(device)
                self.prelu1 = nn.PReLU().to(device)
                self.dout1 = nn.Dropout(0.8).to(device)

                self.fc2 = nn.Linear(500, 250).to(device)
                self.prelu2 = nn.PReLU().to(device)
                self.dout2 = nn.Dropout(0.5).to(device)

                self.fc4 = nn.Linear(250, 100).to(device)
                self.prelu4= nn.PReLU().to(device)
                self.dout4 = nn.Dropout(0.25).to(device)

                self.out = nn.Linear(100, 1).to(device)
                self.out_act = nn.PReLU().to(device)

            def forward(self, input_):
                a1 = self.fc1(input_)
                h1 = self.prelu1(a1)
                dout1 = self.dout1(h1)

                a2 = self.fc2(dout1)
                h2 = self.prelu2(a2)
                dout2 = self.dout2(h2)

                a4 = self.fc4(dout2)
                h4 = self.prelu4(a4)
                dout4 = self.dout4(h4)

                a5 = self.out(dout4)
                y = self.out_act(a5)
                return y

        net = Net(kos.shape[1])


        t = 0
        roc_auc_nn = []
        try:
            for train, test in k_fold.split(kos,to_predict):

                X_train = torch.from_numpy(sc.fit_transform(kos[train])).float().to(device)
                y_train = torch.from_numpy(np.array(to_predict)[train]).to(device).unsqueeze(1)

                lr = 1e-3
                optimizer = torch.optim.Adam(net.parameters(), lr = lr)
                loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

                for t in range(100):
                    prediction = net(X_train)  # input x and predict based on x

                    loss = loss_func(prediction, y_train.float())  # must be (1. nn output, 2. target)

                    optimizer.zero_grad()  # clear gradients for next train
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                X_test = torch.from_numpy(sc.transform(kos[test])).float().to(device)  # torch.from_numpy(X_test).to(device)
                y_test = torch.from_numpy(np.array(to_predict)[test]).to(device).unsqueeze(1)

                test_probs = net(X_test).data.numpy()
                test_predictions = heaviside(test_probs, 0.5)


                roc_auc_nn.append(metrics.roc_auc_score(y_test, test_probs))

        except ValueError:
                print("Error in function %s" % (function))
                #unused_functions.append(function)
                # continue
        print('Test AUROC Nn: ', np.mean(roc_auc_nn))
        temp = []
        dict_nn[function]=np.mean(roc_auc_nn)
        print("Nn mean AUROC: ", np.mean(list(dict_nn.values())))
        dict_nn_df = pd.DataFrame(list(dict_nn.items()), columns=['function', 'Mean 5Fold AUROC'])
        dict_nn_df.to_csv('nn_auroc.csv',index=False)
