import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from functions.oversampling import *
from functions.classifiers import *
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--input_file', type=str, help='input file')
parser.add_argument('--oversampling', type=str, default='Baseline', help='[Baseline, SMOTE, SMOTEEN, ADASYN]')
parser.add_argument('--oversampling_threshold', type=int, default=2000, help='minimum no. of instances for each class')
parser.add_argument('--inner_layers', type=list, default=[100,200,100], help='inner layers of Bayesian NN')
parser.add_argument('--optimizer', type=str, default='ADAM', help='[ADAM, SGD]')
parser.add_argument('--kl_weight', type=float, default=0.1, help='kl_weight for KL Diversgence Loss for Bayesian NN')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--iterations', type=int, default=2000, help='no. of iterations')

args = parser.parse_args()

if __name__ == "__main__":
    
    data = pd.read_csv(args.input_file)
    data=data.dropna()
    
    # Standardize the data
    main_data, labels, columns = Standardize(data)
    main_data=pd.DataFrame(main_data,columns=columns)
    
    # Split the Data
    X_train,X_test,Y_train,Y_test = train_test_split(main_data, labels, stratify=labels, test_size=0.2)
    
    train_data=np.array(X_train)
    train_labels=np.array(Y_train)
    test_data=np.array(X_test)
    test_labels=np.array(Y_test)
    
    # The data needs to be converted into tensors so that the Bayesian Neural Network can consume them.
    x_train, y_train = torch.from_numpy(train_data).float(), torch.from_numpy(train_labels).long()
    x_test, y_test = torch.from_numpy(test_data).float(), torch.from_numpy(test_labels).long()
    layer_list=args.inner_layers
    layer_list.insert(0,18)
    layer_list.append(7)
    
    model=[]
    layer_num=1
    
    # Defining the structure of the Bayesian NN, see the args.inner_layers variable for more clarity.
    while(layer_num<len(layer_list)):
        
        model.append(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=layer_list[layer_num-1], out_features=layer_list[layer_num]))
        if(layer_num != len(layer_list)-1):
            model.append(nn.ReLU())
        layer_num = layer_num + 1
                    
    model=nn.Sequential(*model)
    
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
    
    if(args.optimizer == 'ADAM'):
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        
    kl_weight = args.kl_weight
    
    # Lets print the model to see how it looks like.
    print(model)
    
    if(args.oversampling == 'Baseline'):
        
        # Training of the Bayesian NN starts
        for step in tqdm(range(args.iterations)):
            pre = model(x_train)
            ce = ce_loss(pre, y_train)
            kl = kl_loss(model)
            cost = ce + kl_weight*kl
    
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        pre = model(x_test)
        _, predicted = torch.max(pre.data, 1)

        total = y_test.size(0)
        correct = (predicted == y_test).sum()

        print('- Accuracy of Bayesian NN: %f %%' % (100 * float(correct) / total))
        print('- CrossEntropy_Loss : %2.2f, KLDivergence_Loss : %2.2f' % (ce.item(), kl.item()))
        
        # Using RF_classifcation and XGB_classification to see how these standard ML models perform.
        RF_Classification(X_train, Y_train, X_test, Y_test)
        XGB_Classification(X_train, Y_train, X_test, Y_test)
        
    else:
        if(args.oversampling == 'SMOTE'):
            X_train, Y_train, x_train, y_train = SMOTE_dataset(args.oversampling_threshold, X_train, Y_train)
        elif(args.oversampling == 'ADAYSN'):
            X_train, Y_train, x_train, y_train = ADAYSN_dataset(args.oversampling_threshold, X_train, Y_train)
        else:
            X_train, Y_train, x_train, y_train = SMOTEENN_dataset(args.oversampling_threshold, X_train, Y_train)
            
        for step in tqdm(range(args.iterations)):
            pre = model(x_train)
            ce = ce_loss(pre, y_train)
            kl = kl_loss(model)
            cost = ce + kl_weight*kl
    
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

        pre = model(x_test)
        _, predicted = torch.max(pre.data, 1)

        total = y_test.size(0)
        correct = (predicted == y_test).sum()

        print('- Accuracy of Bayesian NN: %f %%' % (100 * float(correct) / total))
        print('- CrossEntropy_Loss : %2.2f, KLDivergence_Loss : %2.2f' % (ce.item(), kl.item()))
        RF_Classification(X_train, Y_train, X_test, Y_test)
        XGB_Classification(X_train, Y_train, X_test, Y_test)
    
    
