#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:34:10 2019

@author: alisemidtfjord
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import data_loader
import argparse
import numpy as np
import models 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sbo import soft_brownian_offset
from torchmetrics import AUROC


#----- Change these: Learnings rates (f,g_a and g_e)
lr = 0.0005
lr2 = 0.000005
lr3 = 0.01


parser = argparse.ArgumentParser(description='PyTorch SDENet Training')

#----- Change these: Number of epochs (f,g_a and g_e)
parser.add_argument('--epochs', type=int, default=500, help='number of epochs to train')
parser.add_argument('--epochs_aleatory', type=int, default=500, help='number of epochs to train') 
parser.add_argument('--epochs_epistemic', type=int, default=20, help='number of epochs to train')

parser.add_argument('--lr', default=lr, type=float, help='learning rate')
parser.add_argument('--lr2', default=lr2, type=float, help='learning rate')
parser.add_argument('--lr3', default=lr3, type=float, help='learning rate')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=float, default=12)


args = parser.parse_args()
print(args)

# Device
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

#----- Change this: Number of days/time points in your data
batch_size = int(365)

#----- Change these: Number of years/iterations in training and test data
Iter = 5
Iter_test = 2


# Data
print('==> Preparing data..')


torch.manual_seed(args.seed)


if device == 'cuda':
    cudnn.benchmark = True
    torch.cuda.manual_seed(args.seed)
    

#----- Change this: Name used in data_loader to load your dataset
X_train, y_train, X_test, y_test  = data_loader.load_dataset('load_wind_3sound')


year = np.arange(1,batch_size+1)
all_years = year.copy()
for i in range(Iter-1):
    all_years = np.hstack((all_years,year))
    
    
test_years = year.copy()
for i in range(Iter_test-1):
    test_years = np.hstack((test_years,year))
    
scaler = StandardScaler()
all_years = scaler.fit_transform(all_years.reshape(-1,1))
test_years = scaler.transform(test_years.reshape(-1,1))

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.transform(y_test.reshape(-1,1))

target_scale =scaler.scale_
target_mean =scaler.mean_



T = torch.from_numpy(all_years).type(torch.FloatTensor).reshape(-1)
T_test = torch.from_numpy(test_years).type(torch.FloatTensor).reshape(-1)
    

X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.FloatTensor).reshape(-1)
y_test = torch.from_numpy(y_test).type(torch.FloatTensor).reshape(-1)


#----- Change this: Number of lags to use (excluding the present day)
p = 3
#----- Change this: Number of variables/features
m = 9
#----- Number of days forward to predict: Keep zero for now
l = 0

# Model
print('==> Building model..')
net_drift = models.SDENet_drift(m=m,p=p,l=l)
net_drift = net_drift.to(device)

net_diffusion = models.SDENet_diffusion(m=m,p=p)
net_diffusion = net_diffusion.to(device)

net_epistemic = models.SDENet_epistemic(m=m,p=p)
net_epistemic = net_epistemic.to(device)


optimizer_F = optim.SGD([{'params': net_drift.drift.parameters()}], lr=args.lr, momentum=0.9, weight_decay=5e-4)


optimizer_G = optim.SGD([ {'params': net_diffusion.diffusion.parameters()}], lr=args.lr2, momentum=0.9, weight_decay=5e-4)

optimizer_H = optim.SGD([{'params': net_epistemic.diffusion_epistemic.parameters()}], lr=args.lr3, momentum=0.9, weight_decay=5e-4)


real_label = 0
fake_label = 1
criterion = nn.BCELoss()
auroc =  AUROC(pos_label=1, task='binary')
scaler = StandardScaler()



# Load batches of data
def load_training(iternum):
    x = X_train[iternum*batch_size:(iternum+1)*batch_size]
    y = y_train[iternum*batch_size:(iternum+1)*batch_size]
    t = T[iternum*batch_size:(iternum+1)*batch_size]
    return x, y, t



def load_test(iternum):
    x = X_test[iternum*batch_size:(iternum+1)*batch_size]
    y = y_test[iternum*batch_size:(iternum+1)*batch_size]
    t = T_test[iternum*batch_size:(iternum+1)*batch_size]
    return x, y, t


def delete_last_n_columns(a, n):
    n_cols = a.size()[1]
    assert(n<n_cols)
    first_cols = n_cols - n
    mask = torch.arange(0,first_cols)
    b = torch.index_select(a,1,mask) # Retain first few columns; delete last_n columns
    return b

def sigmoid(z):
    return 1/(1+torch.exp(-z))

# Training
def train_drift(epoch):
    print('\nEpoch: %d' % epoch)
    net_drift.train()

    train_loss = 0
    
    for iternum in range(Iter):
        inputs, targets, t = load_training(iternum)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer_F.zero_grad()
        mean, drift = net_drift(inputs, t)
        targets = (targets[p:]).reshape((-1,1))
        mean = mean[:].reshape((-1,1))
        loss = nn.functional.mse_loss(targets, mean) #change this
        loss.backward()
        optimizer_F.step()
        train_loss += loss.item()

        
      
    print('Train epoch:{} \tLoss: {:.6f}'.format(epoch, train_loss/Iter))
    return(mean, drift)




def test_drift(epoch):
#    print(epoch)
    net_drift.eval()
    test_loss = 0
    with torch.no_grad():
        for iternum in range(Iter_test):
            inputs, targets, t = load_test(iternum)
            inputs, targets = inputs.to(device), targets.to(device)

            prediction, current_drift = net_drift(inputs, t=t)
 
                
            current_mean = (prediction[:].reshape(-1,1))
            targets = (targets[p:].reshape(-1,1))
            
            loss = nn.functional.mse_loss(targets, current_mean)
            test_loss += loss.item()
            if epoch % 100 == 0:
                x = np.arange(len(targets))
                plt.plot(x,current_mean.detach(), label='Predicted')
                plt.plot(x,targets, label='True')
                plt.xlabel('Day of year')
                plt.ylabel('U wind')
                plt.legend()
                plt.show()
    
    print('Test epoch:{} \tLoss: {:.6f}'.format(epoch, np.sqrt(test_loss/(Iter_test))))
    return(current_mean, current_drift)



def train_diffusion(epoch):
    print('\nEpoch: %d' % epoch)
    net_drift.eval()
    net_diffusion.train()

    train_loss_in = 0
    
    for iternum in range(Iter):
        inputs, targets, t = load_training(iternum)

        inputs, targets= inputs.to(device), targets.to(device)
        optimizer_G.zero_grad()
    
        
        prediction, current_drift = net_drift(inputs, t)

            
        current_mean = prediction[:].detach().reshape(-1,1)*target_scale
        targets = targets[p:].reshape(-1,1)*target_scale
        
        
        predict_sigma_2 = net_diffusion(inputs, t).reshape(-1)
        predict_sigma_2 = predict_sigma_2[:]

        sigma_2 = ((targets-current_mean.detach()))**2
        sigma_2 = (sigma_2.reshape(-1, 1)).reshape(-1)
        
    
        loss_in = nn.functional.mse_loss(predict_sigma_2.type(torch.FloatTensor), sigma_2.type(torch.FloatTensor))
        loss_in.backward()


        optimizer_G.step()
        train_loss_in += loss_in.item()
      
    print('Train epoch:{} \gLoss: {:.6f}'.format(epoch, train_loss_in/Iter))
    return(predict_sigma_2)



def test_diffusion(epoch):
    net_diffusion.eval()
    test_loss_in = 0
    with torch.no_grad():
        for iternum in range(Iter_test):
            inputs, targets, t = load_test(iternum)
            inputs, targets = inputs.to(device), targets.to(device)

 
            prediction, current_drift = net_drift(inputs, t)
  
                
            current_mean = prediction[:].detach().reshape(-1,1)*target_scale
            targets = targets[p:].reshape(-1,1)*target_scale
 
            sigma_2 = ((targets-current_mean.detach()))**2
            sigma_2 = (sigma_2.reshape(-1, 1)).reshape(-1)

            predict_in = net_diffusion(inputs, t).reshape(-1)
            predict_in = predict_in[:]
            loss_in = nn.functional.mse_loss(predict_in.detach().type(torch.FloatTensor), sigma_2.type(torch.FloatTensor))

            test_loss_in += loss_in.item()
            
            if epoch % 100 == 0:
                x = np.arange(0,len(targets))
                plt.plot(x,sigma_2)
                plt.plot(x,predict_in.detach())
                plt.show()
            
    
    print('Test epoch:{} \gLoss: {:.6f}'.format(epoch, np.sqrt(test_loss_in/(Iter_test))))
    return(predict_in.detach())
           


def train_epistemic(epoch):
    print('\nEpoch: %d' % epoch)
    net_epistemic.train()
    train_loss = 0
    
    for iternum in range(Iter):
        inputs, targets, t = load_training(iternum)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs
        
        label = torch.full((batch_size,1), real_label, device=device)
        label = label[p:]
        

        optimizer_H.zero_grad()
        
        
        out = inputs[:len(inputs)-p]
        out_lags = out.clone()
        for i in range(1,p+1):
            out_lag = inputs[i:len(inputs)-p+i]
            out_lags = torch.column_stack((out_lags, out_lag))
            
        
        
        t_p = t[p:].numpy().reshape((-1,1))
        
        X_train2 = np.hstack((out_lags,t_p))
        
        X_ood_lags = soft_brownian_offset(X_train2, d_min=6.0, d_off=5.8, n_samples=((int((365-p)/2))), softness = 0.5)
        X_ood_lags_1 = soft_brownian_offset(np.column_stack((out,t_p)), d_min=6.0, d_off=5.8, n_samples=((int((365-p)/2))), softness = 0.5)
        X_ood_lags_pres = X_ood_lags_1.copy()

        for i in range(p):
            X_ood_lags_pres = np.column_stack((X_ood_lags_1[:,:-1],X_ood_lags_pres))
            

        X_ood_lags = np.row_stack((X_ood_lags, X_ood_lags_pres))
        X_ood_lags = X_ood_lags[X_ood_lags[:,-1]<X_train2[:,-1].max()]
        X_ood_lags = X_ood_lags[X_ood_lags[:,-1]>X_train2[:,-1].min()]
        
        label_fake = torch.full((len(X_ood_lags),1), fake_label, device=device)
        
        label = torch.vstack((label,label_fake))
        
        X = np.vstack((X_train2,X_ood_lags))
        
        
        X = torch.from_numpy(X)

        
        predict= net_epistemic(X)
        
        
        loss = criterion(predict.to(torch.float32), label.detach().to(torch.float32))
        loss.backward()

        
        train_loss += loss.item()
        optimizer_H.step()
        
        
      
    print('Train epoch:{} Loss 1:  {:.6f}'.format(epoch, train_loss/Iter))
    return()

def test_epistemic(epoch):
    net_epistemic.eval()
    test_loss = 0
    auc = 0
    
    for iternum in range(Iter_test):
        inputs, targets, t = load_test(iternum)

        inputs, targets = inputs.to(device), targets.to(device)
        
        
        label = torch.full((batch_size,1), real_label, device=device)
        label = label[p:]
    

        out = inputs[:len(inputs)-p]
        out_lags = out.clone()
        for i in range(1,p+1):
            out_lag = inputs[i:len(inputs)-p+i]
            out_lags = torch.column_stack((out_lags, out_lag))
        
        t_p = t[p:].numpy().reshape((-1,1))
        
        X_test = np.hstack((out_lags,t_p))
        

        X_ood_lags = soft_brownian_offset(X_test, d_min=1.2, d_off=1.2, n_samples=((365-p)),softness=0.5, random_state = 111)
        X_ood_lags = X_ood_lags[X_ood_lags[:,-1]<X_test[:,-1].max()]
        X_ood_lags = X_ood_lags[X_ood_lags[:,-1]>X_test[:,-1].min()]
        
        label_fake = torch.full((len(X_ood_lags),1), fake_label, device=device)
        
        label = torch.vstack((label,label_fake))

        X = np.vstack((X_test,X_ood_lags))
        
        X = torch.from_numpy(X)
    
        
        predict_in = net_epistemic(X)
        

        
        loss = criterion(predict_in.to(torch.float32), label.detach().to(torch.float32))

        test_loss += loss.item()
        
        auc_loc = auroc(predict_in.to(torch.float32), label.detach().to(torch.int64), task='binary')
        auc += auc_loc
        
        
        
        predict_new = net_epistemic(torch.from_numpy(X_test))
        
        predict_new = predict_new[1:len(X)]
        
        
    print('Test epoch:{} \gLoss: {:.6f} ROC AUC: {:.6f}'.format(epoch, test_loss/Iter_test, auc/Iter_test))
    return(predict_new.detach())



print('Train drift')
for epoch in range(0, args.epochs):
    train_drift(epoch)
    pred, drift = test_drift(epoch)
    
    
print('Train diffusion')
for epoch in range(0, args.epochs_aleatory):
    train_diffusion(epoch)
    diffusion = test_diffusion(epoch)
    
    
print('Train epistemic')
for epoch in range(0, args.epochs_epistemic):
    train_epistemic(epoch)
    pred_epistemic = test_epistemic(epoch)
    
for iternum in range(Iter_test):
    inputs, targets, t = load_test(iternum)
    inputs, targets = inputs.to(device), targets.to(device)

    
    out = inputs[:len(inputs)-p]
    out_lags = out.clone()
    for i in range(1,p+1):
        out_lag = inputs[i:len(inputs)-p+i]
        out_lags = torch.column_stack((out_lags, out_lag))
        
    prediction, current_drift = net_drift(inputs, t)

    
    apred = np.array(prediction.detach()).reshape(-1,1)*target_scale+target_mean
    apred = apred[:]
    
    
    atrue = np.array(targets)
    atrue = atrue[p:].reshape(-1,1)*target_scale+target_mean
    
    adrift= np.array(current_drift.detach())[:,0]
    adrift = adrift[:]
    
    # ---- g_a ----
    
    
    pred_diffusion = net_diffusion(inputs, t)
    adiffusion = np.array(pred_diffusion.detach())
    adiffusion = adiffusion[:]

    
    # ---- g_e ----
    
    
    X_test_ep = torch.column_stack((out_lags,t[p:]))
    
    pred_epistemic = net_epistemic(X_test_ep.detach())
    pred_epistemic = pred_epistemic[:]
    aepistemic = np.array(pred_epistemic.detach())

    x = np.arange(0,len(atrue))
    

    plt.plot(x,apred, label='Predicted')
    plt.plot(x,atrue, label='True')
    diff = (atrue-apred)
    plt.xlabel('Day')
    plt.ylabel('U Wind')
    plt.legend()
    plt.show()
    
    const_ep = 20
    
    
    plt.plot(x,(diff)**2, label=r'$e^2$')
    plt.plot(x,adiffusion, label = r'$g_a^2$ Predicted')
    plt.xlabel('Day')
    plt.ylabel('Residuals squared')
    plt.legend()
    plt.show()
    
    plt.plot(x,diff**2, label=r'$e^2$')
    plt.plot(x,(const_ep*aepistemic)**2, label = r'$g_e^2$')
    plt.xlabel('Day')
    plt.ylabel('Residuals squared')
    plt.legend()
    plt.show()


    plt.plot(x,diff**2, label=r'$e^2$')
    plt.plot(x,(const_ep*aepistemic)**2, label = r'$g_e^2$')
    plt.plot(x,adiffusion, label = r'$g_a^2$')
    plt.xlabel('Day')
    plt.ylabel('Residuals squared')
    plt.legend()
    plt.show()
    
    
    plt.plot(x,diff**2, label=r'$e^2$')
    plt.plot(x,adiffusion+(const_ep*aepistemic)**2, label=r'$g^2$')
    plt.xlabel('Day')
    plt.ylabel('Residuals squared')
    plt.legend()
    plt.show()
    
    ga = np.sqrt(adiffusion)
    ge = np.sqrt((const_ep*aepistemic)**2)
    ci_ge = 1.96 * ge
    ci_ga = 1.96 * ga
    y_max = (apred+ci_ga+ci_ge).max()
    y_min = (apred-ci_ga-ci_ge).min()
    plt.ylim(y_min, y_max)
    plt.plot(x,apred, linewidth=1, label='Predicted')
    plt.fill_between(x, (apred-ci_ga).reshape(-1), (apred+ci_ga).reshape(-1),  alpha=0.4,  edgecolor="none", label=r'95% CI from $g_a$')
    plt.plot(x,atrue, color='darkorange', linewidth=1, label='True')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('U Wind')
    plt.show()
    
    plt.plot(x,apred, linewidth=1, label='Predicted')
    plt.ylim(y_min, y_max)
    plt.fill_between(x, (apred-ci_ge).reshape(-1), (apred+ci_ge).reshape(-1), alpha=0.4, label=r'95% CI from $g_e$')
    plt.plot(x,atrue, color='darkorange', linewidth=1, label='True')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('U Wind')
    plt.show()
    
    plt.plot(x,apred, linewidth=1, label='Predicted')
    plt.ylim(y_min, y_max)
    plt.fill_between(x, (apred-ci_ge-ci_ga).reshape(-1), (apred+ci_ge+ci_ga).reshape(-1), alpha=0.4, label=r'95% CI from g')
    plt.plot(x,atrue, color='darkorange', linewidth=1, label='True')
    plt.legend()
    plt.xlabel('Day')
    plt.ylabel('U Wind')
    plt.show()

    
    if iternum == 0:
        apred_s = apred
        atrue_s = atrue
        a_drift = adrift
        a_diffusion = adiffusion
        a_epistemic = aepistemic
    else:
        apred_s = np.append(apred_s,apred)
        atrue_s = np.append(atrue_s,atrue)
        a_drift = np.append(a_drift,adrift)
        a_diffusion = np.append(a_diffusion,adiffusion)
        a_epistemic = np.append(a_epistemic,aepistemic)

    
x = np.arange(0,len(atrue_s))
plt.plot(x,apred_s, label='Predicted')
plt.plot(x,atrue_s, label='True')
diff = (atrue_s-apred_s)
plt.xlabel('Day')
plt.ylabel('U Wind')
plt.legend()
plt.show()



c = a_drift*target_scale
plt.plot(x,atrue_s)
plt.plot(x,c)
plt.show()


#----- Change this: sigma_max, maximum value of epistemic uncertainty
const_ep = 20

plt.plot(x,diff**2, label=r'$e^2$')
plt.plot(x,(const_ep*a_epistemic)**2 +a_diffusion, label = r'$g_tot^2$')
plt.plot(x,(const_ep*a_epistemic)**2, label = r'$g_e^2$')
plt.xlabel('Day')
plt.ylabel('Residuals squared')
plt.legend()
plt.show()

plt.plot(x,diff**2, label=r'$e^2$')
plt.plot(x,(const_ep*a_epistemic)**2+ a_diffusion, label = r'$g_e^2$')
plt.plot(x,a_diffusion, label = r'$g_a^2$')
plt.xlabel('Day')
plt.ylabel('Residuals squared')
plt.legend()
plt.show()

plt.plot(x,diff**2, label=r'$e^2$')
plt.plot(x,a_diffusion, label = r'$g_a^2$')
plt.xlabel('Day')
plt.ylabel('Residuals squared')
plt.legend()
plt.show()

np.random.seed(111)

Z = np.random.normal(size=len(apred_s))
total_eq = apred_s + ((a_diffusion**(1/2)))*Z
plt.plot(x,total_eq)
plt.plot(x,atrue_s)
plt.show()

Z = np.random.normal(size=len(apred_s))
total_eq = apred_s + ((a_diffusion**(1/2))+const_ep*a_epistemic)*Z
plt.plot(x,total_eq)
plt.plot(x,atrue_s)
plt.show()

plt.plot(x,diff)
plt.plot(x,((a_diffusion**(1/2)))*Z)
plt.show()
         

plt.plot(x,diff)
plt.plot(x,((a_diffusion**(1/2))+const_ep*a_epistemic)*Z)
plt.show()
         

plt.plot(x,diff**2, label='True')
plt.plot(x,a_diffusion, label='Predicted g\u209a')
plt.xlabel('Day of year')
plt.ylabel('Residuals squared')
plt.legend()
plt.show()

plt.plot(x,diff**2, label=r'$e^2$')
plt.plot(x,a_diffusion+(const_ep*a_epistemic)**2, label=r'$g^2$')
plt.xlabel('Day')
plt.ylabel('Residuals squared')
plt.legend(loc='upper center')
plt.show()


ga = np.sqrt(a_diffusion)
ge = np.sqrt((const_ep*a_epistemic)**2)
ci_ge = 1.96 * ge
ci_ga = 1.96 * ga
plt.plot(x,apred_s, color='darkgreen', linewidth=1)
plt.fill_between(x, (apred_s-ci_ga).reshape(-1), (apred_s+ci_ga).reshape(-1),  alpha=0.5,  edgecolor="none")
plt.fill_between(x, (apred_s-ci_ga-ci_ge).reshape(-1), (apred_s-ci_ga).reshape(-1), alpha=0.5, edgecolor="none")
plt.fill_between(x, (apred_s+ci_ga+ci_ge).reshape(-1), (apred_s+ci_ga).reshape(-1), alpha=0.5, edgecolor="none")
plt.plot(x,atrue_s, color='darkorange', linewidth=1)
plt.show()

g_buffer = ((a_diffusion**(1/2))+const_ep*a_epistemic)
bm = diff/g_buffer
plt.plot(x,bm)
plt.show()

from scipy.stats import norm, probplot, kstest
mu, std = norm.fit(bm)
prob = norm.pdf(bm, mu, std)
probplot(bm, dist='norm',sparams=(mu, std), plot=plt)
print(kstest(bm,'norm', args = (mu, std)))
plt.show()


np.sqrt(((diff**2-a_diffusion)**2).mean())
np.sqrt(((diff**2-const_ep*a_epistemic)**2).mean())

print(np.sqrt(((diff**2-(np.sqrt(a_diffusion)+(const_ep*a_epistemic))**2)**2).mean()))


np.sqrt(((diff**2-a_diffusion)**2).mean())
np.sqrt(((diff**2-(const_ep*a_epistemic)**2)**2).mean())


print('RMSE f: ' + str(np.sqrt(((diff)**2).mean())))
print('RMSE bm: ' + str(np.sqrt(((bm)**2).mean())))
print('RMSE g**2: ' + str(np.sqrt(((diff**2-(np.sqrt(a_diffusion)+(const_ep*a_epistemic))**2)**2).mean())))
print('RMSE g: ' + str(np.sqrt(((np.sqrt(diff**2)-np.sqrt((np.sqrt(a_diffusion)+(const_ep*a_epistemic))**2))**2).mean())))


