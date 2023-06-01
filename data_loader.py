    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:20:18 2019

@author: alise midtfjord


This script created and load in the selected datasets.


"""

# Imports
import numpy as np
import sys


def check_os():
    '''
    Checks the type of operating system the script is running on
    
    :return: The system platform 'Windows' or 'Unix'
    '''

    if sys.platform.startswith('win'):
        return 'Windows'
    else:
        return 'Unix'


def load_wind_temp():
    '''
    Load stratosheric U wind and temperature data
    
    :return: The training features
    :return: The training response (u wind)
    :return: The test features
    :return: The test response (u wind)
    :return: The training temperature
    :return: The test temperature
    '''
    
    filename_uwind = 'u_79_19.txt'
    filename_temp = 'temp_79_19.txt'
    uwind = np.loadtxt(filename_uwind)
    temp = np.loadtxt(filename_temp)
    
    train_size = 30
    per_day = 1
    years = 40
    
    X_train = uwind[0:int(train_size*365*per_day)]
    X_train = np.vstack([X_train,temp[0:int(train_size*365*per_day)]]).T
    y_train = uwind[0:int(int(train_size*365*per_day))].reshape(-1) 
    y_train_2 = temp[0:int(int(train_size*365*per_day))].reshape(-1) 
    
    X_test = uwind[int(int(train_size*365*per_day)):365*years]
    X_test = np.vstack([X_test,temp[int(int(train_size*365*per_day)):365*years]]).T
    y_test = uwind[int(int(train_size*365*per_day)):365*years].reshape(-1) 
    y_test_2 = temp[int(int(train_size*365*per_day)):365*years].reshape(-1) 
    
    return X_train, y_train, X_test, y_test, y_train_2, y_test_2




def load_wind_3sound():
    '''
    Load stratosheric U wind and infrasounds data from 3 stations
    
    :return: The training features
    :return: The training response (u wind)
    :return: The test features
    :return: The test response (u wind)
    '''
    
    os_type = check_os()
    
    if os_type == 'Unix':
        slash = '/'
    elif os_type == 'Windows':
        slash = '\\'

    filename_infrasound_37 = f'data{slash}sound_x37.txt'
    filename_infrasound_18 = f'data{slash}sound_x18.txt'
    filename_infrasound_53 = f'data{slash}sound_x53.txt'
    filename_infrasound_37_2 = f'data{slash}sound_y37.txt'
    filename_infrasound_18_2 = f'data{slash}sound_y18.txt'
    filename_infrasound_53_2 = f'data{slash}sound_y53.txt'
    filename_infrasound_37_3 = f'data{slash}sound_ampl37.txt'
    filename_infrasound_18_3 = f'data{slash}sound_ampl18.txt'
    filename_infrasound_53_3 = f'data{slash}sound_ampl53.txt'
    filename_uwind = f'data{slash}wind.txt'
    filename_infrasound_37 = 'data\sound_x37.txt'
    filename_infrasound_18 = 'data\sound_x18.txt'
    filename_infrasound_53 = 'data\sound_x53.txt'
    filename_infrasound_37_2 = 'data\sound_y37.txt'
    filename_infrasound_18_2 = 'data\sound_y18.txt'
    filename_infrasound_53_2 = 'data\sound_y53.txt'
    filename_infrasound_37_3 = 'data\sound_ampl37.txt'
    filename_infrasound_18_3 = 'data\sound_ampl18.txt'
    filename_infrasound_53_3 = 'data\sound_ampl53.txt'
    filename_uwind = 'data\wind.txt'
    
    infrasound_sin_37 = np.loadtxt(filename_infrasound_37)
    infrasound_sin_18 = np.loadtxt(filename_infrasound_18)
    infrasound_sin_53 = np.loadtxt(filename_infrasound_53)
    infrasound_cos_37 = np.loadtxt(filename_infrasound_37_2)
    infrasound_cos_18 = np.loadtxt(filename_infrasound_18_2)
    infrasound_cos_53 = np.loadtxt(filename_infrasound_53_2)
    infrasound_amp_37 = np.loadtxt(filename_infrasound_37_3)
    infrasound_amp_18 = np.loadtxt(filename_infrasound_18_3)
    infrasound_amp_53 = np.loadtxt(filename_infrasound_53_3)
    uwind = np.loadtxt(filename_uwind)
    
    train_size = 5
    per_day = 1
    
    X_train = infrasound_cos_37[0:int(train_size*365*per_day)]
    X_train = np.vstack([X_train,
                         infrasound_sin_18[0:int(train_size*365*per_day)],infrasound_cos_18[0:int(train_size*365*per_day)],
                         infrasound_sin_53[0:int(train_size*365*per_day)],
                         infrasound_cos_53[0:int(train_size*365*per_day)],infrasound_sin_37[0:int(train_size*365*per_day)],
                         infrasound_amp_37[0:int(train_size*365*per_day)],infrasound_amp_18[0:int(train_size*365*per_day)],
                         infrasound_amp_53[0:int(train_size*365*per_day)]]).T
    y_train = uwind[0:int(int(train_size*365*per_day))]
    
    X_test = infrasound_cos_37[int(train_size*365*per_day):]
    X_test = np.vstack([X_test, 
                        infrasound_sin_18[int(int(train_size*365*per_day)):], infrasound_cos_18[int(int(train_size*365*per_day)):],
                        infrasound_sin_53[int(int(train_size*365*per_day)):],
                        infrasound_cos_53[int(int(train_size*365*per_day)):],infrasound_sin_37[int(int(train_size*365*per_day)):],
                        infrasound_amp_37[int(int(train_size*365*per_day)):],infrasound_amp_18[int(int(train_size*365*per_day)):],
                        infrasound_amp_53[int(int(train_size*365*per_day)):]]).T
    y_test = uwind[int(int(train_size*365*per_day)):]
    
    return (X_train, y_train, X_test, y_test)


def sigmoid(x):
    '''
    Calculate the sigmoid function
    
    :param x: data to transform
    :return: The transformed sigmoid function
    '''
    z = 1/(1 + np.exp(-x))
    return(z)


def f1_lags(X):
    '''
    Simulate the drift function for variable 1
    
    :param X: data to calculate drift function for
    :return: The value of the drift
    '''
    a1 = 5
    a2 = 5
    w1 = np.array((0.03,0.02,0.02,0.05,-0.03,0.01,-0.03,-0.01))
    w2 = np.array((0.01, 0.0,-0.005,0.000,-0.01,0,-0.005,0))
    b1 = 0
    b2 = -0
    num = a1*np.tanh(np.dot(np.transpose(w1),X)+b1) + a2*np.tanh(np.dot(np.transpose(w2),X)+b2)
    return(num)

def f2_lags(X):
    '''
    Simulate the drift function for variable 2
    
    :param X: data to calculate drift function for
    :return: The value of the drift
    '''
    a1 = 5
    a2 = 5
    w1 = np.array((0,0.02,0,-0.03,0,0.01,0,0))
    w2 = np.array((0, 0.01,0,-0.005,0,0,0,-0.005))
    b1 = 0.0
    b2 = -0.0
    num = a1*np.tanh(np.dot(np.transpose(w1),X)+b1) + a2*np.tanh(np.dot(np.transpose(w2),X)+b2)
    return(num)


def g1_func_kappa(t):
    '''
    Simulate the diffusion function for variable 1 used in the numerical study
    
    :param t: data to calculate diffusion function for
    :return: The value of the diffusion
    '''
    a = 4 
    b = 0
    num = a*sigmoid(-5*t/36500+b)
    return(num)

def g1_func(t):
    '''
    Simulate the diffusion function for variable 1 used in comparison study
    
    :param t: data to calculate diffusionfunction for
    :return: The value of the diffusion
    '''
    a = 4 
    b = 0
    num = a*sigmoid(-5*t/365+b)
    return(num)

def g2_func(x):
    '''
    Simulate the diffusion function for variable 2 used in numerical and comparison study
    
    :param x: data to calculate diffusion function for
    :return: The value of the diffusion
    '''
    a = 0.125
    w = np.array((1))
    b = 1
    num = a*sigmoid(w*x+b)
    return(num)



def X_func(present,f1,g1,t,deltat=1):
    '''
    Calculate the new value for variable 1 from the drift and diffusion
    
    :param present: Current valur for variable 1
    :param f1: Drift for variable 1
    :param f1: Diffusion for variable 1
    :param t: Current time
    :param deltat: Time differce between present value and new value
    :return: The new value for variable 1
    :return: The random noise
    '''
    z = float(np.random.normal(size=1))
    num = np.array(present+f1*deltat+g1*z*np.sqrt(deltat))
    return(num,z)

def X_func2(present,f1,g1,t,deltat=1):
    '''
    Calculate the new value for variable 2 from the drift and diffusion
    
    :param present: Current valur for variable 2
    :param f1: Drift for variable 2
    :param f1: Diffusion for variable 2
    :param t: Current time
    :param deltat: Time differce between present value and new value
    :return: The new value for variable 2
    :return: The random noise
    '''
    z = float(np.random.normal(size=1))
    num = np.array(present+f1*deltat+g1*z*np.sqrt(deltat))
    return(num,z)

def g1_func_ood(t):
    '''
    Simulate the diffusion function for OOD data for variable 1 
    
    :param t: data to calculate diffusionfunction for
    :return: The value of the diffusion
    '''
    a = 10
    b = 0
    num = a*sigmoid(-5*t/365+b)
    return(num)

def g2_func_ood(x):
    '''
    Simulate the diffusion function for OOD data for variable 2
    
    :param t: data to calculate diffusionfunction for
    :return: The value of the diffusion
    '''
    a = 0.3125
    w = np.array((1))
    b = 1
    num = a*sigmoid(w*x+b)
    return(num)

def X_func_ood(present,f1,g1,t):
    '''
    Calculate the new value for OOD variable 1 from the drift and diffusion
    
    :param present: Current valur for variable 1
    :param f1: Drift for variable 1
    :param f1: Diffusion for variable 1
    :param t: Current time
    :param deltat: Time differce between present value and new value
    :return: The new value for variable 1
    :return: The random noise
    '''
    z = float(np.random.normal(size=1))
    num = np.array(present+f1+g1*z-0*int(t))
    return(num)

def X_func2_ood(present,f1,g1,t):
    '''
    Calculate the new value for OOD variable 2 from the drift and diffusion
    
    :param present: Current valur for variable 2
    :param f1: Drift for variable 2
    :param f1: Diffusion for variable 2
    :param t: Current time
    :param deltat: Time differce between present value and new value
    :return: The new value for variable 2
    :return: The random noise
    '''
    z = float(np.random.normal(size=1))
    num = np.array(present+f1+g1*z-0*int(t))
    return(num)

def eta1_func(t,x):
    '''
    Simulate the initial path for variable 1
    
    :param x: Previous value
    :param t: Current time
    :return: The new value
    '''
    
    num = 1*(np.sin(t*x)+1)
    return(num)
    
def eta2_func(t,x):
    '''
    Simulate the initial path for variable 2
    
    :param x: Previous value
    :param t: Current time
    :return: The new value
    '''
    num = 2*(np.cos(t*x)+1)
    return(num)


def simulate_data():
    '''
    Simulate the idata for the comparison study
    

    :return: The training features
    :return: The training response
    :return: The test features
    :return: The test response
    :return: The training times
    :return: The test times
    :return: The OOD traning features
    :return: the OOD training reponse
    :return: The OOD test features
    :return: The OOD test reponse
    :return: The test diffusion
    :return: The test random noise
    :return: The validation features
    :return: The validation response
    :return: The OOD validation features
    :return: the OOD validation 
    :return: The validation diffusion
    '''
    
    seed_big = 123 #125
    np.random.seed(seed_big)
    mult = 10
    years = 110    
    n = 365
    
    for k in range(0,years):
        
        x1_eta = np.abs(mult*np.random.normal((1)))
        x2_eta = np.abs(mult*np.random.normal((1)))
        
        a1 = eta1_func(-4,x1_eta)
        b1 = eta1_func(-3,x1_eta)
        c1 = eta1_func(-2,x1_eta)
        d1  = eta1_func(-1,x1_eta)
        a2 = eta2_func(-4,x2_eta)
        b2 = eta2_func(-3,x2_eta)
        c2 = eta2_func(-2,x2_eta)
        d2 = eta2_func(-1,x2_eta)
        
        eta = np.array((d1,d2,c1,c2,b1,b2,a1,a2))
        
        
        g2 = g2_func(eta[-1])
        
        for t in range(0,n+8):
            g1 = g1_func(t)
            
            if t == 0:
                f1 = f1_lags(eta)
                f2 = f2_lags(eta)
                X1, error_1 = X_func(eta[0],f1,g1,t)
                X2, error_1  = X_func2(eta[1],f2,g2,t)
                X = np.hstack((X1,X2,eta))
                g_tot = g1+g2
                error = error_1
            else:
                f1 = f1_lags(X[:8])
                f2 = f2_lags(X[:8])
                X1, error_1 = X_func(X[0],f1,g1,t)
                X2, error_1  = X_func2(X[1],f2,g2,t)
                X = np.hstack((X1,X2,X))
                g_tot = np.hstack((g1+g2,g_tot))
                error = np.hstack((error_1,error))
            
                
                
        g2 = g2_func_ood(eta[-1])
                
                
        for t in range(0,n+8):
            g1 = g1_func_ood(t)
            
            if t == 0:
                f1 = f1_lags(eta)
                f2 = f2_lags(eta)
                X1 = X_func_ood(eta[0],f1,g1,t)
                X2 = X_func2_ood(eta[1],f2,g2,t)
                X_ood = np.hstack((X1,X2,eta))
            else:
                f1 = f1_lags(X_ood[:8])
                f2 = f2_lags(X_ood[:8])
                X1 = X_func_ood(X_ood[0],f1,g1,t)
                X2 = X_func2_ood(X_ood[1],f2,g2,t)
                X_ood = np.hstack((X1,X2,X_ood))
                
        X = X[:-24]
        X_ood = X_ood[:-24]
        g_tot = g_tot[:-8]
        error = error[:-8]
        
        if k == 0:
            X_k = X
            X_ood_k = X_ood
            g_k = g_tot
            error_k = error
        else:
            X_k = np.hstack((X,X_k))
            X_ood_k = np.hstack((X_ood,X_ood_k))
            g_k = np.hstack((g_tot,g_k))
            error_k = np.hstack((error,error_k))
            
    
    X = np.flip(X_k)
    X_ood = np.flip(X_ood_k)    
    g_out = np.flip(g_k[:])
    error_out = np.flip(error_k[:])

    train_size = 90
    years = 100
    
    year = np.arange(0,365)
    T = year.copy()
    for i in range(train_size-1):
        T = np.hstack((T,year))
        
        
    year = np.arange(0,365)
    T_test = year.copy()
    for i in range(years-train_size-1):
        T_test = np.hstack((T_test,year))
        
    
    X1 = X[1::2]
    X2 = X[::2]
    
    X1_ood = X_ood[1::2]
    X2_ood = X_ood[::2]
    
    X_train = X1[0:int(train_size*365)]
    X_train = np.vstack([X_train,X2[0:int(train_size*365)]]).T
    y_train = X1[0:int(int(train_size*365))].reshape(-1) 
    
    X_test = X1[int(int(train_size*365)):-10*365]
    X_test = np.vstack([X_test,X2[int(int(train_size*365)):-10*365]]).T
    y_test = X1[int(int(train_size*365)):-10*365].reshape(-1)  
    
    X_val = X1[int(int((train_size+10)*365)):]
    X_val = np.vstack([X_val,X2[int(int((train_size+10)*365)):]]).T
    y_val = X1[int(int((train_size+10)*365)):].reshape(-1)  

    
    X_train_ood = X1_ood[0:int(train_size*365)]
    X_train_ood = np.vstack([X_train_ood,X2_ood[0:int(train_size*365)]]).T
    y_train_ood = X1_ood[0:int(int(train_size*365))].reshape(-1) 
    
    X_test_ood = X1_ood[int(int(train_size*365)):-10*365]
    X_test_ood = np.vstack([X_test_ood,X2_ood[int(int(train_size*365)):-10*365]]).T
    y_test_ood = X1_ood[int(int(train_size*365)):-10*365].reshape(-1)  
    
    X_val_ood = X1_ood[int(int((train_size+10)*365)):]
    X_val_ood = np.vstack([X_val_ood,X2_ood[int(int((train_size+10)*365)):]]).T
    y_val_ood = X1_ood[int(int((train_size+10)*365)):].reshape(-1)  
    
    g_out = g_out[int(int(train_size*365)):-10*365]
    g_out_val = g_out[int(int((train_size+10)*365)):]
    error_out = error_out[int(int(train_size*365)):]
    

    
    return X_train, y_train, X_test, y_test, T, T_test, X_train_ood, y_train_ood, X_test_ood, y_test_ood, g_out, error_out, X_val, y_val, X_val_ood, y_val_ood, g_out_val


def simulate_data_kappa(given_seed):
    '''
    Simulate the idata for the numerical study
    
    :param given_seed: Random seed
    :return: The features
    :return: The response
    :return: The second response (variable 2)
    :return: The times
    :return: The random noise
    :return: The second random noise (variable 2)
    :return: The diffusion term
    :return: The second diffusion term (variable 2)
    :return: The drift term
    :return: The second drift term (variable 2)
    '''

    deltat = 0.01
    seed_big = 123 + given_seed 
    np.random.seed(seed_big)

    mult = 10
    years = 1000
          
    
    for k in range(0,years):
        print('Year ' + str(k))
        
        
        x1_eta = np.abs(mult*np.random.normal((1)))
        x2_eta = np.abs(mult*np.random.normal((1)))
        
        a1 = eta1_func(-12,x1_eta)
        b1 = eta1_func(-11,x1_eta)
        c1 = eta1_func(-10,x1_eta)
        d1  = eta1_func(-9,x1_eta)
        a2 = eta2_func(-12,x2_eta)
        b2 = eta2_func(-11,x2_eta)
        c2 = eta2_func(-10,x2_eta)
        d2 = eta2_func(-9,x2_eta)
        
        eta = np.array((d1,d2,c1,c2,b1,b2,a1,a2))
        

        g2 = g2_func(eta[-1])
        
        for t_8 in range(0,500+2500+8):
            t = t_8-508
            g1 = g1_func_kappa(t)
            
            if t_8 == 0:
                f1 = f1_lags(eta)
                f2 = f2_lags(eta)
                X1, error_1 = X_func(eta[0],f1,g1,t,deltat)
                X2, error_2  = X_func2(eta[1],f2,g2,t,deltat)
                X = np.hstack((X1,X2,eta))
                error = error_1
                error2 = error_2
                g_tot = g1
                g_tot2 = g2
                t_tot = t
                f1_tot = f1
                f2_tot = f2

            else:
                f1 = f1_lags(X[:8])
                f2 = f2_lags(X[:8])
                X1, error_1 = X_func(X[0],f1,g1,t,deltat)
                X2, error_2  = X_func2(X[1],f2,g2,t,deltat)
                X = np.hstack((X1,X2,X))
                error = np.hstack((error_1,error))
                error2 = np.hstack((error_2,error2))
                g_tot = np.hstack((g1,g_tot))
                g_tot2 = np.hstack((g2,g_tot2))
                t_tot = np.hstack((t,t_tot))
                f1_tot = np.hstack((f1,f1_tot))
                f2_tot = np.hstack((f2,f2_tot))
            
        X = X[:-24]
        error = error[:-8]
        error2 = error2[:-8]
        g_tot = g_tot[:-8]
        g_tot2 = g_tot2[:-8]
        t_tot = t_tot[:-8]
        f1_tot = f1_tot[:-8]
        f2_tot = f2_tot[:-8]
        
        if k == 0:
            X_k = X
            error_k = error
            error_k2 = error2
            g_k = g_tot
            g_k2 = g_tot2
            t_k = t_tot
            f1_k = f1_tot
            f2_k = f2_tot

        else:
            X_k = np.hstack((X,X_k))
            error_k = np.hstack((error,error_k))
            error_k2 = np.hstack((error2,error_k2))
            g_k = np.hstack((g_tot,g_k))
            g_k2 = np.hstack((g_tot2,g_k2))
            t_k = np.hstack((t_tot,t_k))
            f1_k = np.hstack((f1_tot,f1_k))
            f2_k = np.hstack((f2_tot,f2_k))
 
    
    X = np.flip(X_k)  
    error_out = np.flip(error_k[:])
    error_out2 = np.flip(error_k2[:])
    g_out = np.flip(g_k[:])
    g_out2 = np.flip(g_k2[:])
    T = np.flip(t_k[:])
    F1 = np.flip(f1_k[:])
    F2 = np.flip(f2_k[:])
    

    
    X1 = X[1::2]
    X2 = X[::2]


    X_train = np.vstack([X1,X2]).T
    y_train = X1.reshape(-1) 
    y_train2 = X2.reshape(-1)

    
    return X_train, y_train, y_train2, T, error_out, error_out2, g_out, g_out2, F1, F2



def load_dataset(dataset, seed=None):
    '''
    Load the selected dataset
    
    :param dataset: Selected dataset
    :param seed: Random seed for simulation
    
    '''
    
    if dataset == 'load_wind_3sound':
         X_train, y_train, X_test, y_test = load_wind_3sound()
         return X_train, y_train, X_test, y_test
    elif dataset =='wind_temp':
         X_train, y_train, X_test, y_test, y_train2, y_test2 = load_wind_temp()
         return X_train, y_train, X_test, y_test, y_train2, y_test2
    elif dataset == 'simulation':
         X_train, y_train, X_test, y_test, T, T_test, X_train_ood, y_train_ood, X_test_ood, y_test_ood,g_out, error_out, X_val, y_val, X_val_ood, y_val_ood, g_out_val = simulate_data()
         return X_train, y_train, X_test, y_test, T, T_test, X_train_ood, y_train_ood, X_test_ood, y_test_ood,g_out, error_out, X_val, y_val, X_val_ood, y_val_ood, g_out_val
    elif dataset== 'simulation_kappa':
        X_train, y_train, y_train2, T, error, error2, g, g2, F1, F2 = simulate_data_kappa(seed)
        return  X_train, y_train, y_train2, T, error, error2, g, g2, F1, F2
