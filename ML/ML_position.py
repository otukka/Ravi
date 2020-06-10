# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:10:50 2019

@author: VStore
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:27:09 2019

@author: VStore
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:48:22 2019

@author: VStore
"""
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.svm import  SVC, SVR
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
#import lightgbm as lgb
from sklearn.metrics import accuracy_score,explained_variance_score,mean_squared_error

# devel:
from DataHandler import DataHandler
from Cases import * 

from ML import ndcg, ML_NDCG






class ML_position:
    
    def __init__(self,case,X_train,X_test,Y_train,Y_test):
        self.case = case
        
        if type(X_train) != type(None):
            self.X_train = X_train.reset_index(drop=True)
            self.Y_train = Y_train.reset_index(drop=True)
            
            self.X_train = self.X_train[self.Y_train.disqualified==0]
            self.Y_train = self.Y_train[self.Y_train.disqualified==0]
    
    
            self.X_test = X_test
            self.Y_test = Y_test 
        else:
            self.X_train,self.X_test,self.Y_train,self.Y_test = X_train,X_test,Y_train,Y_test 
        self.models = {
               'XGBOOST1':self.model1,
               'SVR1':self.model_SVR
               } 
        self.fits = {
               'XGBOOST1':self.fit1,
               'SVR1':self.fit_SVR
               } 
        self.predicts = {
               'XGBOOST1':self.predict1,
               'SVR1':self.predict_SVR
               } 
        
        self.models[self.case['ML_gallop']]()
        
        
    #_____ Common code starts
        
    def model1(self):
        if type(self.X_train) != type(None):
            self.train_data = xgb.DMatrix(self.X_train.iloc[:-5000], label=self.Y_train.iloc[:-5000]['position.BOOL.1.0'])
            self.train_data1 = xgb.DMatrix(self.X_train.iloc[-5000:], label=self.Y_train.iloc[-5000:]['position.BOOL.1.0'])
            self.test_data = xgb.DMatrix(self.X_test, label=self.Y_test['position.BOOL.1.0'])
            
        self.model = []   
        
    def fit(self):
        self.fits[self.case['ML']]()
        
    def predict(self, data):        
        return self.predicts[self.case['ML']](data)
        
    def read_model(self,filenamepath):
        self.model = pickle.load(open(filenamepath+'.dat','rb'))
    
        if self.case['ML'] == 'XGBOOST1':
            self.bags = len(self.model)
            
        
    def save_model(self, filenamepath):
        pickle.dump(self.params,open(filenamepath+'_params.dat','wb'))
        pickle.dump(self.model,open(filenamepath+'.dat','wb'))

    def predict1(self, data):     
        eval_data = xgb.DMatrix(data)
        
        i = 0
        predictions = np.zeros((data.shape[0],self.bags))
        for i in range(self.bags):     
            clf = self.model[i]          
            Y_pred = clf.predict(eval_data,ntree_limit=clf.best_ntree_limit)
            Y_pred = 1-Y_pred
            predictions[:,i] = Y_pred.reshape((self.X_test.shape[0]))
            i += 1
            
        return np.mean(predictions,axis=1)


    def print_attributes(self):
        for model in self.model:
            print(model.attributes())
            
            
    def save_tuned(self,eval_limit,filenamepath):
        
        tuned_list = []
        for model in self.model:
            if float(model.attributes()['best_score']) > eval_limit:
                tuned_list.append(model)

            
        pickle.dump(tuned_list,open(filenamepath+'_tuned.dat','wb'))



    #_____ Common code starts ends
    
    
    
        
    def fit1(self):
        evallist = [(self.train_data, 'train'),(self.train_data1, 'test')]
        num_round = 1000
        
        self.bags = 30
        self.params = []
        seeds = np.random.randint(1,1000,self.bags)
        
        i = 0
        for seed in seeds:
            
            param = {
                'nthread':8,
                'max_depth':int(np.random.randint(1,10,1)),
                 'n_estimators':int(np.random.randint(10e3,10e5,1)),
                 'min_child_weight':float(np.random.randint(0,10,1)),
                 'gamma':float(np.random.randint(0,5,1)),
                 'objective':'binary:logistic',
                 'tree_method': 'gpu_hist',
                 'predictor':'cpu_predictor',
                 'max_bin': 16, 
                 'gpu_id': 0,
                 'max_delta_step':float(np.random.randint(1,10,1)),
                 'learning_rate':float(np.random.randint(5,30,1)*0.01),
                 'subsample  ':float(np.random.randint(10,100,1)*0.01),
                 'colsample_bytree':float(np.random.randint(10,100,1)*0.01),
                 'colsample_bylevel':float(np.random.randint(10,100,1)*0.01),
#                 'base_score':float(np.random.randint(1,100,1)*0.01),
#                 'seed':seed,
                 'eval_metric':'mae',
                 }
            self.params.append(param)
            print('Training model {}'.format(i+1))
            clf = xgb.train(param, self.train_data, num_round, evallist,early_stopping_rounds=5,obj=squared_log,feval=ndcg,maximize=True)
            
            pickle.dump(clf,open('Models/ML_position_'+str(i)+'.dat','wb'))
            i += 1
            clf.__del__()
            del clf
            time.sleep(2)
            
            
        # after training load all to back memory (cpu_predictor)
        for j in range(self.bags):
            self.model.append(pickle.load(open('Models/ML_position_'+str(j)+'.dat','rb')))




    #______   SVR code starts 

    def model_SVR(self):
        self.Y_train = self.Y_train['combinedDisqualified']
        self.model = SVR()
        params = self.model.get_params()
        for param in params:
            if 'model_params' in self.case:
                if param in self.case['model_params']:
                    params[param] = self.case['model_params'][param]
        
                self.model.set_params(**self.case['model_params'])
        print(self.model.get_params())
    def fit_SVR(self):
        self.model.fit(self.X_train,self.Y_train)
    def predict_SVR(self):     
        predictions = self.model.predict(self.X_test)
        return predictions

     #______   SVR code starts ends 


def test():
    dh = DataHandler(case)
    
    X_train, X_test, Y_train, Y_test = dh.get_XY_test_train('2019-05-01')
    
    trainInfo = dh.get_raceInfo('2019-05-01')[0]
    testInfo = dh.get_raceInfo('2019-05-01')[1]
    
    def save1():
        X_train.to_parquet('Datasets/X_train.gzip',compression='gzip',index=True)
        X_test.to_parquet('Datasets/X_test.gzip',compression='gzip',index=True)
        Y_train.to_parquet('Datasets/Y_train.gzip',compression='gzip',index=True)
        Y_test.to_parquet('Datasets/Y_test.gzip',compression='gzip',index=True)
        trainInfo.to_parquet('Datasets/trainInfo.gzip',compression='gzip',index=True)
        testInfo.to_parquet('Datasets/testInfo.gzip',compression='gzip',index=True)
    
    def load1():
        case = case1()
        X_train = pd.read_parquet('Datasets/X_train.gzip')
        X_test = pd.read_parquet('Datasets/X_test.gzip')
        Y_train = pd.read_parquet('Datasets/Y_train.gzip')
        Y_test = pd.read_parquet('Datasets/Y_test.gzip')
        trainInfo = pd.read_parquet('Datasets/trainInfo.gzip')
        testInfo = pd.read_parquet('Datasets/testInfo.gzip')
        
        

    
    X_train.reset_index(drop=True, inplace=True)
    Y_train.reset_index(drop=True, inplace=True)
    
    X_train = X_train[Y_train.disqualified==0]
    Y_train = Y_train[Y_train.disqualified==0]
    
    
    M = ML_position(case,X_train, X_test, Y_train, Y_test)
    M.fit()
    pred = M.predict()
    
    Y_test['prediction'] = pred
    testInfo['prediction'] = pred
    Y_test = Y_test.drop(columns=['position.BOOL.'+str(i)+'.0' for i in np.arange(4,16)])
    
    testInfo.loc[testInfo.disqualificationCode=='p','prediction'] = 0.0
    view = testInfo[(testInfo.position==1)]      
    view = testInfo[(testInfo.position==2)]  


    def save():
        tuned_list = []
        for i in range(M.bags):
            clf = pickle.load(open('Models/ML_position_'+str(i)+'.dat','rb'))
            print(clf.attributes())
            
            if float(clf.attributes()['best_score']) < 0.121:
                tuned_list.append(clf)

            
        pickle.dump(tuned_list,open('Models/ML_position_tuned.dat','wb'))
        
        
        tuned = []
        for model in M.model:
            if float(model.attributes()['best_score']) < 0.14:
                tuned.append(model)
                
        pickle.dump(tuned,open('Models/ML_position_tuned.dat','wb'))
        
        

                
        for model in M.params:
            print(model)

    
    print(Y_test[case['position_column']].sum())
    print('Hevosia testidatassa', Y_test.shape[0])
    for i in np.arange(0.01,1.0,0.01):
        temp = Y_test[(Y_test.prediction >= i) & (Y_test.prediction < (i+0.01))]
        if temp.shape[0] > 0:
            print('Ennusteväli: {:.2f}-{:.2f} '.format(i,i+0.01),end='')
            
            for i in np.arange(1.0,3+1.0,1.0):
                
                s = temp['position.BOOL.'+str(i)].sum()/temp.shape[0]
                print('|{:>6.2%} '.format(s),end='')
    
    
            print(' välillä hevosia {}'.format(temp.shape[0]))
    
    
    print(Y_test[case['position_column']].sum())
    print('Hevosia testidatassa', Y_test.shape[0])
    for i in np.arange(0.01,1.0,0.01):
        temp = Y_test[(Y_test.prediction >= i) & (Y_test.prediction < (i+0.01))]
        if temp.shape[0] > 0:
            print('Ennusteväli: {:.2f}-{:.2f} '.format(i,i+0.01),end='')
            
            for i in np.arange(1.0,5+1.0,1.0):
                
                s = temp['position.BOOL.'+str(i)].sum()
                print('|{:>4} '.format(int(s)),end='')
    
    
            print(' välillä hevosia {}'.format(temp.shape[0]))
            
    print('voittajahevosia',Y_test['position.BOOL.1.0'].sum())
    for i in np.arange(0.01,1.0,0.01):
        temp = Y_test[(Y_test.prediction > i)]
        
        if temp.shape[0] > 0:     
            print('Ennusteväli: {:.2f}-{:.2f} '.format(i,1.00),end='')
            s = temp['position.BOOL.1.0'].sum()/temp.shape[0]
            print('{:>4.2%} '.format(s),end='')
            print(' välillä hevosia {}'.format(temp.shape[0]))
            
    print('voittajahevosia',Y_test['position.BOOL.1.0'].sum())
    for i in np.arange(0.01,1.0,0.01):
        temp = Y_test[(Y_test.prediction < i)]
        
        if temp.shape[0] > 0:     
            print('Ennusteväli: {:.2f}-{:.2f} '.format(0.00,i),end='')
            s = temp['position.BOOL.1.0'].sum()/temp.shape[0]
            print('{:>4.2%} '.format(s),end='')
            print(' välillä hevosia {}'.format(temp.shape[0]))     
            


    testInfo = testInfo[testInfo.disqualificationCode !='p']
    mins = []
    maxes = []
    means = []
    std = []
    ids = np.sort(testInfo.position.unique())
    for i in ids:
    #    print('\n\n','sijoitus: ',i)
    #    print(testInfo[testInfo.position==i]['prediction'].describe())
        mins.append(testInfo[testInfo.position==i]['prediction'].min())
        maxes.append(testInfo[testInfo.position==i]['prediction'].max())
        means.append(testInfo[testInfo.position==i]['prediction'].mean())
        std.append(testInfo[testInfo.position==i]['prediction'].std())
        
        
    
    
    mins = np.asarray(mins)
    maxes = np.asarray(maxes)
    means = np.asarray(means)
    std = np.asarray(std)
    
    import matplotlib.pyplot as plt
    
    
    
    # create stacked errorbars:
    plt.figure(figsize=(10,10))
    plt.errorbar(ids, means, std, fmt='ok',ecolor='blue', lw=4)
    plt.errorbar(ids, means, [means - mins, maxes - means],
                 fmt='.k', ecolor='green', lw=2)
    plt.grid('on')
    plt.xlim(ids.min()-1,ids.max()+1 )
    
    

    
    
    
    # create stacked errorbars:
    plt.figure(figsize=(10,10))
    plt.scatter(Y_test.prediction,Y_test.combinedDisqualified, s=10,marker='x')
    
    plt.grid('on')
    #
    #
