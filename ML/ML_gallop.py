# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 22:27:09 2019

@author: VStore
"""
import pickle
import time

import numpy as np
import pandas as pd

from scipy.stats import mode
from sklearn.svm import  SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,explained_variance_score,mean_squared_error

import xgboost as xgb
#import lightgbm as lgb


# devel:
from DataHandler import DataHandler
from Cases import * 




class ML_gallop:
    
    def __init__(self,case,X_train,X_test,Y_train,Y_test):
        self.case = case
        
        self.column = self.case['gallop_column']
        
#        self.scaler = StandardScaler()
#        self.scaler.fit(X_train)
#        self.X_train,self.X_test,self.Y_train,self.Y_test = self.scaler.transform(X_train),self.scaler.transform(X_test),Y_train,Y_test
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
            self.train_data = xgb.DMatrix(self.X_train.iloc[:-5000], label=self.Y_train.iloc[:-5000]['position'])
            self.train_data1 = xgb.DMatrix(self.X_train.iloc[-5000:], label=self.Y_train.iloc[-5000:]['position'])
            self.test_data = xgb.DMatrix(self.X_test, label=self.Y_test['position'])
        
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

            predictions[:,i] = Y_pred.reshape((self.X_test.shape[0]))
            i += 1
            
        return np.mean(predictions,axis=1)


    def print_attributes(self):
        for model in self.model:
            print(model.attributes())
            
            
    def save_tuned(self,eval_limit,filenamepath):
        
        tuned_list = []
        for model in self.model:
            if float(model.attributes()['best_score']) < eval_limit:
                tuned_list.append(model)

            
        pickle.dump(tuned_list,open(filenamepath+'_tuned.dat','wb'))



    #_____ Common code starts ends
        
        
    def fit1(self):
        evallist = [(self.train_data, 'train'),(self.train_data1, 'test')]
        num_round = 1000
        
        self.bags = 40
        self.params = []
        seeds = np.random.randint(1,1000,self.bags)
        
        i = 0
        for seed in seeds:
            
            param = {
                'nthread':8,
                'max_depth':int(np.random.randint(1,20,1)),
                 'n_estimators':int(np.random.randint(10e3,10e5,1)),
                 'min_child_weight':float(np.random.randint(0,10,1)),
                 'gamma':float(np.random.randint(0,5,1)),
                 'objective':'binary:logistic',
                 'tree_method': 'gpu_hist',
                 'predictor':'cpu_predictor',
                 'max_bin': 16, 
                 'gpu_id': 0,
                 'max_delta_step':float(np.random.randint(1,10,1)),
                 'learning_rate':float(np.random.randint(1,30,1)*0.01),
                 'subsample  ':float(np.random.randint(1,100,1)*0.01),
                 'colsample_bytree':float(np.random.randint(1,100,1)*0.01),
                 'colsample_bylevel':float(np.random.randint(1,100,1)*0.01),
                 'base_score':float(np.random.randint(1,100,1)*0.01),
                 'seed':seed,
                 'eval_metric':'mae',
                 }
            self.params.append(param)
            clf = xgb.train(param, self.train_data, num_round, evallist,early_stopping_rounds=5)
            print('Training model {}'.format(i+1))
            pickle.dump(clf,open('Models/ML_gallop_'+str(i)+'.dat','wb'))
            i += 1
            clf.__del__()
            del clf
            time.sleep(2)
            
            
        # after training load all to back memory (cpu_predictor)
        for j in range(self.bags):
            self.model.append(pickle.load(open('Models/ML_gallop_'+str(j)+'.dat','rb')))





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
    case = case1()
    #case['gallop_column'] = 'disqualified'
    #case['gallop_column'] = 'galloped'
    case['gallop_column'] = 'combinedDisqualified'
    dh = DataHandler(case)
    X_train, X_test, Y_train, Y_test = dh.get_XY_test_train('2019-05-01')
    #trainInfo = dh.get_raceInfo('2019-05-01')[0]
    #testInfo = dh.get_raceInfo('2019-05-01')[1]
    #
    #
    M = ML_gallop(case,X_train, X_test, Y_train, Y_test)
    M.fit()
#    M.read_model()
    pred = M.predict()
    
    
    

    def save():
        tuned_list = []
        for i in range(M.bags):
            clf = pickle.load(open('Models/ML_gallop_'+str(i)+'.dat','rb'))
            print(clf.attributes())
            
            if float(clf.attributes()['best_score']) < 0.35:
                tuned_list.append(clf)

            
        pickle.dump(tuned_list,open('Models/ML_gallop_tuned.dat','wb'))
                
        for params in M.params:
            print(params)
        
    Y_test['prediction'] = pred
    testInfo['prediction'] = pred
    
    
    print('Hevosia testidatassa', Y_test.shape[0])
    for i in np.arange(0.05,1.0,0.05):
        
        temp = Y_test[(Y_test.prediction >= i) & (Y_test.prediction < (i+0.05))]
        if len(temp) > 0:
            s  = temp[case['gallop_column']].sum()/temp.shape[0]
            print('Ennusteväli: {:.2f}-{:.2f} laukka-% välillä: {:.2%} välillä hevosia {}'.format(i,i+0.05,s,temp.shape[0]))
    
    print('\n')
    for i in np.arange(0.01,1.0,0.01):
        temp = Y_test[(Y_test.prediction > i)]
        
        if temp.shape[0] > 0:     
            print('Ennusteväli: {:.2f}-{:.2f} '.format(i,1.00),end='')
            s = temp[case['gallop_column']].sum()/temp.shape[0]
            print('{:>4.2%} '.format(s),end='')
            print(' välillä hevosia {}'.format(temp.shape[0]))
    
    
    print('\n')
    for i in np.arange(0.01,1.0,0.01):
        temp = Y_test[(Y_test.prediction < i)]
        
        if temp.shape[0] > 0:     
            print('Ennusteväli: {:.2f}-{:.2f} '.format(i,1.00),end='')
            s = temp[case['gallop_column']].sum()/temp.shape[0]
            print('{:>4.2%} '.format(s),end='')
            print(' välillä hevosia {}'.format(temp.shape[0]))
    
    
    
    def test():
        print('Hevosia testidatassa', Y_test.shape[0])
        for i in np.arange(0.01,1.0,0.01):
            
            temp = Y_test[(Y_test.prediction >= i) & (Y_test.prediction < (i+0.01))]
            if len(temp) > 0:
                s  = temp[case['gallop_column']].sum()/temp.shape[0]
                print('Ennusteväli: {:.2f}-{:.2f} laukka-% välillä: {:.2%} välillä hevosia {}'.format(i,i+0.01,s,temp.shape[0]))




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
    
    
    import matplotlib.pyplot as plt
    
    
    
    # create stacked errorbars:
    plt.figure(figsize=(10,10))
    plt.scatter(Y_test.prediction,Y_test.combinedDisqualified, s=10,marker='x')
    
    plt.grid('on')

#if __name__ == '__main__':
#    main()
