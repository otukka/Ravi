# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 00:31:38 2019

@author: VStore
"""

def get_predicted_position(case,pred):
    if case['form'] == 'regression':
        pred_reshaped = pred.reshape((-1,case['n_competitors']))

        predicted_order = np.argsort(np.argsort(pred_reshaped,axis=1),axis=1).ravel()

        y_true = Y_test['position'].values-1


        print(accuracy_score(y_true,predicted_order))


        # Lähtönumerokohtainen tarkkuus
        for i in np.arange(case['n_competitors']):
            print(accuracy_score(y_true[i::12],predicted_order[i::12]))


        # minkälaisia sijoituksia kullekin lähtöpaikalle ennustetaan vs oikea
        print(np.mean(np.argsort(np.argsort(pred_reshaped,axis=1),axis=1),axis=0))
        print(np.mean(y_true.reshape((-1,case['n_competitors'])),axis=0))


        df = pd.DataFrame({'startNumber':Y_test['startNumber'],'position':Y_test['position'],'prediction':pred,'predictedOrder':predicted_order})

