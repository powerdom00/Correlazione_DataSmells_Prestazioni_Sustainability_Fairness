
import os
import pandas as pd
import csv
from sklearn.metrics import *
import math

'''
This function loads a dataset from disk and returns it as a DataFrame
'''
def open_dataset(path):
    df = pd.read_csv(path)
    return df

'''
This function saves a DataFrame as csv
'''
def save_dataset(df, path):
    df.to_csv(path, index_label='ID')

'''
 This function saves computed data as csv
'''
def to_csv(path, data, dataset):
    if dataset == "german":
        column_names = [
            'dataset','iteration','model','type','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff',
            'age_mean_diff','age_eq_opp_diff','age_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
        ]
    elif dataset == "adult":
        column_names = [
            'dataset','iteration','model','type','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff',
            'race_mean_diff','race_eq_opp_diff','race_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
        ]
    elif dataset == "mep":
        column_names = [
            'dataset','iteration','model','type','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff',
            'race_mean_diff','race_eq_opp_diff','race_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
        ]
    elif dataset == "compas":
        column_names = [
        'dataset','iteration','model','type','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff',
        'race_mean_diff','race_eq_opp_diff','race_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
        'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "bank-full":
        column_names = [
        'dataset','iteration','model','type','sol_name','age_mean_diff','age_eq_opp_diff','age_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "heart_disease":
        column_names = [
        'dataset','iteration','model','type','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "nursery":
        column_names = [
        'dataset','iteration','model','type','sol_name','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "student-por":
        column_names = [
        'dataset','iteration','model','type','sol_name','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "fpes":
        column_names = [
        'dataset','iteration','model','type','sol_name','race_mean_diff','race_eq_opp_diff','race_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "diabetic":
        column_names = [
        'dataset','iteration','model','type','sol_name','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff','race_mean_diff','race_eq_opp_diff','race_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]
    elif dataset == "speedDating":
        column_names = [
        'dataset','iteration','model','type','sol_name', 'age_mean_diff','age_eq_opp_diff','age_avg_odds_diff','sex_mean_diff','sex_eq_opp_diff','sex_avg_odds_diff','race_mean_diff','race_eq_opp_diff','race_avg_odds_diff','overall_mean_diff','overall_eq_opp_diff',
            'overall_avg_odds_diff','accuracy','f1_score','precision','recall','elapsed_time', 'size'
    ]


    exists = os.path.isfile(path)
    with open(path,'a') as f:
        writer = csv.DictWriter(f,fieldnames=column_names)
        if exists == False:
            writer.writeheader()
            writer.writerow(data)
        else:
            writer.writerow(data)

'''
This functions computes Precision, Accuracy, Recall and F1-score for a given ML model
'''
def validate(ml_model, X_test, y_test, results):

    pred = ml_model.predict(X_test)
    
    accuracy = ml_model.score(X_test,y_test)

    f1 = f1_score(y_test,pred,average='macro')

    precision = precision_score(y_test,pred,average='macro')

    recall = recall_score(y_test,pred,average='macro')

    results['accuracy'] = round(accuracy,3)
    results['f1_score'] = round(f1,3)
    results['precision'] = round(precision,3)
    results['recall'] = round(recall,3)

    return results


def get_mb_size(path):
    size = os.path.getsize(path)
    '''
    if size == 0:
       return "0B"
    size_name = ("B", "KB", "MB", "GB")
    i = int(math.floor(math.log(size, 1024)))
    p = math.pow(1024, i)
    s = round(size / p, 2)
    return "%s %s" % (s, size_name[i])
    '''
    return size/1024
   
   
   
   
