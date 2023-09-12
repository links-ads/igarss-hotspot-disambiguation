import pandas as pd
import numpy as np
import math
import os
import argparse

from datetime import datetime

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import xgboost as xgb
        
def plot_rforest_importance(forest, feature_names, filename):
    fig, ax = plt.subplots(figsize=(16, 10))
    importances = forest.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    forest_importances.plot.bar(ax=ax)
    ax.set_title("Feature importances")
    ax.set_ylabel("Mean decrease in impurity") 
    plt.savefig(filename, bbox_inches='tight')
    plt.close() 
    
def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("", flush=True)
            
def pipeline_mlp(trainval, test, save_path):
    
    params = {
    'hidden_layer_sizes': [(50,), (100,), (200,)],
    'solver': ['sgd'],
    'learning_rate': ['adaptive'],
    'early_stopping': [True],
    'activation': ['logistic', 'relu'],
    'alpha': [0.0001, 0.001, 0.01],
    'random_state': [42]
    }
    
    print("pipeline_mlp", flush=True)
    print(params, flush=True)
    
    y_trainval = trainval['is_positive']
    X_trainval = trainval.drop(columns=['is_positive'])
    
    y_test = test['is_positive']
    X_test = test.drop(columns=['is_positive'])
    
    model = MLPClassifier()
     
    # kfold cross validation
    kf_total = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=kf_total)
    gs.fit(X_trainval, y_trainval)
    model = gs.best_estimator_
    print(gs.best_params_, flush=True)
    
    # test
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).sum() / (len(y_test))
    prec = precision_score(y_test, y_pred, average=None)
    prec_binary = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred)
    print(f"\tPrecision_binary: {prec_binary} positive | {prec}", flush=True)
    print(f"\tBest: {acc} accuracy | {prec} prec | {rec} recall | {f1} f1", flush=True)
    report_best_scores(gs.cv_results_, 1)  
    return
    
def pipeline_logistic(trainval, test, params = {
        'penalty': ['l1','l2'], 
        'C': [0.001,0.01,0.1,1,10,100,1000],
        'random_state': [42],
        'n_jobs':[4]
        }):
    
    print("pipeline_logistic", flush=True)
    print(params, flush=True)
    
    y_trainval = trainval['is_positive']
    X_trainval = trainval.drop(columns=['is_positive'])
    
    y_test = test['is_positive']
    X_test = test.drop(columns=['is_positive'])
    
    model = LogisticRegression()
     
    # kfold cross validation
    kf_total = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=kf_total)
    gs.fit(X_trainval, y_trainval)
    model = gs.best_estimator_
    print(gs.best_params_, flush=True)
    
    # test
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).sum() / (len(y_test))
    prec = precision_score(y_test, y_pred, average=None)
    prec_binary = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred)
    print(f"\tPrecision_binary: {prec_binary} positive | {prec}", flush=True)
    print(f"\tBest: {acc} accuracy | {prec} prec | {rec} recall | {f1} f1", flush=True)
    report_best_scores(gs.cv_results_, 1)
    
    return
    
def pipeline_xgboost(trainval, test, save_path):
    params = {
                'colsample_bytree': [1], 
                'learning_rate': [0.1], 
                'max_depth': [12], 
                'n_estimators': [1000], 
                'nthread': [8], 
                'seed': [42], 
                'subsample': [0.8],
                'scale_pos_weight': [9],
            }
    
    params["scale_pos_weight"] = [len(trainval[trainval['is_positive'] == 0]) / len(trainval[trainval['is_positive'] == 1])]
    
    print("pipeline_xgboost", flush=True)
    print(params, flush=True)
    
    y_trainval = trainval['is_positive']
    X_trainval = trainval.drop(columns=['is_positive'])
    
    y_test = test['is_positive']
    X_test = test.drop(columns=['is_positive'])
    
    model = xgb.XGBClassifier()
     
    # kfold cross validation
    kf_total = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=kf_total)
    gs.fit(X_trainval, y_trainval)
    model = gs.best_estimator_
    print(gs.best_params_, flush=True)
    
    # test
    y_pred = model.predict(X_test)
    acc = (y_pred == y_test).sum() / (len(y_test))
    prec = precision_score(y_test, y_pred, average=None)
    prec_binary = precision_score(y_test, y_pred, average='binary')
    rec = recall_score(y_test, y_pred, average=None)
    f1 = f1_score(y_test, y_pred)
    print(f"\tPrecision_binary: {prec_binary} positive | {prec}", flush=True)
    print(f"\tBest: {acc} accuracy | {prec} prec | {rec} recall | {f1} f1", flush=True)
    report_best_scores(gs.cv_results_, 1)
    
    plot_rforest_importance(model, X_trainval.columns, os.path.join(save_path, "xgboost_importance.png"))
    
    return
    
def pipeline_rforest(trainval, test, save_path):
    
    EnsembleParams={
        "Bagging":{
            "max_samples":[0.25, 0.5, 0.75, 1.0],
            "n_estimators":[10, 100, 1000]
        },
        "RandomForest":{
            "estimators":[10, 100, 1000]
        }
    }
    
    DecisionTreeParams = {
        "WithoutPruning": {
            'criterion': ['gini', 'entropy'],
            'min_samples_split': [ 0.01, 0.1, 0.25],
            'min_samples_leaf': [ 0.01, 0.1, 0.25]
        },
        "WithPruning": {
            'criterion': ['gini', 'entropy'],
            'ccp_alpha': [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]
        }
    }
    
    y_trainval = trainval['is_positive']
    X_trainval = trainval.drop(columns=['is_positive'])
    
    y_test = test['is_positive']
    X_test = test.drop(columns=['is_positive'])
    
    for model,name in zip([BaggingClassifier(estimator=RandomForestClassifier()), RandomForestClassifier()], ['Bagging', 'RandomForest']):
        for pruning in DecisionTreeParams:
            
            params = {
                'random_state': [42],
                'n_jobs': [8]
            }
            params = dict(params, **EnsembleParams[name])
            
            if name == "Bagging":
                decisionTreeParams = {}
                for key in DecisionTreeParams[pruning]:
                    decisionTreeParams[f'estimator__{key}'] = DecisionTreeParams[pruning][key]
            else:
                decisionTreeParams = DecisionTreeParams[pruning]
            params = dict(params, **decisionTreeParams)
            
            print(f"\n{name} {pruning}", flush=True)
            print(params, flush=True)
            
            # kfold cross validation
            kf_total = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            gs = GridSearchCV(estimator=model, param_grid=params, scoring='f1', cv=kf_total)
            gs.fit(X_trainval, y_trainval)
            model = gs.best_estimator_
            print(gs.best_params_, flush=True)
            
            # test
            y_pred = model.predict(X_test)
            acc = (y_pred == y_test).sum() / (len(y_test))
            prec = precision_score(y_test, y_pred, average=None)
            prec_binary = precision_score(y_test, y_pred, average='binary')
            rec = recall_score(y_test, y_pred, average=None)
            f1 = f1_score(y_test, y_pred)
            print(f"\tPrecision_binary: {prec_binary} positive | {prec}", flush=True)
            print(f"\tBest: {acc} accuracy | {prec} prec | {rec} recall | {f1} f1", flush=True)
            plot_rforest_importance(model, list(trainval.columns), os.path.join(save_path, "rf_feature_importance.png") )
            model.save(os.path.join(save_path, f"{name}{key}.joblib"))
    
    return     
    
   
def model_experiments(pipeline_func,
                      use_features=[],
                      dataset_file='path/to/dataset',
                      test_list_ids=range(0,4),
                      trainval_list_ids=range(4,171),
                      save_path='path/to/save/results',
                      norm=False,
                      oversample=False,
                      undersample=False):
    
    dataset = pd.read_csv(dataset_file)
    
    
    test_ids = []
    for i in test_list_ids:
        test_ids += list(pd.read_csv(os.path.join(dataset_file, 'subpath/to/id_lists/_{i}.csv'), names=['id'])['id'])
        
    trainval_ids = []
    for i in trainval_list_ids:
        trainval_ids += list(pd.read_csv(os.path.join(dataset_file, 'subpath/to/id_lists/_{i}.csv'), names=['id'])['id'])
    
    traintestval = dataset[dataset['id'].isin(test_ids + trainval_ids)]
    if(norm):
        scaler = StandardScaler()
        
        ids = list(traintestval['id'].copy(deep=True))
        labels = list(traintestval['is_positive'].copy(deep=True))
        
        features_df = traintestval[use_features]
        normalized_data = scaler.fit_transform(features_df)
        normalized_df = pd.DataFrame(normalized_data, columns=use_features)
        normalized_df['id'] = ids
        normalized_df["is_positive"] = labels
        
        
        
        test = normalized_df[normalized_df['id'].isin(test_ids)][use_features+['is_positive']]
        trainval = normalized_df[normalized_df['id'].isin(trainval_ids)][use_features+['is_positive']]
    else:
        test = traintestval[traintestval['id'].isin(test_ids)][use_features+['is_positive']]
        trainval = traintestval[traintestval['id'].isin(trainval_ids)][use_features+['is_positive']]
    
    print(trainval.head(), flush=True)
    
    if(oversample):
        positive = trainval[trainval["is_positive"] == 1]
        positive_oversampled = resample(positive,
                                replace=True,    
                                n_samples=len(positive),
                                random_state=42)
        trainval = pd.concat([trainval, positive_oversampled])
    if(undersample):
        negative = trainval[trainval["is_positive"] == 0]
        negative_undersampled = resample(negative,
                                replace=True,    
                                n_samples=len(trainval[trainval["is_positive"] == 1]),
                                random_state=42)
        trainval = pd.concat([trainval[trainval["is_positive"] == 1], negative_undersampled])
    
    print("Beginning of experiments ...", flush=True)
    pipeline_func(trainval, test, save_path)
    print("End of experiments", flush=True)
    
def interval_type(interval):
    try:
        start, end = interval.split('-')
        return range(int(start), int(end)+1)
    except:
        raise argparse.ArgumentTypeError("The format has to be start-end")
    
def bool_type(val):
    try:
        return val.lower() == 'true'
    except:
        raise argparse.ArgumentTypeError("The format has to be true/false")

parser = argparse.ArgumentParser()
# dataset args
parser.add_argument('--data_file', type=str, help="Dataset file", required=True)
parser.add_argument('--save_path', type=str, help="Folder to save results and models", required=True)
parser.add_argument('--train_ids', type=interval_type, help="List of train ids", required=False, default="0-34")
parser.add_argument('--test_ids', type=interval_type, help="List of test ids", required=False, default="35-170")
# data sources args
parser.add_argument('--mv', type=bool_type, help='Use MODIS/VIIRS', required=False, default='True')
parser.add_argument('--lc', type=bool_type, help='Use landcover', required=False, default='False')
parser.add_argument('--s3', type=bool_type, help='Use Sentinel 3', required=False, default='False')
parser.add_argument('--hprev', type=bool_type, help='Use # previous hotspots', required=False, default='False')
# model args
parser.add_argument('--xgb', type=bool_type, help='Use xgboost', required=False, default='False')
parser.add_argument('--mlp', type=bool_type, help='Use MLP', required=False, default='False')
parser.add_argument('--log', type=bool_type, help='Use logistic regression', required=False, default='False')
# get preprocessing args
parser.add_argument('--norm', type=bool_type, help='Normalize data', required=False, default='False')
parser.add_argument('--over', type=bool_type, help='Random oversampling for positive hotspots', required=False, default='False')
parser.add_argument('--under', type=bool_type, help='Random undersampling for negative hotspots', required=False, default='False')
if __name__ == '__main__':
    args = parser.parse_args()
    print(f"[{datetime.now()}]", flush=True)
    print(args, flush=True)
        
    features = []
    if(args.mv):
        features += ["frp","t_21","t_31","t_m13","t_m15","t_i4","t_i5"]
    if(args.lc):
        features += ["LC"]
    if(args.s3):
        features += ["S3_float32_OLCI_0","S3_float32_OLCI_1","S3_float32_OLCI_2","S3_float32_OLCI_3","S3_float32_OLCI_4","S3_float32_OLCI_5","S3_float32_OLCI_6","S3_float32_OLCI_7","S3_float32_OLCI_9","S3_float32_OLCI_10","S3_float32_OLCI_11","S3_float32_OLCI_12","S3_float32_OLCI_13","S3_float32_OLCI_14","S3_float32_OLCI_15","S3_float32_OLCI_16","S3_float32_OLCI_17","S3_float32_OLCI_18","S3_float32_OLCI_19","S3_float32_OLCI_20","S3_float32_SLSTR_reflectance_4","S3_float32_SLSTR_reflectance_5","S3_float32_SLSTR_brightness_temperature_0","S3_float32_SLSTR_brightness_temperature_1","S3_float32_SLSTR_brightness_temperature_2","S3_float32_SLSTR_brightness_temperature_3","S3_float32_SLSTR_brightness_temperature_4"]
    if(args.hprev):
        features += ["count_12h","count_24h","count_36h"]
        
    print(features, flush=True)
    
    dataset_file = args.data_file
    
    # launch experiments
    if(args.xgb):
        model_experiments(pipeline_xgboost, use_features=features, norm=args.norm, dataset_file = dataset_file, test_list_ids=args.test_ids, trainval_list_ids=args.train_ids, oversample=args.over, undersample=args.under, save_path=args.save_path)
    if(args.mlp):
        model_experiments(pipeline_mlp, use_features=features, norm=args.norm, dataset_file = dataset_file, test_list_ids=args.test_ids, trainval_list_ids=args.train_ids, oversample=args.over, undersample=args.under, save_path=args.save_path)
    if(args.log):
        model_experiments(pipeline_logistic, use_features=features, norm=args.norm, dataset_file = dataset_file, test_list_ids=args.test_ids, trainval_list_ids=args.train_ids, oversample=args.over, undersample=args.under, save_path=args.save_path)
        
    print(f"[{datetime.now()}]", flush=True)