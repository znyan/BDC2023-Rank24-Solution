import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import lightgbm as lgb
import catboost as ctb
import xgboost as xgb
from config import parse_args


def run(args, train_data: pd.DataFrame, test_data: pd.DataFrame):
    set1, set2 = set(train_data.columns), set(test_data.columns)
    diff = list((set1 - (set1 & set2)) | (set2 - (set1 & set2)))
    drop_features = ['id'] + diff + ['service_name_sequence', 'host_ip_sequence', 'endpoint_name_sequence']
    labels = ['NO_FAULT',
              'LTE4MDk5Mzk2NjU1NjM1ODI0NDc=', 'LTcxMDU4NjY3NDcyMTgwNTc5MDE=', 'LTkyMDExNjM1MjY4NDg4ODU5Mjk=', 'NDExNzk3NjQ4ODg3NTY0OTQ3OA==', 'ODI4MTMxNDkzODEzNTg5OTE4Mg==',
              'node-worker1', 'node-worker2', 'node-worker3']
    features = [c for c in test_data.columns if c not in drop_features + labels]
    print('features -', len(features))
    
    
    test_preds_lgb = np.zeros((len(test_data), len(labels)))
    test_preds_ctb = np.zeros((len(test_data), len(labels)))
    test_preds_xgb = np.zeros((len(test_data), len(labels)))
    for k, label in tqdm(enumerate(labels)):
        for i in range(args.n_splits):
            lgb_model = lgb.Booster(model_file=os.path.join(args.model_dir, 'lgb', f'label{k}_fold{i}.txt'))
            test_preds_lgb[:, k] += lgb_model.predict(test_data[lgb_model.feature_name()]) / args.n_splits
            
            ctb_model = ctb.CatBoostClassifier()
            ctb_model.load_model(os.path.join(args.model_dir, 'ctb', f'label{k}_fold{i}.bin'))
            test_preds_ctb[:, k] += ctb_model.predict_proba(test_data[ctb_model.feature_names_])[:, 1] / args.n_splits
            
            xgb_model = xgb.Booster()
            xgb_model.load_model(os.path.join(args.model_dir, 'xgb', f'label{k}_fold{i}.json'))
            dtest = xgb.DMatrix(test_data[xgb_model.feature_names])
            test_preds_xgb[:, k] += xgb_model.predict(dtest) / args.n_splits
    
    test_preds = test_preds_lgb * 0.3 + test_preds_ctb * 0.3 + test_preds_xgb * 0.4
    submission = pd.DataFrame(test_preds, columns=labels)
    submission.index = test_data['id']
    submission.reset_index(inplace=True)
    submission = submission.melt(id_vars="id", value_vars=labels, value_name="score", var_name="source")
    submission.to_csv("./data/submission/submission.csv", index=False)
            
            
if __name__ == '__main__':
    args = parse_args()
    train_data = pd.read_csv(os.path.join(args.feat_dir, 'train_data.csv'), nrows=5).sort_values(by='id')
    test_data = pd.read_csv(os.path.join(args.feat_dir, 'test_data.csv')).sort_values(by='id')
    
    run(args, train_data, test_data)
