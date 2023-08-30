import os
import warnings

import numpy as np
import pandas as pd

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from config import parse_args


def sScore(y_true, y_pred, num_classes: int):
    score = []
    for i in range(num_classes):
        score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
        
    return score

warnings.filterwarnings('ignore')
    

def run(args, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    set1, set2 = set(train_data.columns), set(test_data.columns)
    diff = list((set1 - (set1 & set2)) | (set2 - (set1 & set2)))
    drop_features = ['id'] + diff + ['service_name_sequence', 'host_ip_sequence', 'endpoint_name_sequence']
    labels = ['NO_FAULT',
              'LTE4MDk5Mzk2NjU1NjM1ODI0NDc=', 'LTcxMDU4NjY3NDcyMTgwNTc5MDE=', 'LTkyMDExNjM1MjY4NDg4ODU5Mjk=', 'NDExNzk3NjQ4ODg3NTY0OTQ3OA==', 'ODI4MTMxNDkzODEzNTg5OTE4Mg==',
              'node-worker1', 'node-worker2', 'node-worker3']
    features = [c for c in train_data.columns if c not in drop_features + labels]
    print('features - ', len(features))
    
    oof_preds = np.zeros((len(train_data), len(labels)))
    test_preds = np.zeros((len(test_data), len(labels)))
    importance = pd.DataFrame({'features': features})
    
    X = train_data[features]
    mskf = MultilabelStratifiedKFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)
    for k, label in enumerate(labels):
        y = train_data[label]
        for i, (trn_idx, val_idx) in enumerate(mskf.split(train_data[features], train_data[labels])):
            X_trn, X_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            
            dtrn = lgb.Dataset(X_trn, label=y_trn)
            dval = lgb.Dataset(X_val, label=y_val)
            
            parameters = {
                'random_state': args.seed,
                'learning_rate': 0.1,
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'num_leaves': 63,
                'feature_fraction': 0.5,
                'bagging_fraction': 0.9,
                'min_data_in_leaf': 32,
                'verbose': -1,
                'nthread': 8
            }
            
            lgb_model = lgb.train(
                parameters,
                dtrn,
                num_boost_round=2000,
                valid_sets=[dval],
                early_stopping_rounds=50,
                verbose_eval=-1,
            )
            
            oof_preds[val_idx, k] = lgb_model.predict(X_val[features], num_iteration=lgb_model.best_iteration)
            print(f'label {k + 1} / fold {i + 1} - auc {roc_auc_score(y_val, oof_preds[val_idx, k])}')
            lgb_model.save_model(f'model/lgb/label{k}_fold{i}.txt')
            # importance[f'gain_label{k + 1}_fold{i + 1}'] = lgb_model.feature_importance(importance_type='gain')
            
            test_preds[:, k] += lgb_model.predict(test_data[features], num_iteration=lgb_model.best_iteration) / args.n_splits
    
    print(f"* CV sScore - {np.mean(sScore(np.array(train_data[labels]), oof_preds, len(labels)))}")
    print(f"* NO_FAULT AUC Score - {roc_auc_score(train_data['NO_FAULT'], oof_preds[:, 0])}")
    
    # importance.to_csv("importance_lgb.csv", index=False)
    
    # submission = pd.DataFrame(test_preds, columns=labels)
    # submission.index = test_data['id']
    # submission.reset_index(inplace=True)
    # submission = submission.melt(id_vars="id", value_vars=labels, value_name="score", var_name="source")
    # submission.to_csv("submission_lgb.csv", index=False)
   
           
if __name__ == '__main__':    
    args = parse_args()
    train_data = pd.read_csv(os.path.join(args.feat_dir, 'train_data.csv')).sort_values(by='id')
    test_data = pd.read_csv(os.path.join(args.feat_dir, 'test_data.csv')).sort_values(by='id')

    # train_tfidf_svd = pd.read_csv('./train_tfidf_svd.csv')
    # test_tfidf_svd = pd.read_csv('./test_tfidf_svd.csv')

    # train_data = pd.merge(train_data, train_tfidf_svd, on='id')
    # test_data = pd.merge(test_data, test_tfidf_svd, on='id')

    run(args, train_data, test_data)
