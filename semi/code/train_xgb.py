import os
import warnings

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb

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
    drop_features = ['id'] + diff
    labels = ['NO_FAULT',
        '3d82f4ad7f114cbdbd469fc897b001a1',
        '05be78908e3c4818b1caa00b71d8bb11',
        '6cc7dc7bb5fa4327a20c883ab00ab2fe',
        '8b18231981e0440488bbac370b1464cf',
        '2170e75abdf54178afcd5ffecb387eee',
        '4287f5cca47742008a8fb965908e5dea',
        '00c1ba198361424c9597328ea33d0d15',
        'c453a975de5148e4b1c47be258a646c9',
        '15b7f6577fec4c16b01ee2e053b1f201',
        'ea4bdf00441c4157a99a9c72bb7f4eb2',
        '14eb4630112b4ce9bd88d93104b4570e',
        '53f1acb37db941b8b9c77dfefecb157b',
        '8b3eee3cc4fe4568b5ba4125c1a4047f',
        'e97a387ed0204878b0660f0090bfacd6',
        'f1023ca9976e4a5eaaaaed244acd2f4a',
        '122ec12af3744773b9b04c6c8e929711',
        'faf90b12d1cf478e810172eb6aced658',
        '0d1304f1f40743dea03be55bca96c32b',
        '03d1f58da52d49dbb815cda9be061d25',
        '36c4ac32f7504f13b7aef941de9ecc81',
        'node-worker1',
        'node-worker2',
        'node-worker3']
    features = [c for c in train_data.columns if c not in drop_features + labels]
    print('features - ', len(features))
    
    oof_preds = np.zeros((len(train_data), len(labels)))
    test_preds = np.zeros((len(test_data), len(labels)))
    importance = pd.DataFrame({'features': features})
    
    X = train_data[features]
    dtest = xgb.DMatrix(test_data[features])
    mskf = MultilabelStratifiedKFold(n_splits=args.n_splits, random_state=args.seed, shuffle=True)
    if not os.path.exists(os.path.join(args.model_dir, 'xgb')):
        os.mkdir(os.path.join(args.model_dir, 'xgb'))
    for k, label in enumerate(labels):
        y = train_data[label]
        for i, (trn_idx, val_idx) in enumerate(mskf.split(train_data[features], train_data[labels])):
            X_trn, X_val = X.iloc[trn_idx, :], X.iloc[val_idx, :]
            y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]
            
            dtrn = xgb.DMatrix(X_trn, label=y_trn)
            dval = xgb.DMatrix(X_val, label=y_val)
            
            parameters = {
                'seed': args.seed,
                'objective': 'binary:logistic',
                'eta': 0.3,
                'eval_metric': 'auc',
                'tree_method': 'gpu_hist',
                'predictor': 'gpu_predictor',
                'gpu_id': 0,
                'nthread': args.num_threads,
                'n_jobs': args.num_threads,
            }
            
            model = xgb.train(parameters, dtrn, 2000, [(dval, 'eval')], early_stopping_rounds=25, verbose_eval=25)
            
            oof_preds[val_idx, k] = model.predict(dval, ntree_limit=model.best_ntree_limit)
            print(f'label {k + 1} / fold {i + 1} - auc {roc_auc_score(y_val, oof_preds[val_idx, k])}')
            # model.save_model(os.path.join(args.model_dir, f'xgb/label{k}_fold{i}.json'))
            
            test_preds[:, k] += model.predict(dtest) / args.n_splits
    
    print(f"* CV sScore - {np.mean(sScore(np.array(train_data[labels]), oof_preds, len(labels)))}")
    print(f"* NO_FAULT AUC Score - {roc_auc_score(train_data['NO_FAULT'], oof_preds[:, 0])}")
    
    submission = pd.DataFrame(test_preds, columns=labels)
    submission.index = test_data['id']
    submission.reset_index(inplace=True)
    submission = submission.melt(id_vars="id", value_vars=labels, value_name="score", var_name="source")
    submission.to_csv(os.path.join(args.result_dir, "results.csv"), index=False)
    
    return submission
   
           
if __name__ == '__main__':    
    args = parse_args()
    
    train_data = pd.read_parquet(os.path.join(args.feat_dir, 'train_data.parquet'), engine='fastparquet').sort_values(by='id')
    test_data = pd.read_parquet(os.path.join(args.feat_dir, 'test_data.parquet'), engine='fastparquet').sort_values(by='id')
    
    # train_drop = [c for c in train_data.columns if (('instance' in c) and ('job' not in c)) or (('instance' not in c) and ('job' in c))]
    # test_drop = [c for c in test_data.columns if (('instance' in c) and ('job' not in c)) or (('instance' not in c) and ('job' in c))]
    # train_data = train_data[[c for c in train_data.columns if c not in train_drop]]
    # test_data = test_data[[c for c in test_data.columns if c not in test_drop]]
    
    drop = []
    for c in test_data:
        if test_data[c].isna().mean() > 0.95:
            drop.append(c)
    
    train_data = train_data[[c for c in train_data.columns if c not in drop]]
    test_data = test_data[[c for c in test_data.columns if c not in drop]]
    
    run(args, train_data, test_data)
