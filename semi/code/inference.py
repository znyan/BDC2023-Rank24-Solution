import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import xgboost as xgb
from config import parse_args


def run(args, test_data: pd.DataFrame):
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
    features = [c for c in test_data.columns if c not in labels]
    print('features -', len(features))
    
    test_preds_xgb = np.zeros((len(test_data), len(labels)))
    for k, label in tqdm(enumerate(labels)):
        for i in range(args.n_splits):
            xgb_model = xgb.Booster()
            xgb_model.load_model(os.path.join(args.model_dir, 'xgb', f'label{k}_fold{i}.json'))
            dtest = xgb.DMatrix(test_data[xgb_model.feature_names])
            test_preds_xgb[:, k] += xgb_model.predict(dtest) / args.n_splits
    
    test_preds = test_preds_xgb
    submission = pd.DataFrame(test_preds, columns=labels)
    submission.index = test_data['id']
    submission.reset_index(inplace=True)
    submission = submission.melt(id_vars="id", value_vars=labels, value_name="score", var_name="source")
    submission.to_csv(os.path.join(args.result_dir, 'results.csv'), index=False)
            
            
if __name__ == '__main__':
    args = parse_args()
    test_data = pd.read_parquet(os.path.join(args.feat_dir, 'test_data.parquet'), engine='fastparquet').sort_values(by='id')
    
    run(args, test_data)
