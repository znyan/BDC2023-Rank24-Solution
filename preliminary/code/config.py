import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="BDC 2023")

    parser.add_argument('--seed', type=int, default=2023)

    parser.add_argument('--data_dir', type=str, default='data/contest_data')
    parser.add_argument('--data_idx', type=str, default='0,1,2,3,4', help='sep by `,` eg. 0,1,2,3,4')
    parser.add_argument('--feat_dir', type=str, default='data/pretrain_model/')
    parser.add_argument('--model_dir', type=str, default='data/best_model/')
    
    parser.add_argument('--n_splits', type=int, default=5)

    return parser.parse_args()
