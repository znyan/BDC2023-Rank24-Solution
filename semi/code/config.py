import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="BDC 2023")

    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--num_threads', type=int, default=64)

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--feat_dir', type=str, default='./model/tmp_data')
    parser.add_argument('--model_dir', type=str, default='./model/model_data')
    parser.add_argument('--result_dir', type=str, default='./result')
    parser.add_argument('--data_idx', type=str, default='1,2,3,4,5', help='sep by `,` eg. 0,1,2,3,4')
    
    
    parser.add_argument('--n_splits', type=int, default=5)
    
    
    return parser.parse_args()
