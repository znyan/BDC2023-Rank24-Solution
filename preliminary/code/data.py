import os
import warnings

import threading
import multiprocessing
from tqdm import tqdm

import random
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
from sklearn.decomposition import TruncatedSVD

from config import parse_args


warnings.filterwarnings('ignore')

class T:
    labels = [  
        'LTE4MDk5Mzk2NjU1NjM1ODI0NDc=', 'LTcxMDU4NjY3NDcyMTgwNTc5MDE=', 'LTkyMDExNjM1MjY4NDg4ODU5Mjk=','NDExNzk3NjQ4ODg3NTY0OTQ3OA==',
        'ODI4MTMxNDkzODEzNTg5OTE4Mg==', 'node-worker1', 'node-worker2', 'node-worker3'
    ]
    idx_to_label = {k: v for k, v in enumerate(labels)}
    label_to_idx = {v: k for k, v in idx_to_label.items()}
    
    trace_service_names = [  # 'LTE5NDEyNzM0Nzc0NDcxMzM1MjI='
        'LTE2Nzc3NzQ3OTMwOTk3NTY1MTI=', 'LTE4MDk5Mzk2NjU1NjM1ODI0NDc=',                                 'LTI2NDk2NjgzMzM4MDU3NTMzMQ==',
        'LTI5MjAwNzM1OTUyMDQ3MTYxNzg=', 'LTM0NjY2MjE2NDUyNzM5Nzg0NjY=', 'LTM1ODcwMzQ2Njg1Nzg4MTM3ODU=', 'LTQ3NTg1MzE0NjMzMjE3NzYyNzg=',
        'LTQ4MzU0NjU1NzUwMzg4MjM0Njc=', 'LTU4NzY2NjAyMzY3ODY0OTk0NTI=', 'LTU5MjA5MjM1NjYzMzAwMzY0NDA=', 'LTY3Mzg5NzE2ODY2NzU5MzM0MDE=',
        'LTc1ODUwNjQyNDIxOTk3NjA3MDQ=', 'LTc3NjI2MDIyMDI0MTc3NDA4MjY=', 'LTcxMDU4NjY3NDcyMTgwNTc5MDE=', 'LTcyMTg1NzU1MzE2ODIxMTU1NDA=',
        'LTk4OTAyNzg2MTE1MTM4OTQyMw==', 'LTk5MjA5NDQwODc4ODk5NjQ5Mw==', 'LTkyMDExNjM1MjY4NDg4ODU5Mjk=', 'MTI3NjQ3MzcxODEzNjUxMDczNw==',
        'MTgxMzI5MDI4OTEyNDU1NjQ4NQ==', 'NDExNzk3NjQ4ODg3NTY0OTQ3OA==', 'NDMyMjcxNTcxODI4ODUzMDU4NQ==', 'NDUyMDE5OTQ0ODQxOTE2Mzk2NQ==',
        'NjAxMTE1OTUxNzYyOTc2MTU3Mg==', 'NzcwODAzNzA4OTczMTczMzkzNg==', 'ODA3MTAxMTA2MzI2NjM4NjY2NA==', 'ODE2MDA0NjYxMjY1Mjk1NjQ4MA==',
        'ODI4MTMxNDkzODEzNTg5OTE4Mg==', 'ODIxNTUzODY4NTY1MzQ3MDUyMA==', 'ODczNDg0OTI2NDI2MTUxNDU5Ng=='
    ]
    idx_to_trace_service_name = {k: v for k, v in enumerate(trace_service_names)}
    trace_service_name_to_idx = {v: k for k, v in idx_to_trace_service_name.items()}
    
    trace_host_ips = [
        '10.244.1.10', '10.244.1.11', '10.244.1.12', '10.244.1.13', '10.244.1.14', '10.244.1.15', '10.244.1.16', '10.244.1.17',
        '10.244.1.18', '10.244.1.19', '10.244.1.20', '10.244.1.21', '10.244.1.22', '10.244.1.23', '10.244.1.24', '10.244.1.25',
        '10.244.1.27', '10.244.1.28', '10.244.1.29', '10.244.1.30', '10.244.1.31', '10.244.1.37', '10.244.1.42', '10.244.2.10',
        '10.244.2.11', '10.244.2.13', '10.244.2.14', '10.244.2.15', '10.244.2.16', '10.244.2.17', '10.244.2.18', '10.244.2.19',
        '10.244.2.20', '10.244.2.21', '10.244.2.22', '10.244.2.23', '10.244.2.24', '10.244.2.25', '10.244.2.27', '10.244.2.31',
        '10.244.3.10', '10.244.3.11', '10.244.3.12', '10.244.3.13', '10.244.3.14', '10.244.3.15', '10.244.3.16', '10.244.3.17',
        '10.244.3.18', '10.244.3.19', '10.244.3.20', '10.244.3.21', '10.244.3.22', '10.244.3.23', '10.244.3.24', '10.244.3.25',
        '10.244.3.26', '10.244.3.27', '10.244.3.28', '10.244.3.31', '10.244.3.34', '10.244.3.35'
    ]
    idx_to_trace_host_ip = {k: v for k, v in enumerate(trace_host_ips)}
    trace_host_ip_to_idx = {v: k for k, v in idx_to_trace_host_ip.items()}
    
    def transfer_endpoint_name(endpoint_name):
        prefix = endpoint_name.split('/')[0]
        if prefix == 'DELETE:':
            return '/'.join(endpoint_name.split('/')[0:5])
        elif prefix == 'GET:':
            return '/'.join(endpoint_name.split('/')[0:5])
        else:  # 'POST:' 'HikariCP' 'Mysql'
            return endpoint_name
            
    trace_endpoint_names = [
        'DELETE:/api/v1/adminuserservice/users', 'DELETE:/api/v1/userservice/users',
        'GET:/api/v1/adminrouteservice/adminroute', 'GET:/api/v1/verifycode/generate', 'GET:/api/v1/verifycode/verify',
        'HikariCP/Connection/close', 'HikariCP/Connection/getConnection',
        'Mysql/JDBI/Connection/close', 'Mysql/JDBI/PreparedStatement/executeQuery', 'Mysql/JDBI/Statement/execute',  'Mysql/JDBI/Statement/executeQuery',
        'POST:/api/v1/adminbasicservice/adminbasic/contacts', 'POST:/api/v1/adminbasicservice/adminbasic/stations',
        'POST:/api/v1/adminrouteservice/adminroute', 'POST:/api/v1/admintravelservice/admintravel',
        'POST:/api/v1/adminuserservice/users', 'POST:/api/v1/users/login', 'POST:/api/v1/userservice/users/register'
    ]  
    idx_to_trace_endpoint_name = {k: v for k, v in enumerate(trace_endpoint_names)}
    trace_endpoint_name_to_idx = {v: k for k, v in idx_to_trace_endpoint_name.items()}
    
    log_services = [
        'LTE2Nzc3NzQ3OTMwOTk3NTY1MTI=', 'LTE4MDk5Mzk2NjU1NjM1ODI0NDc=', 'LTE5NDEyNzM0Nzc0NDcxMzM1MjI=', 'LTE5Nzg1ODAwNDM1MDA5MjAwNDI=',
        'LTI2NDk2NjgzMzM4MDU3NTMzMQ==', 'LTI3Nzc1OTY5MDU5MjE2OTUxMzg=', 'LTI4ODExNDIzMjQ3MTc4NTMyMzY=', 'LTI5MjAwNzM1OTUyMDQ3MTYxNzg=',
        'LTI5NjU5NTY0MjI5OTM5Mjk2MTU=', 'LTM0NjY2MjE2NDUyNzM5Nzg0NjY=', 'LTM1ODcwMzQ2Njg1Nzg4MTM3ODU=', 'LTQ0ODYyMTk0NDM4NzYzOTUyMjA=',
        'LTQ3NTg1MzE0NjMzMjE3NzYyNzg=', 'LTQ4MzU0NjU1NzUwMzg4MjM0Njc=', 'LTU4NzY2NjAyMzY3ODY0OTk0NTI=', 'LTU5MjA5MjM1NjYzMzAwMzY0NDA=',
        'LTY3Mzg5NzE2ODY2NzU5MzM0MDE=', 'LTc0MDgwMjEzMzUzOTI3OTk2MjI=', 'LTc0NDY3MTM5NjE1ODEyMjU4NTY=', 'LTc1ODUwNjQyNDIxOTk3NjA3MDQ=',
        'LTc3NjI2MDIyMDI0MTc3NDA4MjY=', 'LTcxMDU4NjY3NDcyMTgwNTc5MDE=', 'LTcyMTg1NzU1MzE2ODIxMTU1NDA=', 'LTg5MDUwMDgwMjMyOTc3NTc3MDg=',
        'LTk4OTAyNzg2MTE1MTM4OTQyMw==', 'LTk5MjA5NDQwODc4ODk5NjQ5Mw==', 'LTkyMDExNjM1MjY4NDg4ODU5Mjk=', 'MTI3NjQ3MzcxODEzNjUxMDczNw==',
        'MTgxMzI5MDI4OTEyNDU1NjQ4NQ==', 'MzY0Mzc4ODExMjI1Mjk1NDg2NQ=='  'Mzc4MDAzMDYyNTE3OTQ3OTc0Mw==', 'MzgyNTg4MjA2MjU2MDQwMjUwNQ==',
        'NDE4OTIyNjA5NzE1OTg0OTc4Mw==', 'NDExNzk3NjQ4ODg3NTY0OTQ3OA==', 'NDMyMjcxNTcxODI4ODUzMDU4NQ==', 'NDUxMTQ0NTM2NDY0NDY0NjEzNA==',
        'NDUyMDE5OTQ0ODQxOTE2Mzk2NQ==', 'NTM2Nzc5ODg5MjA1NTU2MDkzMg==', 'NjAxMTE1OTUxNzYyOTc2MTU3Mg==', 'NjY4NTM0NjU1NzQ4NjUwODA1MA==',
        'Njc5NzEzNDUyNTQ3NDczNTgyMw==', 'NzcwODAzNzA4OTczMTczMzkzNg==', 'ODA3MTAxMTA2MzI2NjM4NjY2NA==', 'ODE2MDA0NjYxMjY1Mjk1NjQ4MA==',
        'ODI4MTMxNDkzODEzNTg5OTE4Mg==', 'ODIxNTUzODY4NTY1MzQ3MDUyMA==', 'ODU3Mjk4OTAwNTAxMTY4NjQ3', 'ODc3NDE3MjQzNDMwMTY3NjA4Mg==',
        'ODczNDg0OTI2NDI2MTUxNDU5Ng==', 'elasticsearch-58cd769777-778q4.log', 'elasticsearch-58cd769777-7cfcd.log', 'elasticsearch-58cd769777-bml5l.log',
        'elasticsearch-58cd769777-clwcq.log', 'elasticsearch-58cd769777-dt7tz.log', 'elasticsearch-58cd769777-gwmhm.log', 'elasticsearch-58cd769777-j4fb9.log',
        'elasticsearch-58cd769777-k8mn7.log', 'elasticsearch-58cd769777-phx22.log', 'elasticsearch-58cd769777-pq274.log', 'elasticsearch-58cd769777-xn2j8.log',
        'nacosdb-mysql-0-xenon.log', 'nacosdb-mysql-1-xenon.log', 'nacosdb-mysql-2-xenon.log', 'node-worker1',
        'node-worker2', 'node-worker3']
    idx_to_log_service = {k: v for k, v in enumerate(log_services)}
    log_service_to_idx = {v: k for k, v in idx_to_log_service.items()}
    

def group_norm(df):
    _mean = df['value'].mean()
    _std = df['value'].std()

    df["value"] = df["value"].apply(lambda x: (x - _mean) / _std)
    
    return df


def extract_features(log: pd.DataFrame, metric: pd.DataFrame, trace: pd.DataFrame) -> dict:    
        
    feat = {}
    
    # if len(metric) == 52000:
    #     feat['metric_length'] = 1
        
        # metric['tags'] = metric['tags'].apply(lambda x: json.loads(x)['service'])
        
        # for stats_func in ['std', 'skew']:
        #     for k in T.labels:
        #         feat[f'metric_service{T.label_to_idx[k]}_value_{stats_func}'] = 0
            
        #     for k, v in metric.groupby('tags')['value'].agg(stats_func).to_dict().items():
        #         if k in T.labels:
        #             feat[f'metric_service{T.label_to_idx[k]}_value_{stats_func}'] = v


    # else:
    #     feat['metric_length'] = 0
    
    # log
    if len(log) > 0:
        feat['log_length'] = len(log)

        log['message_length'] = log['message'].apply(lambda x: len(x))
        for t in ['INFO', 'ERROR', 'WARN', 'DEBUG']:
            log[f'{t}_count'] = log['message'].str.count(t)
            feat[f'log_{t}_count'] = log[f'{t}_count'].mean()
            
        feat['log_service_nunique'] = log['service'].nunique()
        feat['message_length_std'] = log['message'].fillna("").map(len).std()
        feat['message_length_ptp'] = log['message'].fillna("").map(len).agg('ptp')
        feat['log_info_length'] = log['message'].map(lambda x:x.split("INFO")).map(len).agg('ptp')

        text_list = ['user','mysql']
        for text in text_list:
              feat[f'message_{text}_mean'] = log['message'].str.contains(text, case=False).mean()
        feat[f'message_mysql_mean'] = 1 if feat[f'message_mysql_mean'] > 0 else 0
    
        
        # groupby: service
        log_groupby_service = log.groupby('service')

        for k, v in log_groupby_service.apply(lambda x: len(x)).to_dict().items():
            if k in T.log_services:
                feat[f'service{T.log_service_to_idx[k]}_count_log1p'] = np.log1p(v)
        
        ## message_length: mean, std, skew, kurt, min, max, ptp, autocorr
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in log_groupby_service['message_length'].agg(stats_func).to_dict().items():
                if k in T.log_services:
                    feat[f'service{T.log_service_to_idx[k]}_message_length_{stats_func.__name__}'] = v
    
        ## error_type: mean
        for stats_func in [pd.Series.mean]:
            for t in ['INFO', 'ERROR', 'WARN', 'DEBUG']:
                for k, v in log_groupby_service[f'{t}_count'].agg(stats_func).to_dict().items():
                    if k in T.log_services:
                        feat[f'service{T.log_service_to_idx[k]}_{t}_{stats_func.__name__}'] = v
        
        ## timestamp: std, skew, kurt, autocorr
        for stats_func in [pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in log_groupby_service['timestamp'].agg(stats_func).to_dict().items():
                if k in T.log_services:
                    feat[f'service{T.log_service_to_idx[k]}_timestamp_{stats_func.__name__}'] = v
        
        ## timestamp_diff: min, max, ptp, mean, std, skew, kurt, autocorr
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in log_groupby_service.apply(lambda x: x['timestamp'].diff().agg(stats_func)).to_dict().items():
                if k in T.log_services:
                    feat[f'service_name{T.log_service_to_idx[k]}_timestamp_diff_{stats_func.__name__}'] = v


        # feat['log_length'] = len(log)

        # feat['log_timestamp_std'] = log['timestamp'].std()
        # feat['log_timestamp_skew'] = log['timestamp'].skew()
        # feat['log_timestamp_kurt'] = log['timestamp'].kurt()
        
        # for stats_func in ['std', 'skew']:
        #     for k, v in log.groupby('service')['timestamp'].agg(stats_func).to_dict().items():
        #         if k in T.labels:
        #             feat[f'log_service{T.label_to_idx[k]}_timestamp_{stats_func}'] = v

    else:
        feat['log_length'] = 0
    
    # trace
    
    # for stats_func in [pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
    #     for k in range(len(T.trace_service_names)):
    #         feat[f'service_name{k}_timestamp_{stats_func.__name__}'] = 0
            
    # for stats_func in [pd.Series.mean, pd.Series.median, pd.Series.std, pd.Series.max, pd.Series.min, np.ptp]:
    #     for k in range(len(T.trace_service_names)):
    #         feat[f'service_name{k}_cost_time_{stats_func.__name__}'] = 0
            
    # for k in range(len(T.trace_service_names)):
    #     feat[f'service_name{k}_status_code_error_count'] = 0

    if len(trace) > 0:
        feat['trace_length'] = len(trace)
        
        trace = trace.sort_values(by='timestamp')
        trace['cost_time'] = trace['end_time'] - trace['start_time']
        trace['endpoint_name'] = trace['endpoint_name'].apply(T.transfer_endpoint_name)
        
        # sequence: service_name, host_ip, endpoint_name
        feat[f'service_name_sequence'] = ' '.join(trace['service_name'].map(T.trace_service_name_to_idx).dropna().astype(int).astype(str).tolist())
        feat[f'host_ip_sequence'] = ' '.join(trace['host_ip'].map(T.trace_host_ip_to_idx).dropna().astype(int).astype(str).tolist())
        feat[f'endpoint_name_sequence'] = ' '.join(trace['endpoint_name'].map(T.trace_endpoint_name_to_idx).dropna().astype(int).astype(str).tolist())
            
        # cost_time: mean, median, std, min, max, ptp
        for stats_func in [pd.Series.mean, pd.Series.median, pd.Series.std, pd.Series.min, pd.Series.max, np.ptp]:
            feat[f'cost_time_{stats_func.__name__}'] = stats_func(trace['cost_time'])

        # groupby: service_name
        trace_groupby_service_name = trace.groupby('service_name')
        
        ## nunique: host_ip, endpoint_name
        for t in ['host_ip', 'endpoint_name']:
            for k, v in trace_groupby_service_name[t].nunique().to_dict().items():
                if k in T.trace_service_names:
                    feat[f'service_name{T.trace_service_name_to_idx[k]}_{t}_nunique'] = v
                
        ## count: log1p
        for k, v in trace_groupby_service_name.apply(lambda x: len(x)).to_dict().items():
            if k in T.trace_service_names:
                feat[f'service_name{T.trace_service_name_to_idx[k]}_count_log1p'] = np.log1p(v)
        
        ## timestamp: std, skew, kurt, autocorr
        for stats_func in [pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in trace_groupby_service_name['timestamp'].agg(stats_func).to_dict().items():
                if k in T.trace_service_names:
                    feat[f'service_name{T.trace_service_name_to_idx[k]}_timestamp_{stats_func.__name__}'] = v
        
        ## cost_time mean, std, min, max, ptp
        for stats_func in [pd.Series.mean, pd.Series.std, pd.Series.min, pd.Series.max, np.ptp]:
            for k, v in trace_groupby_service_name['cost_time'].agg(stats_func).to_dict().items():
                if k in T.trace_service_names:
                    feat[f'service_name{T.trace_service_name_to_idx[k]}_cost_time_{stats_func.__name__}'] = v

        ## status_code_error: count
        for k, v in trace_groupby_service_name.apply(lambda x: len(x[x['status_code'] != 200])).to_dict().items():
            if k in T.trace_service_names:
                feat[f'service_name{T.trace_service_name_to_idx[k]}_status_code_error_count'] = v
        
        ## start_time_diff, end_time_diff, cost_time_diff: min, max, ptp, mean, std, skew, kurt, autocorr
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            # for k, v in trace_groupby_service_name.apply(lambda x: stats_func(x['timestamp'].sort_values().diff().fillna(0))).dropna().to_dict().items():
            #     feat[f'service_name{T.trace_service_name_to_idx[k]}_timestamp_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_service_name.apply(lambda x: x['cost_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_service_names:
                    feat[f'service_name{T.trace_service_name_to_idx[k]}_cost_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_service_name.apply(lambda x: x['start_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_service_names:
                    feat[f'service_name{T.trace_service_name_to_idx[k]}_start_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_service_name.apply(lambda x: x['end_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_service_names:
                    feat[f'service_name{T.trace_service_name_to_idx[k]}_end_time_diff_{stats_func.__name__}'] = v
            
            
        # groupby: host_ip
        trace_groupby_host_ip = trace.groupby('host_ip')
        
        ## nunique: service_name, endpoint_name
        for t in ['service_name', 'endpoint_name']:
            for k, v in trace_groupby_host_ip[t].nunique().to_dict().items():
                if k in T.trace_host_ips:
                    feat[f'service_name{T.trace_host_ip_to_idx[k]}_{t}_nunique'] = v
        
        ## count: log1p
        for k, v in trace_groupby_host_ip.apply(lambda x: len(x)).to_dict().items():
            if k in T.trace_host_ips:
                feat[f'host_ip{T.trace_host_ip_to_idx[k]}_count_log1p'] = np.log1p(v)
        
        ## timestamp: std, skew, kurt, autocorr
        for stats_func in [pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in trace_groupby_host_ip['timestamp'].agg(stats_func).to_dict().items():
                if k in T.trace_host_ips:
                    feat[f'host_ip{T.trace_host_ip_to_idx[k]}_timestamp_{stats_func.__name__}'] = v
        
        ## cost_time sum, mean, std, min, max, ptp
        for stats_func in [pd.Series.sum, pd.Series.mean, pd.Series.std, pd.Series.min, pd.Series.max, np.ptp]:
            for k, v in trace_groupby_host_ip['cost_time'].agg(stats_func).to_dict().items():
                if k in T.trace_host_ips:
                    feat[f'host_ip{T.trace_host_ip_to_idx[k]}_cost_time_{stats_func.__name__}'] = v

        ## status_code_error: count
        for k, v in trace_groupby_host_ip.apply(lambda x: len(x[x['status_code'] != 200])).to_dict().items():
            if k in T.trace_host_ips:
                feat[f'host_ip{T.trace_host_ip_to_idx[k]}_status_code_error_count'] = v
        
        ## start_time_diff, end_time_diff, cost_time_diff: min, max, ptp, std, skew, kurt, autocorr
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in trace_groupby_host_ip.apply(lambda x: x['cost_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_host_ips:
                    feat[f'host_ip{T.trace_host_ip_to_idx[k]}_cost_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_host_ip.apply(lambda x: x['start_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_host_ips:
                    feat[f'host_ip{T.trace_host_ip_to_idx[k]}_start_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_host_ip.apply(lambda x: x['end_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_host_ips:
                    feat[f'host_ip{T.trace_host_ip_to_idx[k]}_end_time_diff_{stats_func.__name__}'] = v


        # groupby: endpoint_name
        trace_groupby_endpoint_name = trace.groupby('endpoint_name')
        
        ## nunique: service_name, host_ip
        for t in ['service_name', 'host_ip']:
            for k, v in trace_groupby_endpoint_name[t].nunique().to_dict().items():
                if k in T.trace_endpoint_names:
                    feat[f'service_name{T.trace_endpoint_name_to_idx[k]}_{t}_nunique'] = v
                
        ## count: log1p
        for k, v in trace_groupby_endpoint_name.apply(lambda x: len(x)).to_dict().items():
            if k in T.trace_endpoint_names:
                feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_count_log1p'] = np.log1p(v)
        
        ## timestamp: std, skew, kurt, autocorr
        for stats_func in [pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in trace_groupby_endpoint_name['timestamp'].agg(stats_func).to_dict().items():
                if k in T.trace_endpoint_names:
                    feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_timestamp_{stats_func.__name__}'] = v
        
        ## cost_time sum, mean, std, min, max, ptp
        for stats_func in [pd.Series.sum, pd.Series.mean, pd.Series.std, pd.Series.min, pd.Series.max, np.ptp]:
            for k, v in trace_groupby_endpoint_name['cost_time'].agg(stats_func).to_dict().items():
                if k in T.trace_endpoint_names:
                    feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_cost_time_{stats_func.__name__}'] = v

        ## status_code_error: count
        for k, v in trace_groupby_endpoint_name.apply(lambda x: len(x[x['status_code'] != 200])).to_dict().items():
            if k in T.trace_endpoint_names:
                feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_status_code_error_count'] = v
        
        ## start_time_diff, end_time_diff, cost_time_diff: min, max, ptp, std, skew, kurt, autocorr
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
            for k, v in trace_groupby_endpoint_name.apply(lambda x: x['cost_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_endpoint_names:
                    feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_cost_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_endpoint_name.apply(lambda x: x['start_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_endpoint_names:
                    feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_start_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_endpoint_name.apply(lambda x: x['end_time'].diff().agg(stats_func)).to_dict().items():
                if k in T.trace_endpoint_names:
                    feat[f'endpoint_name{T.trace_endpoint_name_to_idx[k]}_end_time_diff_{stats_func.__name__}'] = v
    else:
        feat['trace_length'] = 0

    return feat


def read_data_by_id(data_dir: str, idx: str, _id: str) -> dict:
    log, metric, trace = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    if os.path.exists(os.path.join(data_dir, idx, 'log', f'{_id}_log.csv')):
        log = pd.read_csv(os.path.join(data_dir, idx, 'log', f'{_id}_log.csv'))
    if os.path.exists(os.path.join(data_dir, idx, 'metric', f'{_id}_metric.csv')):
        metric = pd.read_csv(os.path.join(data_dir, idx, 'metric', f'{_id}_metric.csv'))
    if os.path.exists(os.path.join(data_dir, idx, 'trace', f'{_id}_trace.csv')):
        trace = pd.read_csv(os.path.join(data_dir, idx, 'trace', f'{_id}_trace.csv'))
        
    return extract_features(log=log, metric=metric, trace=trace)


def read_train_data_by_idx(data_dir: str, idx: str) -> pd.DataFrame:
    label = pd.read_csv(os.path.join(data_dir, idx, f'training_ground_truth_label_{idx}.csv'))
    label = pd.pivot(label, index='id', columns='source', values='score').reset_index()
    label = label.groupby('id').apply(lambda x: {k: list(v.values())[0] for k, v in x.to_dict().items()}).to_dict()
    
    log_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, idx, 'log'))]
    metric_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, idx, 'metric'))]
    trace_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, idx, 'trace'))]
    id_list = list(set(log_id_list) | set(metric_id_list) | set(trace_id_list))
    
    # multiprocessing
    pbar = tqdm(total=len(id_list))
        
    pool = multiprocessing.Pool(processes=8)
    labels = []
    result = []
    for _id in id_list:
        # labels.append(label[label['id'] == _id].iloc[0].to_dict())
        labels.append(label[_id])
        res = pool.apply_async(read_data_by_id, args=(data_dir, idx, _id,), callback=lambda x: pbar.update(1))
        result.append(res)
        
    pool.close()
    pool.join()
    
    # pool result
    records = []
    for i, res in enumerate(result):
        item = labels[i]
        item.update(res.get())
        records.append(item)
        
    return pd.DataFrame.from_records(records)


def read_train_data(data_dir: str, data_idx: str) -> pd.DataFrame:
    data = []
    for idx in data_idx.split(','): 
        data.append(read_train_data_by_idx(data_dir, idx))
    
    return pd.concat(data, ignore_index = True)
    
    
def read_test_data(data_dir: str) -> pd.DataFrame:
    log_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, 'testing', 'log'))]
    metric_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, 'testing', 'metric'))]
    trace_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, 'testing', 'trace'))]
    id_list = list(set(log_id_list) | set(metric_id_list) | set(trace_id_list))

    # multiprocessing
    pbar = tqdm(total=len(id_list))

    pool = multiprocessing.Pool(processes=8)
    labels = []
    result = []
    for _id in id_list:
        labels.append({'id': _id})
        res = pool.apply_async(read_data_by_id, args=(data_dir, 'testing', _id,), callback=lambda x: pbar.update(1))
        result.append(res)
        
    pool.close()
    pool.join()
    
    records = []
    for i, res in enumerate(result):
        item = labels[i]
        item.update(res.get())
        records.append(item)
        
    return pd.DataFrame.from_records(records)


if __name__ == '__main__':
    args = parse_args()
    
    train_data = read_train_data(args.data_dir, args.data_idx).sort_values(by='id')
    train_data.to_csv(os.path.join(args.feat_dir, 'train_data.csv'), index=False)

    test_data = read_test_data(args.data_dir).sort_values(by='id')
    test_data.to_csv(os.path.join(args.feat_dir, 'test_data.csv'), index=False)