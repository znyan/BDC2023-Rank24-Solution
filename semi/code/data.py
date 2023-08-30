import os
import gc
import warnings
import random


import threading
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm

import re
import json

import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
# from sklearn.decomposition import TruncatedSVD

from config import parse_args

warnings.filterwarnings('ignore')
args = parse_args()

class T:
    drop_features = []
    
    _labels = [
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
        'node-worker3'
    ]
    
    _metric_level_0 = ['node_cpu_seconds_total']  # 0.04726
    _metric_level_1 = ['container_cpu_load_average_10s',  # 0.02274
                       'container_cpu_system_seconds_total',
                       'container_cpu_user_seconds_total', 
                       'container_cpu_usage_seconds_total']
    _metric_level_2 = ['node_network_transmit_colls_total',  # 0.02048
                       'node_network_receive_fifo_total',
                       'node_network_receive_frame_total',
                       'node_network_receive_packets_total',
                       'node_network_transmit_carrier_total',
                       'node_network_up',
                       'node_network_transmit_compressed_total',
                       'node_network_transmit_drop_total',
                       'node_network_receive_compressed_total',
                       'node_network_transmit_fifo_total',
                       'node_network_transmit_packets_total',
                       'node_network_transmit_errs_total',
                       'node_network_protocol_type',
                       'node_network_receive_bytes_total',
                       'node_network_address_assign_type',
                       'node_network_net_dev_group',
                       'node_network_mtu_bytes',
                       'node_network_info',
                       'node_network_iface_link_mode',
                       'node_network_iface_id',
                       'node_network_flags',
                       'node_network_dormant',
                       'node_network_device_id',
                       'node_network_carrier_changes_total',
                       'node_network_transmit_queue_length',
                       'node_network_carrier',
                       'node_network_receive_errs_total',
                       'node_network_receive_drop_total',
                       'node_network_transmit_bytes_total',
                       'node_network_receive_multicast_total',
                       'node_network_iface_link']
    _metric_level_3 = ['node_network_speed_bytes']  # 0.017618
    _metric_level_4 = ['container_network_transmit_packets_dropped_total',  # 0.01402
                       'container_network_transmit_packets_total', 
                       'container_network_receive_bytes_total',
                       'container_network_receive_errors_total',
                       'container_network_receive_packets_dropped_total',
                       'container_network_receive_packets_total',
                       'container_network_transmit_bytes_total',
                       'container_network_transmit_errors_total']
    _metric_level_5 = ['node_cpu_guest_seconds_total']  # 0.0118
    _metric_level_6 = ['container_cpu_cfs_throttled_seconds_total',  # 0.00859
                       'container_cpu_cfs_throttled_periods_total',
                       'container_cpu_cfs_periods_total']
    _metric_level_7 = ['node_disk_io_now',  # 0.00148
                       'node_disk_info',
                       'node_disk_io_time_seconds_total',
                       'node_disk_io_time_weighted_seconds_total',
                       'node_disk_read_bytes_total',
                       'node_disk_read_time_seconds_total',
                       'node_disk_reads_completed_total',
                       'node_disk_reads_merged_total',
                       'node_disk_write_time_seconds_total',
                       'node_disk_writes_completed_total',
                       'node_disk_writes_merged_total',
                       'node_disk_written_bytes_total']
    _metric_level_8 = ['resp_time',  # 0.00139
                       'error_count',
                       'cpm',
                       'success_rate']
    _metric_level_9 = ['node_memory_SUnreclaim_bytes',
                       'node_memory_Shmem_bytes',
                       'node_memory_HugePages_Rsvd',
                       'node_memory_HugePages_Free',
                       'node_memory_HardwareCorrupted_bytes',
                       'node_memory_Dirty_bytes',
                       'node_memory_DirectMap4k_bytes',
                       'node_memory_DirectMap2M_bytes',
                       'node_memory_Committed_AS_bytes',
                       'node_memory_CommitLimit_bytes',
                       'node_memory_CmaTotal_bytes',
                       'node_memory_CmaFree_bytes',
                       'node_memory_Cached_bytes',
                       'node_memory_Buffers_bytes',
                       'node_memory_Bounce_bytes',
                       'node_memory_AnonPages_bytes',
                       'node_memory_AnonHugePages_bytes',
                       'node_memory_Active_file_bytes',
                       'node_memory_Active_bytes',
                       'node_memory_Active_anon_bytes',
                       'node_load5',
                       'node_load15',
                       'node_load1',
                       'node_memory_HugePages_Surp',
                       'node_memory_HugePages_Total',
                       'node_memory_Hugepagesize_bytes',
                       'node_memory_Writeback_bytes',
                       'node_memory_Slab_bytes',
                       'node_memory_SwapCached_bytes',
                       'node_memory_SwapFree_bytes',
                       'node_memory_SwapTotal_bytes',
                       'node_memory_Unevictable_bytes',
                       'node_memory_VmallocChunk_bytes',
                       'node_memory_VmallocTotal_bytes',
                       'node_memory_VmallocUsed_bytes',
                       'node_memory_WritebackTmp_bytes',
                       'node_memory_SReclaimable_bytes',
                       'node_memory_Inactive_anon_bytes',
                       'node_memory_PageTables_bytes',
                       'node_memory_NFS_Unstable_bytes',
                       'node_memory_Mlocked_bytes',
                       'node_memory_MemTotal_bytes',
                       'node_memory_MemFree_bytes',
                       'node_memory_MemAvailable_bytes',
                       'node_memory_Mapped_bytes',
                       'node_memory_Inactive_file_bytes',
                       'node_memory_Inactive_bytes',
                       'node_memory_KernelStack_bytes']
    
#     metric_level_0 = _metric_level_8
#     metric_level_1 = _metric_level_2 + _metric_level_3 + _metric_level_7 + _metric_level_9  # 'instance', 'job', 'metric_name'
#     metric_
    
    
    metric_level_0 = _metric_level_0 + _metric_level_5  # 'cpu', 'instance', 'job', 'mode', 'metric_name'
    metric_level_1 = _metric_level_1 + _metric_level_4 + _metric_level_6  # 'container', 'image', 'instance', 'job', 'kubernetes_io_hostname', 'namespace', 'metric_name'
    metric_level_2 = _metric_level_2 + _metric_level_3 + _metric_level_7 + _metric_level_9  # 'instance', 'job', 'metric_name'
    metric_level_3 = _metric_level_8  # 'service_name', 'metric_name'
    
    log_message_pattern = re.compile('(?:.*\s)?([A-Z]+)\s.*\[(.*?)\]\s(.*):\s\[(.*?)\]\[(.*?)\].*')
    
    def transfer_host_ip(host_ip):
        return '.'.join(host_ip.split('.')[:-1])
    
    def transfer_endpoint_name(endpoint_name):
        prefix = endpoint_name.split('/')[0]
        if prefix == '':
            return '/'.join(endpoint_name.split('/')[0:-2]).replace(':', '').replace('/', '_')
        elif prefix == 'GET:':
            return '/'.join(endpoint_name.split('/')[1:-2]).replace(':', '').replace('/', '_') 
        elif prefix == 'POST:':
            return '/'.join(endpoint_name.split('/')[1:-2]).replace(':', '').replace('/', '_') 
        elif prefix == 'DELETE:':
            return '/'.join(endpoint_name.split('/')[0:-2]).replace(':', '').replace('/', '_')
        else:  # 'HikariCP' 'Mysql', 'PUT'
            return endpoint_name.replace(':', '').replace('/', '_')

def group_norm(df):
    _mean = df['value'].mean()
    _std = df['value'].std()

    df["value"] = df["value"].apply(lambda x: (x - _mean) / _std)
    
    return df


def read_csv_from_content(csv_content):
    csv_bytes = csv_content.read()
    csv_str = csv_bytes.decode('utf-8')
    csv_data = pd.read_csv(StringIO(csv_str))
    return csv_data

                
def extract_features(log: pd.DataFrame, metric: pd.DataFrame, trace: pd.DataFrame) -> dict:
    feat = {}
    
    if len(metric) > 0:
        feat['metric_length'] = len(metric)
        metric['tags'] = metric['tags'].apply(lambda x: eval(x))
        metric['metric_name'] = metric['tags'].apply(lambda x: x['metric_name'])
        
        for key in ['container', 'image', 'cpu', 'instance', 'job', 'mode', 'kubernetes_io_hostname', 'namespace', 'service_name']:
            metric[key] = metric['tags'].apply(lambda x: x[key] if key in x.keys() else 'NA')
        
        # groupby: metric_name
        metric_groupby_metric_name = metric.groupby('metric_name')
        
        ## value: min, max, ptp, mean, std, skew, kurt
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:
            for k, v in metric_groupby_metric_name.apply(lambda x: stats_func(x['value'])).to_dict().items():
                feat[f'metric_metric_name_{k}_value_{stats_func.__name__}'] = v
        ## value_diff: min, max, ptp, mean, std, skew, kurt
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
            for k, v in metric_groupby_metric_name.apply(lambda x: stats_func(x['value'].diff().fillna(0))).to_dict().items():
                feat[f'metric_metric_name_{k}_value_diff_{stats_func.__name__}'] = v
        
        
#         # groupby: metric_name, service_name
#         metric['service_name'] = metric['tags'].apply(lambda x: x['service_name'] if 'service_name' in x.keys() else 'NA')
#         metric_groupby_metric_name_service_name = metric[metric['service_name'] != 'NA'].groupby(['metric_name', 'service_name'])
#         if len(metric_groupby_metric_name_service_name) > 0:
#             for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
#                 for (k0, k1), v in metric_groupby_metric_name_service_name.apply(lambda x: stats_func(x['value'])).to_dict().items():
#                     feat[f'metric_metric_name_{k0}_service_name_{k1}_value_{stats_func.__name__}'] = v
                        
                        
#         # groupby: metric_name, instance
#         metric['instance'] = metric['tags'].apply(lambda x: x['instance'].replace('.', '_').replace(':', '_') if 'instance' in x.keys() else 'NA')
#         metric_groupby_metric_name_instance = metric[metric['instance'] != 'NA'].groupby(['metric_name', 'instance'])
#         if len(metric_groupby_metric_name_instance) > 0:
#             for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
#                 for (k0, k1), v in metric_groupby_metric_name_instance.apply(lambda x: stats_func(x['value'])).to_dict().items():
#                     feat[f'metric_metric_name_{k0}_instance_{k1}_value_{stats_func.__name__}'] = v
                    
                    
#         # groupby: metric_name, job
#         metric['job'] = metric['tags'].apply(lambda x: x['job'] if 'job' in x.keys() else 'NA')
#         metric_groupby_metric_name_job = metric[metric['job'] != 'NA'].groupby(['metric_name', 'job'])
#         if len(metric_groupby_metric_name_job) > 0:
#             for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
#                 for (k0, k1), v in metric_groupby_metric_name_job.apply(lambda x: stats_func(x['value'])).to_dict().items():
#                     feat[f'metric_metric_name_{k0}_job_{k1}_value_{stats_func.__name__}'] = v
                    
#         # groupby: metric_name, instance, job
#         metric_groupby_metric_name_instance_job = metric[(metric['instance'] != 'NA') & (metric['job'] != 'NA')].groupby(['metric_name', 'instance', 'job'])
#         if len(metric_groupby_metric_name_instance_job) > 0:
#             for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
#                 for (k0, k1, k2), v in metric_groupby_metric_name_instance_job.apply(lambda x: stats_func(x['value'])).to_dict().items():
#                     feat[f'metric_metric_name_{k0}_instance_{k1}_job_{k2}_value_{stats_func.__name__}'] = v 
                
        # level_0: groupby: 'metric_name', 'cpu', 'instance', 'job', 'mode'
        metric_groupby_metric_level_0 = metric[(metric['metric_name'].isin(T.metric_level_0)) & (metric['cpu'] != 'NA') & (metric['instance'] != 'NA') & (metric['job'] != 'NA') & (metric['mode'] != 'NA')].groupby(['metric_name', 'cpu', 'instance', 'job', 'mode'])
        if len(metric_groupby_metric_level_0) > 0:
            for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
                for (k0, k1, k2, k3, k4), v in metric_groupby_metric_level_0.apply(lambda x: stats_func(x['value'])).to_dict().items():
                    feat[f'metric_metric_name_{k0}_cpu_{k1}_instance_{k2}_job_{k3}_mode_{k4}_value_{stats_func.__name__}'] = v

        # level_1: groupby:  'metric_name', 'container', 'image', 'instance', 'job', 'kubernetes_io_hostname', 'namespace',
        metric_groupby_metric_level_1 = metric[(metric['metric_name'].isin(T.metric_level_1)) & (metric['container'] != 'NA') & (metric['image'] != 'NA') & (metric['instance'] != 'NA') & (metric['job'] != 'NA') & (metric['kubernetes_io_hostname'] != 'NA') & (metric['namespace'] != 'NA')].groupby(['metric_name', 'container', 'image', 'instance', 'job', 'kubernetes_io_hostname', 'namespace'])
        if len(metric_groupby_metric_level_1) > 0:
            for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
                for (k0, k1, k2, k3, k4, k5, k6), v in metric_groupby_metric_level_1.apply(lambda x: stats_func(x['value'])).to_dict().items():
                    feat[f'metric_metric_name_{k0}_container_{k1}_image_{k2}_instance_{k3}_job_{k4}_kubernetes_io_hostname_{k5}_namespace_{k6}_value_{stats_func.__name__}'] = v
                    
        # level_2: groupby: 'metric_name', 'instance', 'job'
        metric_groupby_metric_level_2 = metric[(metric['metric_name'].isin(T.metric_level_2)) & (metric['instance'] != 'NA') & (metric['job'] != 'NA')].groupby(['metric_name', 'instance', 'job'])
        if len(metric_groupby_metric_level_2) > 0:
            for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
                for (k0, k1, k2), v in metric_groupby_metric_level_2.apply(lambda x: stats_func(x['value'])).to_dict().items():
                    feat[f'metric_metric_name_{k0}_instance_{k1}_job_{k2}_value_{stats_func.__name__}'] = v
                    
  

        # level_3: groupby: 'metric_name', 'service_name'
        metric_groupby_metric_level_3 = metric[(metric['metric_name'].isin(T.metric_level_3)) & (metric['service_name'] != 'NA')].groupby(['metric_name', 'service_name'])
        if len(metric_groupby_metric_level_3) > 0:
            for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
                for (k0, k1), v in metric_groupby_metric_level_3.apply(lambda x: stats_func(x['value'])).to_dict().items():
                    feat[f'metric_metric_name_{k0}_service_name_{k1}_value_{stats_func.__name__}'] = v
                    
    else:
        feat['metric_length'] = 0
    
    # log
#     if len(log) > 0:
#         feat['log_length'] = len(log)

#         log_message = log['message'].str.extract(T.log_message_pattern, expand=True)
#         log_message.columns = [f'message_{i}' for i in range(5)]
#         log_message['message_1'] = log_message['message_1'].apply(lambda x: str(x).split('-')[0])
#         for c in log_message.columns:
#             log_message[c] = log_message[c].astype(str)
#         log = pd.concat([log, log_message], axis=1)

#         # groupby: service, message_{i}
#         for g in ['service'] + list(log_message.columns):
#             log_groupby_g = log.groupby(g)
        
#             ## timestamp: std, skew, kurt
#             for stats_func in [pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
#                 for k, v in log_groupby_g['timestamp'].agg(stats_func).to_dict().items():
#                     feat[f'log_{g}_{k}_timestamp_{stats_func.__name__}'] = v

#             ## timestamp_diff: min, max, ptp, mean, std, skew, kurt
#             for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt, pd.Series.autocorr]:   
#                 for k, v in log_groupby_g.apply(lambda x: stats_func(x['timestamp'].diff().fillna(0))).to_dict().items():
#                     feat[f'log_{g}_{k}_timestamp_diff_{stats_func.__name__}'] = v

#     else:
#         feat['log_length'] = 0
    
    # trace
    if len(trace) > 0:
        feat['trace_length'] = len(trace)
        # trace = trace.sort_values(by='timestamp')
        
        trace['cost_time'] = trace['end_time'] - trace['start_time']
        trace['host_ip'] = trace['host_ip'].apply(T.transfer_host_ip)
        trace['endpoint_name'] = trace['endpoint_name'].apply(T.transfer_endpoint_name)
        
        # groupby: trace_id -> start_time_diff, end_time_diff
        trace_groupby_trace_id = trace.groupby('trace_id').apply(lambda x: x[['start_time', 'end_time']].diff().fillna(0)).reset_index()
        trace_groupby_trace_id.columns = ['trace_id', 'index', 'start_time_diff', 'end_time_diff']
        trace_groupby_trace_id = trace_groupby_trace_id.set_index('index')
        trace = trace.merge(trace_groupby_trace_id[['start_time_diff', 'end_time_diff']], left_index=True, right_index=True).groupby('trace_id').apply(lambda x: x)
        
        
        # groupby: service_name
        trace_groupby_service_name = trace.groupby('service_name')

        ## cost_time mean, std, min, max, ptp
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
            for k, v in trace_groupby_service_name['cost_time'].agg(stats_func).to_dict().items():
                feat[f'trace_service_name_{k}_cost_time_{stats_func.__name__}'] = v
            for k, v in trace_groupby_service_name['start_time_diff'].agg(stats_func).to_dict().items():
                feat[f'trace_service_name_{k}_start_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_service_name['end_time_diff'].agg(stats_func).to_dict().items():
                feat[f'trace_service_name_{k}_end_time_diff_{stats_func.__name__}'] = v
            
        # groupby: host_ip
        trace_groupby_host_ip = trace.groupby('host_ip')

        ## cost_time sum, mean, std, min, max, ptp
        for stats_func in [pd.Series.min, pd.Series.max, np.ptp, pd.Series.mean, pd.Series.std, pd.Series.skew, pd.Series.kurt]:
            for k, v in trace_groupby_host_ip['cost_time'].agg(stats_func).to_dict().items():
                feat[f'trace_host_ip_{k}_cost_time_{stats_func.__name__}'] = v
            for k, v in trace_groupby_host_ip['start_time_diff'].agg(stats_func).to_dict().items():
                feat[f'trace_host_ip_{k}_start_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_host_ip['end_time_diff'].agg(stats_func).to_dict().items():
                feat[f'trace_host_ip_{k}_end_time_diff_{stats_func.__name__}'] = v
                
        # groupby: endpoint_name
        trace_groupby_endpoint_name = trace.groupby('endpoint_name')

        ## cost_time sum, mean, std, min, max, ptp
        for stats_func in [pd.Series.mean, pd.Series.median, pd.Series.std, pd.Series.min, pd.Series.max, np.ptp]:
            for k, v in trace_groupby_endpoint_name['cost_time'].agg(stats_func).to_dict().items():
                feat[f'trace_endpoint_name_{k}_cost_time_{stats_func.__name__}'] = v
            for k, v in trace_groupby_endpoint_name['start_time_diff'].agg(stats_func).to_dict().items():
                feat[f'trace_endpoint_name_{k}_start_time_diff_{stats_func.__name__}'] = v
            for k, v in trace_groupby_endpoint_name['end_time_diff'].agg(stats_func).to_dict().items():
                feat[f'trace_endpoint_name_{k}_end_time_diff_{stats_func.__name__}'] = v
                    
    else:
        feat['trace_length'] = 0

    return feat


def _extract_features(log: pd.DataFrame, metric: pd.DataFrame, trace: pd.DataFrame) -> dict:  
    feat = {}
    feat['log_length'] = len(log)
    feat['metric_length'] = len(metric)
    feat['trace_length'] = len(trace)
    return feat

def read_data_by_id(data_dir: str, idx: str, i: str) -> dict:
    log, metric, trace = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # path_log = os.path.join(data_dir, idx, 'log', f'{i}_log.csv')
    # if os.path.exists(path_log):
    #     log = pd.read_csv(path_log)
        
    path_metric = os.path.join(data_dir, idx, 'metric', f'{i}_metric.csv')
    if os.path.exists(path_metric):
        metric = pd.read_csv(path_metric)
        
    path_trace = os.path.join(data_dir, idx, 'trace', f'{i}_trace.csv')
    if os.path.exists(path_trace):
        trace = pd.read_csv(path_trace)
        
    feat = {'id': i}
    feat.update(extract_features(log=log, metric=metric, trace=trace))
    return feat


def read_train_data_by_idx(data_dir: str, idx: str) -> pd.DataFrame:
    label = pd.read_csv(os.path.join(data_dir, idx, f'ground_truth_label_{idx}.csv'))
    label = pd.pivot(label, index='id', columns='source', values='score').reset_index()
    label = label.groupby('id').apply(lambda x: {k: list(v.values())[0] for k, v in x.to_dict().items()}).to_dict()
    
    labels = pd.DataFrame.from_records(list(label.values()))
    records = Parallel(n_jobs=args.num_threads, backend="multiprocessing")(delayed(read_data_by_id)(data_dir, idx, i) for i in tqdm(label.keys()))
    feats = pd.DataFrame.from_records(records)
            
    return pd.merge(labels, feats, on='id')


def read_train_data(data_dir: str, data_idx: str) -> pd.DataFrame:
    data = []
    for idx in data_idx.split(','): 
        train_data_idx = read_train_data_by_idx(data_dir, idx)
        data.append(train_data_idx)
        gc.collect()
        print(train_data_idx.shape)
        
    return pd.concat(data, ignore_index = True)


def read_test_data(data_dir: str) -> pd.DataFrame:
    log_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, '0', 'log'))]
    metric_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, '0', 'metric'))]
    trace_id_list = [i.split('_')[0] for i in os.listdir(os.path.join(data_dir, '0', 'trace'))]
    id_list = list(set(log_id_list) | set(metric_id_list) | set(trace_id_list))

    records = Parallel(n_jobs=args.num_threads, backend="multiprocessing")(delayed(read_data_by_id)(data_dir, '0', i) for i in tqdm(id_list))
    return pd.DataFrame.from_records(records)


if __name__ == '__main__':
    test_data = read_test_data(args.data_dir).sort_values(by='id')
    test_data.to_parquet(os.path.join(args.feat_dir, 'test_data.parquet'), index=False)
    print(test_data.shape)
    
    train_data = read_train_data(args.data_dir, args.data_idx).sort_values(by='id')
    train_data.to_parquet(os.path.join(args.feat_dir, 'train_data.parquet'), index=False)
    print(train_data.shape)
