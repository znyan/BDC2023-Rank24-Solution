# 代码说明

## 环境配置
- Python 3.8 / Cuda 11.6
- pyarrow==12.0.1
- fastparquet==2023.7.0
- iterative_stratification==0.1.7
- joblib==1.3.1
- numpy==1.24.1
- pandas==2.0.3
- scikit_learn==1.3.0
- tqdm==4.64.1
- xgboost==1.7.6


## 数据
使用的数据均为赛题数据，无外部数据。

## 预训练模型
未使用预训练模型。

## 算法

### 整体思路介绍（必选）
特征工程 + 树模型，为每个异常类型都使用K折交叉验证构建K个树模型。

### 特征工程
- metric表
  对一百多个指标按出现频率分组，对于每一组指标，按tags中包含的字段groupby：
  - 'metric_name', 'cpu', 'instance', 'job', 'mode'
  - 'metric_name', 'container', 'image', 'instance', 'job', 'kubernetes_io_hostname', 'namespace'
  - 'metric_name', 'instance', 'job'
  - 'metric_name', 'service_name'
  groupby后统计以下特征：min, max, ptp, mean, std, skew, kurt。

- log表：未统计该表的相关特征

- trace表
  先按trace_id对每一组内的start_time、end_time进行差分统计，对service_name、host_ip、endpoint_name进行分组groupby操作，统计min, max, ptp, mean, std, skew, kurt。


### 模型结果
- XgBoost 二分类模型

### 数据扩增
统计每一列特征的缺失率，去除空值比例大于95%的特征。

## 训练流程
- step1. `python code/data.py` 特征提取
- step2. `python code/train.py` 模型训练及推理

## 测试流程
- `python codeinference.py` 读取训练好的模型推理出结果
