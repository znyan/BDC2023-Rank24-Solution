# 代码说明

## 环境配置（必选）
- Python 3.8.10
- catboost==1.1.1
- iterative_stratification==0.1.7
- lightgbm==3.3.5
- numpy==1.23.5
- pandas==1.5.2
- scikit_learn==1.2.1
- tqdm==4.61.2
- xgboost==1.7.4
依赖包已放入`requirements.txt`中，执行`init.sh`时会完成上述依赖包的安装。

## 数据
使用的数据均为比赛提供的数据，未使用外部数据。

## 预训练模型
未使用预训练模型。

## 算法（必选）

### 整体思路介绍（必选）

#### 特征工程
此部分核心实现在`code/data.py`的`extract_features`函数中，特征工程仅使用了trace、log表，未使用metric，主要特征如下：
- trace表
    - groupby_service_name特征：按service_name分组，构建host_ip、endpoint_name的统计特征（count/nunique），构建cost_time（end_time-start_time）的统计特征（mean/median/std/min/max/ptp），时间戳的统计特征（std/skew/kurt/autocorr）以及start_time、end_time、cost_time一阶差分的统计特征（min/max/ptp/mean/std/skew/kurt/autocorr）；
    - groupby_host_ip特征：按host_ip分组，构建的特征与service_name一致；
    - groupby_endpoint_name特征：对endpoint_name按一定的规则简化后分组统计，构建的特征与service_name一致。
    
- log表
    - groupby_service特征：按service分组，构建消息长度的统计特征（mean/std/skew/kurt/min/max/ptp/autocorr）、时间戳的统计特征（std/skew/kurt/autocorr）、时间戳一阶差分的统计特征（min/max/ptp/mean/std/skew/kurt/autocorr）。


#### 模型和方法
对于9种异常类型分别构建二分类模型，并使用五折交叉验证，使用了以下三种集成学习算法：
- Lightgbm
- CatBoost
- XgBoost

### 损失函数
对数损失函数（Log loss）

### 模型集成
三个模型在测试集上的预测结果的加权平均。

## 训练流程
1. python data.py: 生成构造的特征表格，存在data/pretrain_model中；
2. python train_lgb.py: 训练LightGBM模型，模型保存在data/best_model/lgb下；
3. python train_ctb.py: 训练CatBoost模型，模型保存在data/best_model/ctb下；
4. python train_xgb.py: 训练XgBoost模型，模型保存在data/best_model/xgb下。

## 测试流程
1. python inference.py: 读取LightGBM、CatBoost、XgBoost在每一分类、每一折的模型，在测试集上分别推理结果，加权求和得到最终结果。

## 其他注意事项
1. 由于有多个、多折模型，`best_model`使用文件夹存储；
2. 生成的特征已提前放在`data/pretrain_model`下。
