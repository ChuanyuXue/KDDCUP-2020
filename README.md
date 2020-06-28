# KDDCUP-2020
2020-KDDCUP，Debiasing赛道 第6名解决方案

This repository contains the 6th solution on KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing Challenge.

赛题链接：https://tianchi.aliyun.com/competition/entrance/231785/introduction

解决方案blog: https://zhuanlan.zhihu.com/p/149424540

数据集下载链接：
underexpose_train.zip	271.62MB	http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231785/underexpose_train.zip
underexpose_test.zip	3.27MB	   http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/231785/underexpose_test.zip



## 解决方案
1. 如下文件结构所示，我们先对数据做预处理“1_DataPreprocessing”，将倒数第二次点击当答案生成线下训练集（存于user_data/model_1），将倒数第一次
点击当答案生成线下验证集（存于user_data/offline），线上待预测数据存于user_data/dataset。我们依据点击数的周期变换，将time转换为了
日期（04_TransformDateTime-Copy1.py），还生成了文本相似性、图像相似性文件（05_Generate_img_txt_vec.py）。

2. 依次选用线下训练集、线下验证集和线上待预测数据中的点击日志训练deepwalk、node2vec模型（“deep_node_model.py”）。进而，融合文本相似性
、deepwalk、node2vec修改了ItemCF算法，计算并存储商品相似性（“01_itemCF_Mundane_model1.py”等）。此外，基于召回的商品相似性构建商品相似性网络，
计算并存储RA、AA、CN、HDI、HPI、LHN1等二阶相似性（“RA_Wu_model1.py”等）。

3. 实现Self-Attentive Sequnetial Model，预测召回的用户-商品对的发生点击的概率（“3_NN”）。

4. 基于存储的商品相似性为每个待预测用户召回1000候选商品（“3_Recall”）。
5. 为召回列表中的商品-用户对生成排序特征（“4_RankFeature”）。

6. 将召回列表中真正发生点击的用户-商品对视为正样，按1:5的正负比例从召回列表中随机选取负样，生成6个数据集。进而，采用catboost和lightgbm
建模，为点击量少的商品赋予更大的权重，采用算数平均值、几何平均值与调和平均值做模型融合，并依据商品点击量进行后处理（“5_Modeling”）。

**最终我们的方案取得了Track-A 1th，Track-B 6th的成绩。**


## 文件结构
数据可以在比赛官方网站中下载，按照以下路径创建文件夹以及放置数据。

    │  feature_list.csv                               # List the features we used in ranking process
    │  main.sh                                        # Run this script to start the whole process
    │  project_structure.txt                          # The tree structure of this project
    │  
    ├─code
    │  │  __init__.py
    │  │  
    │  ├─1_DataPreprocessing                          # Generate validation-set, create timestamp and generate item feature vectors
    │  │      01_Generate_Offline_Dataset_origin.py   
    │  │      02_Generate_Model1_Dataset_origin.py
    │  │      03_Create_Model1_Answer.py
    │  │      03_Create_Offline_Answer.py
    │  │      04_TransformDateTime-Copy1.py
    │  │      05_Generate_img_txt_vec.py
    │  │      ipynb_file.zip
    │  │      
    │  ├─2_Similarity                                 # Generate item-item similarity matrix 
    │  │      01_itemCF_Mundane_model1.py
    │  │      01_itemCF_Mundane_offline.py
    │  │      01_itemCF_Mundane_online.py
    │  │      deep_node_model.py
    │  │      ipynb_file.zip
    │  │      RA_Wu_model1.py
    │  │      RA_Wu_offline.py
    │  │      RA_Wu_online.py
    │  │      
    │  ├─3_NN                                         # Generate deep-learning based result
    │  │      config.py
    │  │      ItemFeat2.py
    │  │      model2.py
    │  │      modules.py
    │  │      Readme
    │  │      sampler2.py
    │  │      sas_rec.py
    │  │      util.py
    │  │      
    │  ├─3_Recall                                     # Recall candidates
    │  │      01_Recall-Wu-model1.py
    │  │      01_Recall-Wu-offline.py
    │  │      01_Recall-Wu-online.py
    │  │      ipynb_file.zip
    │  │      
    │  ├─4_RankFeature                                # Generate feature for ranking
    │  │      01_sim_feature_model1.py
    │  │      01_sim_feature_model1_RA_AA.py
    │  │      01_sim_feature_offline.py
    │  │      01_sim_feature_offline_RA_AA.py
    |  |      ……
    │  │      10_emergency_feature_offline.py
    │  │      10_emergency_feature_online.py
    │  │      4_RankFeature.zip
    │  │      
    │  └─5_Modeling                                  # Build Catboost and LightGBM model
    │          ipynb_file.zip
    │          Model_Offline.py
    │          Model_Online.py
    │          
    ├─data                                           # Origin dataset
    │  ├─underexpose_test
    │  └─underexpose_train
    ├─prediction_result
    └─user_data                                      # Containing intermediate files
        ├─dataset
        │  ├─new_recall
        │  ├─new_similarity
        │  └─nn
        ├─model_1
        │  ├─new_recall
        │  ├─new_similarity
        │  └─nn
        └─offline
            ├─new_recall
            ├─new_similarity
            └─nn
        
## Python库环境依赖
    lightgbm==2.2.1
    tensorflow==1.13.1
    joblib==0.15.1
    gensim==3.4.0
    pandas==0.25.1
    numpy==1.16.3
    networkx==2.4
    tqdm==4.46.0

## 声明/
本项目库专门存放KDD2020挑战赛的相关代码文件，所有代码仅供各位同学学习参考使用。如有任何对代码的问题请邮箱联系：cs_xcy@126.com/chuanyu.xue@uconn.edu

If you have any issue please feel free to contact me at cs_xcy@126.com

天池ID：GrandRookie，
BruceQD，
七里z，
青禹小生，
蓝绿黄红，
LSH123，
XMNG，
wenwen_123，
**小雨姑娘**，
wbbhcb
