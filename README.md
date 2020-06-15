# KDDCUP-2020
2020-KDDCUP，Debiasing赛道 第6名解决方案

This repository contains the 6th solution on KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing Challenge.

## 解决方案blog

## 文件结构
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


## 声明
本项目库专门存放KDD2020挑战赛的相关代码文件，所有代码仅供各位同学学习参考使用。如有任何对代码的问题请邮箱联系：cs_xcy@126.com

If you have any issue please feel free to contact me at cs_xcy@126.com

天池ID： **小雨姑娘**


