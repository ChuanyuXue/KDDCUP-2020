
## Before started, make sure you put
## 'underexpose_item_feat.csv' in ./user_data/dataset
## 'w2v_txt_vec.txt' in ./user_data
## 'w2v_img_vec.txt' in ./user_data

# Generate dataset
python ./code/1_DataPreprocessing/01_Generate_Offline_Dataset_origin.py
python ./code/1_DataPreprocessing/02_Generate_Model1_Dataset_origin.py
python ./code/1_DataPreprocessing/03_Create_Model1_Answer.py
python ./code/1_DataPreprocessing/04_TransformDateTime-Copy1.py

# Generate Similarity
python ./code/2_Similarity/deep_node_model.py
python ./code/2_Similarity/01_itemCF_Mundane_model1.py
python ./code/2_Similarity/01_itemCF_Mundane_offline.py
python ./code/2_Similarity/01_itemCF_Mundane_online.py
python ./code/2_Similarity/RA_Wu_model1.py
python ./code/2_Similarity/RA_Wu_offline.py
python ./code/2_Similarity/RA_Wu_online.py


# Generate candidates
python ./code/3_Recall/01_Recall-Wu-model1.py
python ./code/3_Recall/01_Recall-Wu-offline.py
python ./code/3_Recall/01_Recall-Wu-online.py

# NN model
python ./code/3_NN/ItemFeat2.py
# train 1 online
python ./code/3_NN/sas_rec.py --kind 1 --train 1
# train 2 offline
python ./code/3_NN/sas_rec.py --kind 2 --train 1
# train 3 model
python ./code/3_NN/sas_rec.py --kind 3 --train 1
# test 1
python ./code/3_NN/sas_rec.py --kind 1 --test 1
# tets 2
python ./code/3_NN/sas_rec.py --kind 2 --test 1
# test 3
python ./code/3_NN/sas_rec.py --kind 3 --test 1 

# Generate feature
python ./code/4_RankFeature/01_sim_feature_model1.py
python ./code/4_RankFeature/01_sim_feature_model1_RA_AA.py
python ./code/4_RankFeature/01_sim_feature_offline.py
python ./code/4_RankFeature/01_sim_feature_offline_RA_AA.py
python ./code/4_RankFeature/01_sim_feature_online.py
python ./code/4_RankFeature/01_sim_feature_online_RA_AA.py

python ./code/4_RankFeature/02_itemtime_feature_model1.py
python ./code/4_RankFeature/02_itemtime_feature_offline.py
python ./code/4_RankFeature/02_itemtime_feature_online.py

python ./code/4_RankFeature/03_count_feature_model1.py
python ./code/4_RankFeature/03_count_feature_offline.py
python ./code/4_RankFeature/03_count_feature_online.py

python ./code/4_RankFeature/04_NN_feature_model1.py
python ./code/4_RankFeature/04_NN_feature_offline.py
python ./code/4_RankFeature/04_NN_feature_online.csv.py

python ./code/4_RankFeature/05_txt_feature_model1.py
python ./code/4_RankFeature/05_txt_feature_offline.py
python ./code/4_RankFeature/05_txt_feature_online.py

python ./code/4_RankFeature/06_interactive_model1.py
python ./code/4_RankFeature/06_interactive_offline.py
python ./code/4_RankFeature/06_interactive_online.py

python ./code/4_RankFeature/07_count_detail_model1.py
python ./code/4_RankFeature/07_count_detail_offline.py
python ./code/4_RankFeature/07_count_detail_online.py

python ./code/4_RankFeature/08_user_feature_model1.py
python ./code/4_RankFeature/08_user_feature_offline.py
python ./code/4_RankFeature/08_user_feature_online.py

python ./code/4_RankFeature/09_partial_sim_feature_model1.py
python ./code/4_RankFeature/09_partial_sim_feature_offline.py
python ./code/4_RankFeature/09_partial_sim_feature_online.py

python ./code/4_RankFeature/10_emergency_feature_model1.py
python ./code/4_RankFeature/10_emergency_feature_offline.py
python ./code/4_RankFeature/10_emergency_feature_online.py

# Build model
python ./code/5_Modeling/Model_Online.py

