import pandas as pd
from gensim.models import KeyedVectors



train_path = './data/underexpose_train/'
item = pd.read_csv(train_path+'underexpose_item_feat.csv',header=None)

item[1] = item[1].apply(lambda x: float(str(x).replace('[', '')))
item[256] = item[256].apply(lambda x: float(str(x).replace(']', '')))
item[128] = item[128].apply(lambda x: float(str(x).replace(']', '')))
item[129] = item[129].apply(lambda x: float(str(x).replace('[', '')))
item.columns = ['item_id'] + ['txt_vec_{}'.format(f) for f in range(0, 128)] + ['img_vec_{}'.format(f) for f in
                                                                                range(0, 128)]
item_nun=item['item_id'].nunique()

item[['item_id'] + ['img_vec_{}'.format(f) for f in range(0, 128)]].to_csv("user_data/w2v_img_vec.txt", sep=" ",
                                                                                header=[str(item_nun), '128'] + [""] * 127,
                                                                                index=False,
                                                                                encoding='UTF-8')

item[['item_id'] + ['txt_vec_{}'.format(f) for f in range(0, 128)]].to_csv("user_data/w2v_txt_vec.txt",
                                                                                sep=" ",
                                                                                header=[str(item_nun), '128'] + [""] * 127,
                                                                                index=False,
                                                                                encoding='UTF-8')

txt_vec_model = KeyedVectors.load_word2vec_format("./user_data/" + 'w2v_txt_vec.txt', binary=False)
txt_vec_model = KeyedVectors.load_word2vec_format("./user_data/" + 'w2v_img_vec.txt', binary=False)