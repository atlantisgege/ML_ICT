from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd
import torch
import random


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate(correct_label,pred_score,path):
    r2 = r2_score(correct_label, pred_score)
    mae = mean_absolute_error(correct_label, pred_score)
    mse = mean_squared_error(correct_label, pred_score)
    y_pred = pred_score.tolist()
    c = {"true":correct_label, "pred":y_pred}
    labels = pd.DataFrame(c)
    labels.to_csv(path)
    return r2, mae, mse

def Rdsplit(total_sample, random_state = 888, split_size = 0.2):
    base_indices = np.arange(total_sample) 
    base_indices = shuffle(base_indices, random_state = random_state) 
    cv = int(len(base_indices) * split_size)
    idx_1 = base_indices[0:cv]
    idx_2 = base_indices[(cv):(2*cv)]
    idx_3 = base_indices[(2*cv):(3*cv)]
    idx_4 = base_indices[(3*cv):(4*cv)]
    idx_5 = base_indices[(4*cv):len(base_indices)]
    print(len(idx_1), len(idx_2), len(idx_3), len(idx_4), len(idx_5))
    return base_indices, idx_1, idx_2, idx_3, idx_4, idx_5


init_seeds(0)

x_data = pd.read_csv('./data/x_data_scale_final.csv',sep=',')
y_data = pd.read_csv('./data/predict_y_1_drop.csv',sep=',')

x_data_add_y= x_data
x_data_add_y['y_value']= y_data['COG']
x_data_add_y_sort = x_data_add_y.sort_values(by='y_value', ascending= True)

total_num = len(x_data_add_y_sort)
neg_num = int(0.3*total_num)
neg_samples = x_data_add_y_sort.iloc[:neg_num]
print(len(neg_samples))
pos_samples = x_data_add_y_sort.iloc[-neg_num:]
print(len(pos_samples))
neg_samples['clf_label'] = 0
print(neg_samples['clf_label'])
pos_samples['clf_label'] = 1
print(pos_samples['clf_label'])

select_variables = neg_samples.columns[3:-3]
# print(select_variables)

neg_x_data_value = neg_samples[select_variables].values
pos_x_data_value = pos_samples[select_variables].values
neg_y_label_value = neg_samples['clf_label'].values
pos_y_label_value = pos_samples['clf_label'].values


neg_sample_id = neg_samples.columns[:3]
pos_sample_id = pos_samples.columns[:3]
neg_sample_weight = neg_samples.columns[-3]
pos_sample_weight = pos_samples.columns[-3]
neg_x_data_id = neg_samples[neg_sample_id].values
pos_x_data_id = pos_samples[pos_sample_id].values
neg_x_data_weight = neg_samples[neg_sample_weight].values
pos_x_data_weight = pos_samples[pos_sample_weight].values


select_variables_list = select_variables.tolist()

neg_sample_id_list = neg_sample_id.tolist()


x_columns = neg_sample_id_list + select_variables_list + [neg_sample_weight]
y_columns = neg_sample_id_list + ['clf_label']



base_indices, idx_1, idx_2, idx_3, idx_4, idx_5 = Rdsplit(neg_num)

idx_all = [idx_1, idx_2, idx_3, idx_4, idx_5]
for i in range(5):
    print("fit cv: "+str(i))
    cv = i
    index_valid = idx_all[i]
    index_train = list(set(base_indices) - set(index_valid))

    neg_x_train = np.array([neg_x_data_value[j] for j in index_train])
    pos_x_train = np.array([pos_x_data_value[j] for j in index_train])
    x_train = np.concatenate([neg_x_train, pos_x_train], axis=0)

    neg_y_train = np.array([neg_y_label_value[j] for j in index_train])
    pos_y_train = np.array([pos_y_label_value[j] for j in index_train])
    y_train = np.concatenate([neg_y_train, pos_y_train], axis=0)

    neg_x_test = np.array([neg_x_data_value[j] for j in index_valid])
    pos_x_test = np.array([pos_x_data_value[j] for j in index_valid])
    x_test = np.concatenate([neg_x_test, pos_x_test], axis=0)

    neg_y_test = np.array([neg_y_label_value[j] for j in index_valid])
    pos_y_test = np.array([pos_y_label_value[j] for j in index_valid])
    y_test = np.concatenate([neg_y_test, pos_y_test], axis=0)

    neg_x_train_id = pd.DataFrame(np.array([neg_x_data_id[j] for j in index_train]), columns=neg_sample_id)
    neg_x_train_weight = pd.DataFrame(np.array([neg_x_data_weight[j] for j in index_train]), columns=['W_FSTUWT'])
    neg_x_test_id = pd.DataFrame(np.array([neg_x_data_id[j] for j in index_valid]), columns=neg_sample_id)
    neg_x_test_weight = pd.DataFrame(np.array([neg_x_data_weight[j] for j in index_valid]), columns=['W_FSTUWT'])

    pos_x_train_id = pd.DataFrame(np.array([pos_x_data_id[j] for j in index_train]), columns=pos_sample_id)
    pos_x_train_weight = pd.DataFrame(np.array([pos_x_data_weight[j] for j in index_train]), columns=['W_FSTUWT'])
    pos_x_test_id = pd.DataFrame(np.array([pos_x_data_id[j] for j in index_valid]), columns=pos_sample_id)
    pos_x_test_weight = pd.DataFrame(np.array([pos_x_data_weight[j] for j in index_valid]), columns=['W_FSTUWT'])

    x_train_id = pd.concat([neg_x_train_id,pos_x_train_id],axis=0,ignore_index=True)
    x_test_id = pd.concat([neg_x_test_id,pos_x_test_id],axis=0,ignore_index=True)
    x_train_weight = pd.concat([neg_x_train_weight,pos_x_train_weight],axis=0,ignore_index=True)
    x_test_weight = pd.concat([neg_x_test_weight,pos_x_test_weight],axis=0,ignore_index=True)


    x_train_df = pd.DataFrame(x_train, columns=select_variables)
    x_test_df = pd.DataFrame(x_test, columns=select_variables)

    x_train_id_df = pd.concat([x_train_id,x_train_df],axis=1,ignore_index=True)

    x_train_id_weight_df = pd.concat([x_train_id_df,x_train_weight],axis=1,ignore_index=True)

    x_train_id_weight_df.columns=x_columns
    x_test_id_df = pd.concat([x_test_id,x_test_df],axis=1,ignore_index=True)
    x_test_id_weight_df = pd.concat([x_test_id_df,x_test_weight],axis=1,ignore_index=True)
    x_test_id_weight_df.columns=x_columns

    y_train_df = pd.DataFrame(y_train, columns=['clf_lable'])
    y_train_id_df = pd.concat([x_train_id,y_train_df],axis=1,ignore_index=True)
    y_train_id_df.columns=y_columns

    y_test_df = pd.DataFrame(y_test, columns=['clf_lable'])
    y_test_id_df = pd.concat([x_test_id,y_test_df],axis=1,ignore_index=True)
    y_test_id_df.columns=y_columns

    x_train_id_weight_df.to_csv("./cv_data/classification/x-train-cv%s.csv"%(cv),index=False, header=True)
    y_train_id_df.to_csv("./cv_data/classification/y-train-cv%s.csv"%(cv),index=False, header=True)
    x_test_id_weight_df.to_csv("./cv_data/classification/x-test-cv%s.csv"%(cv),index=False,header=True)
    y_test_id_df.to_csv("./cv_data/classification/y-test-cv%s.csv"%(cv),index=False, header=True)


