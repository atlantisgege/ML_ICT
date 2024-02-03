from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, accuracy_score, matthews_corrcoef, roc_auc_score
import numpy as np
import pandas as pd
import random
import os


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def calculate(correct_label,pred_score,path):
    AUC = roc_auc_score(correct_label, pred_score[:,1])
    precision, recall, thresholds = precision_recall_curve(correct_label, pred_score[:,1])
    prc_auc = auc(recall, precision)
    predicted_labels = np.argmax(pred_score, axis=1)
    y_pred = predicted_labels.tolist()
    c = {"true":correct_label, "pred":y_pred}
    labels = pd.DataFrame(c)
    labels.to_csv(path, index=False)
    ACC = accuracy_score(correct_label, predicted_labels)
    CM = confusion_matrix(correct_label, predicted_labels)
    TN = CM[0][0]
    FP = CM[0][1]
    FN = CM[1][0]
    TP = CM[1][1]
    print('TN: {}, FP: {}, FN: {}, TP: {}'.format(TN, FP, FN, TP))
    Rec = TP / (TP + FN)
    Pre = TP / (TP + FP)
    F1 = 2*Rec*Pre/(Rec+Pre)
    MCC = matthews_corrcoef(correct_label, predicted_labels)
    return ACC, AUC, F1, MCC


## task
task = 'classification'

## hyper-parameters
n_estimators = 450
max_depth=10
min_samples_leaf = 1

## output
os.makedirs(('./output/{}/RF/estimator{}_depth{}/'.format(task, n_estimators, max_depth)), exist_ok=True)
file_metrics = './output/{}/RF/estimator{}_depth{}/metrics.txt'.format(task, n_estimators, max_depth)
results = ('cv\tACC_train\tAUC_train\tF1_train\tMCC_train\tACC_test\tAUC_test\tF1_test\tMCC_test')
with open(file_metrics, 'w') as f:
    f.write(results + '\n')


## five-fold
for cv in range(5):
    print("cv: "+str(cv))
    init_seeds(0)

    x_train_data = pd.read_csv("./cv_data/%s/x-train-cv%s.csv"%(task, cv),sep=',')
    y_train_data = pd.read_csv("./cv_data/%s/y-train-cv%s.csv"%(task, cv),sep=',')
    x_test_data = pd.read_csv("./cv_data/%s/x-test-cv%s.csv"%(task, cv),sep=',')
    y_test_data = pd.read_csv("./cv_data/%s/y-test-cv%s.csv"%(task, cv),sep=',')

    select_variables = x_train_data.columns[3:-1]

    x_train_value = x_train_data[select_variables].values
    x_test_value = x_test_data[select_variables].values
    print(len(x_train_value))
    print(x_train_value[-1])
    y_train_value = y_train_data['clf_label'].values
    print(len(y_train_value))
    print(y_train_value[-1])
    y_test_value = y_test_data['clf_label'].values

    train_data = list(zip(x_train_value,y_train_value))
    random.shuffle(train_data)
    x_train_value,y_train_value = zip(*train_data)
    

    ## model
    model = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            criterion='gini',
            max_depth=max_depth,
            random_state=0,
            n_jobs=-1)

    
    model.fit(x_train_value,y_train_value)
    pred_y_train = model.predict_proba(x_train_value)
    print(pred_y_train)
    pred_y_test = model.predict_proba(x_test_value)
    print(pred_y_test)

    # Get numerical feature importances
    importances = list(model.feature_importances_)
    
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(select_variables, importances)]
    
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    
    # Feature  importances 
    feat_list = []
    importance_list = []
    for pair in feature_importances:
        feat_list.append(pair[0])
        importance_list.append(pair[1])
    feat_df = pd.DataFrame(columns=['features','importance'])
    feat_df['features'] = feat_list
    feat_df['importance'] = importance_list
    feat_df.to_csv('./output/{}/RF/estimator{}_depth{}/cv{}.csv'.format(task, n_estimators, max_depth, cv), index=False)

    ## metrics
    path_train = './output/{}/RF/estimator{}_depth{}/cv{}-train.csv'.format(task, n_estimators, max_depth, cv)
    path_test = './output/{}/RF/estimator{}_depth{}/cv{}-test.csv'.format(task, n_estimators, max_depth, cv)
    ACC_train, AUC_train, F1_train, MCC_train = calculate(y_train_value,pred_y_train,path_train)
    ACC_test, AUC_test, F1_test, MCC_test = calculate(y_test_value,pred_y_test,path_test)

    result = [cv, ACC_train, AUC_train, F1_train, MCC_train, ACC_test, AUC_test, F1_test, MCC_test]

    with open(file_metrics, 'a') as f:
        f.write('\t'.join(map(str, result)) + '\n')

    print("finish cv: "+str(cv))



