from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
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
n_neighbors = 300


## output
os.makedirs(('./output/{}/knn/n_neighbors{}/'.format(task, n_neighbors)), exist_ok=True)
file_metrics = './output/{}/knn/n_neighbors{}/metrics.txt'.format(task, n_neighbors)
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
    

    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(x_train_value,y_train_value)

    pred_y_train = model.predict_proba(x_train_value)
    print(pred_y_train)
    pred_y_test = model.predict_proba(x_test_value)
    print(pred_y_test)


    ## metrics
    path_train = './output/{}/knn/n_neighbors{}/cv{}-train.csv'.format(task, n_neighbors, cv)
    path_test = './output/{}/knn/n_neighbors{}/cv{}-test.csv'.format(task, n_neighbors, cv)
    ACC_train, AUC_train, F1_train, MCC_train = calculate(y_train_value,pred_y_train,path_train)
    ACC_test, AUC_test, F1_test, MCC_test = calculate(y_test_value,pred_y_test,path_test)

    result = [cv, ACC_train, AUC_train, F1_train, MCC_train, ACC_test, AUC_test, F1_test, MCC_test]

    with open(file_metrics, 'a') as f:
        f.write('\t'.join(map(str, result)) + '\n')

    print("finish cv: "+str(cv))



