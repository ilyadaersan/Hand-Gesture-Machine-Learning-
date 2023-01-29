import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report, roc_auc_score
from xgboost import XGBClassifier

#uploading data
train = loadmat(r"train.mat")
train_target = loadmat(r'train_target.mat')

X=train['D']
y=train_target['target']

X= pd.DataFrame(X).applymap(np.absolute)

test = loadmat(r"test.mat")
test_target = loadmat(r'test_target.mat')

test_data=test['D']
test_data_target=test_target['target']

test_data= pd.DataFrame(test_data).applymap(np.absolute)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



xgb_param = {
 'eta'
 : [0.3]
}
#
model=XGBClassifier()

gs_xgb = GridSearchCV(model,
                      param_grid=xgb_param,
                      scoring='accuracy',
                      cv=5)

gs_xgb.fit(X_train, y_train)


y_train_pred_xgb = gs_xgb.predict(X_train)
y_test_pred_xgb = gs_xgb.predict(X_test)

print('XGBClassifier Grid Model Train Report: \n Confussion Matrix \n')
print(confusion_matrix(y_train,y_train_pred_xgb),'\n')
print('XGBClassifier Grid report \n')
print(classification_report(y_train,y_train_pred_xgb),'\n')

print('XGBClassifier Grid Model Test Report: \n Confussion Matrix \n')
print(confusion_matrix(y_test,y_test_pred_xgb),'\n')
print('XGBClassifier Grid report \n')
print(classification_report(y_test,y_test_pred_xgb),'\n')

y_last_pred = gs_xgb.predict(test_data)


print('XGBClassifier Grid Model Test Report: \n Confussion Matrix \n')
print(confusion_matrix(test_data_target,y_last_pred),'\n')
print('XGBClassifier Grid report \n')
print(classification_report(test_data_target,y_last_pred),'\n')

roc_auc_score(test_data_target, gs_xgb.predict_proba(test_data), multi_class='ovr')
print(roc_auc_score(test_data_target, gs_xgb.predict_proba(test_data), multi_class='ovr'))









