from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

from data_preprocessing import y_train
from feature_selection import x_train_selected, x_test_selected

#random forest classifier
#since we are dealing with multiclass data, the one versus rest strategy is used

rf=OneVsRestClassifier(RandomForestClassifier(max_features=0.2))
rf.fit(x_train_selected,y_train)
y_pred = rf.predict(x_test_selected)
pred_prob = rf.predict_proba(x_test_selected)
