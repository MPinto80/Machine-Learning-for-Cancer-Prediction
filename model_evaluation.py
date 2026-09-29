from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

from data_preprocessing import y_test, labels
from model_training import y_pred

import numpy as np
import pandas as pd

#accuracy
accuracy=np.round(balanced_accuracy_score(y_test,y_pred),4)
print('accuracy:%0.4f'%accuracy)

#precision
precision=np.round(precision_score(y_test,y_pred,average = 'weighted'),4)
print('precision:%0.4f'%precision)

#recall
recall=np.round(recall_score(y_test,y_pred,average = 'weighted'),4)
print('recall:%0.4f'%recall)

#f1score
f1score=np.round(f1_score(y_test,y_pred,average='weighted'),4)
print('f1score:%0.4f'%f1score)

report=classification_report(y_test,y_pred, target_names=labels)
print('\n')
print('classification report \n\n')
print(report)

#generate confusion matrix
cm=confusion_matrix(y_test,y_pred)
cm_df=pd.DataFrame(cm,index=labels,columns=labels)

print(cm_df)


