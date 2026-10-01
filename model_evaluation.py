import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    ConfusionMatrixDisplay, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
import numpy as np
import pandas as pd

def evaluate_model(rf, x_test_selected, y_test, labels, classes):
    print('Evaluating model...')
    y_pred = rf.predict(x_test_selected)
    pred_prob = rf.predict_proba(x_test_selected)

    accuracy = np.round(balanced_accuracy_score(y_test, y_pred), 4)
    print('accuracy:%0.4f' % accuracy)

    precision = np.round(precision_score(y_test, y_pred, average='weighted'), 4)
    print('precision:%0.4f' % precision)

    recall = np.round(recall_score(y_test, y_pred, average='weighted'), 4)
    print('recall:%0.4f' % recall)

    f1score = np.round(f1_score(y_test, y_pred, average='weighted'), 4)
    print('f1score:%0.4f' % f1score)

    report = classification_report(y_test, y_pred, target_names=labels)
    print('\nclassification report \n\n', report)

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)

    sns.heatmap(cm_df, annot=True, cmap='Blues')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Seaborn Confusion Matrix')
    plt.show()

    disp = ConfusionMatrixDisplay.from_estimator(rf, x_test_selected, y_test, xticks_rotation='vertical',
                                                 cmap='Blues', display_labels=labels)
    plt.title('Sklearn Confusion Matrix')
    plt.show()

    y_test_binarized = label_binarize(y_test, classes=classes)
    fpr, tpr, thresh = {}, {}, {}
    roc_auc = dict()
    n_class = classes.shape[0]

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test_binarized[:, i], pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], linestyle='--',
                 label='%s vs Rest (AUC=%0.2f)' % (labels[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'b--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='lower right')
    plt.show()
