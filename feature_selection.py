from sklearn.feature_selection import mutual_info_classif
from data_preprocessing import x_train_norm, y_train, x_test_norm
import numpy as np

mi = mutual_info_classif(x_train_norm, y_train)[0]

# select top n features
n_features = 300
selected_scores_indices = np.argsort(mi)[::-1][0:n_features]

x_train_selected = x_train_norm[:, selected_scores_indices]
x_test_selected = x_test_norm[:, selected_scores_indices]

if __name__ == '__main__':
    print(x_train_selected.shape)
    print(x_test_selected.shape)