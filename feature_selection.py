from sklearn.feature_selection import mutual_info_classif
import numpy as np

def select_features(x_train_norm, x_test_norm, y_train, n_features):
    print(f'Selecting top {n_features} features (this may take 30-60 seconds)...')
    mi = mutual_info_classif(x_train_norm, y_train)

    selected_scores_indices = np.argsort(mi)[::-1][0:n_features]

    x_train_selected = x_train_norm[:, selected_scores_indices]
    x_test_selected = x_test_norm[:, selected_scores_indices]
    
    return x_train_selected, x_test_selected