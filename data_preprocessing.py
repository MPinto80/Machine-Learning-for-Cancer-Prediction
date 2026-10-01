from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

def preprocess_data(df, test_size, random_state):
    print('Preprocessing data...')
    x = df.iloc[:, 0:-1]
    y = df.iloc[:, -1]

    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)
    labels = label_encoder.classes_
    classes = np.unique(y_encoded)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded, test_size=test_size, random_state=random_state
    )

    min_max_scaler = MinMaxScaler()
    x_train_norm = min_max_scaler.fit_transform(x_train)
    # FIX: Use transform on test data to avoid data leakage
    x_test_norm = min_max_scaler.transform(x_test)

    return x_train_norm, x_test_norm, y_train, y_test, labels, classes
