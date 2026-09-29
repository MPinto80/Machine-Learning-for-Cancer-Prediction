from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from data_exploration import df
import numpy as np

#separating the feature values because scikit-learn requires that features and class are separated before parsing them to the classifiers.

x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

#encoding target labels (y) with values between 0 and n_classes-1
#encoding will be done usgin the LabelEncoder

label_encoder = LabelEncoder()
label_encoder.fit(y)
y_encoded = label_encoder.transform(y)
labels=label_encoder.classes_
classes=np.unique(y_encoded)

#split data into training and test sets
x_train,x_test,y_train,y_test=train_test_split(x,y_encoded,test_size=0.2,random_state=42)

print(df.iloc[:,0:10].describe())

#scale data between 0 and 1
min_max_scaler = MinMaxScaler()
x_train_norm=min_max_scaler.fit_transform(x_train)
x_test_norm=min_max_scaler.fit_transform(x_test)
