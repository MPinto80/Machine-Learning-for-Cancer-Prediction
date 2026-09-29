import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/cancer_gene_expression.csv')

#understanding the dataset
print(df.shape)
print(df.columns[0:3])
print(df.columns[-1])

#checking for missing values
datanull = df.isnull().sum()
g=[i for i in datanull if i>0]

print('columns with missing values:%d'%len(g))

#checking how many cancer types are there in the data
print(df['Cancer_Type'].value_counts())

#plotbar for easier visualization
df['Cancer_Type'].value_counts().plot.bar()
plt.show()




