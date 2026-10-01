import pandas as pd
import matplotlib.pyplot as plt

def load_data(filepath):
    print('Loading data...')
    df = pd.read_csv(filepath)
    return df

def explore_data(df):
    print("Dataset Shape:", df.shape)
    print("First 3 columns:", df.columns[0:3])
    print("Target column:", df.columns[-1])

    datanull = df.isnull().sum()
    g = [i for i in datanull if i > 0]
    print('Columns with missing values: %d' % len(g))
    
    print("\nCancer Type Distribution:")
    print(df['Cancer_Type'].value_counts())

    df['Cancer_Type'].value_counts().plot.bar()
    plt.title('Cancer Type Distribution')
    plt.show()

if __name__ == '__main__':
    from config import DATA_PATH
    df = load_data(DATA_PATH)
    explore_data(df)