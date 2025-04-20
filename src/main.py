import pandas as pd
from sklearn.datasets import load_breast_cancer

if __name__ == "__main__":
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    print(df.head())
