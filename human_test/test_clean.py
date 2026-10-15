
import pandas as pd

from solution import Cleaner

if __name__ == "__main__":

    df = pd.read_csv('../train.csv')
    clean = Cleaner(df)
    clean.clean()
    clean.cleanedX.to_csv('cleanedX.csv',index=False)