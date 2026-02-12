import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    w_hopping = pd.read_csv("snr_w_hopping2.csv")
    wo_hopping = pd.read_csv("snr_wo_hopping2.csv")

    sns.lineplot(
        data=w_hopping,
        x="t",
        y="snr"
    )


    sns.lineplot(
        data=wo_hopping,
        x="t",
        y="snr"
    )
    plt.show()