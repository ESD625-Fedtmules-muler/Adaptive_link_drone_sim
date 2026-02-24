import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":

    df_iso_whole = pd.read_csv("Isotropic_path.csv")
    df_dir_whole = pd.read_csv("Directional_path.csv")

    df_iso = df_iso_whole[(df_iso_whole["time"] > 120) & (df_iso_whole["time"] < 320)]
    df_dir = df_dir_whole[(df_dir_whole["time"] > 120) & (df_dir_whole["time"] < 320)]

    df_iso_whole = df_iso_whole[df_iso_whole["time"] < 320]
    df_dir_whole = df_dir_whole[df_dir_whole["time"] < 320]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    weights_iso = np.max(df_iso["time"])/len(df_iso["time"])
    weights_dir = np.max(df_dir["time"])/len(df_dir["time"])



    ###Isotropisk antenne
    sns.lineplot(
        data=df_iso_whole, 
        x="time", 
        y="d2b", 
        label="Drone to base", 
        errorbar=("sd", 2),
        ax=axes[0,0]
        )
    sns.scatterplot(
        data=df_iso_whole, 
        x="time", 
        y="d2b", 
        label="Sample", 
        s=1,
        ax=axes[0,0]
        )

    sns.lineplot(
        data=df_iso_whole, 
        x="time", 
        y="b2d", 
        label="Base to drone", 
        errorbar=("sd", 2),
        ax=axes[0,0]
        )
    sns.scatterplot(
        data=df_iso_whole, 
        x="time", 
        y="b2d", 
        label="Sample", 
        s=1,
        ax=axes[0,0]
        )

    ######Direktionel antenne
    sns.lineplot(
        data=df_dir_whole, 
        x="time", 
        y="d2b", 
        label="Drone to base", 
        errorbar=("sd", 2),
        ax=axes[0,1]
        )
    sns.scatterplot(
        data=df_dir_whole, 
        x="time", 
        y="d2b", 
        label="Sample", 
        s=1,
        ax=axes[0,1]
        )
    sns.lineplot(
        data=df_dir_whole, 
        x="time", 
        y="b2d", 
        label="Base to drone", 
        errorbar=("sd", 2),
        ax=axes[0,1]
        )
    sns.scatterplot(
        data=df_dir_whole, 
        x="time", 
        y="b2d", 
        label="Sample", 
        s=1,
        ax=axes[0,1]
        )
    sns.histplot(data=df_iso, x="d2b", 
                 legend="Drone to base", 
                 weights=weights_iso, 
                 bins=100,
                 kde=False,
                 stat="count",
                 alpha=0.6,
                 element="step",
                 ax=axes[1, 0],
                 )
    sns.histplot(data=df_iso, 
                 x="b2d", 
                 legend="Base to drone", 
                 weights=weights_iso, 
                 bins=100,
                 kde=False,
                 stat="count",
                 alpha=0.6,
                 element="step",
                 ax=axes[1, 0]
                 )
    sns.histplot(data=df_dir, x="d2b", 
                 legend="Drone to base", 
                 weights=weights_dir, 
                 bins=100,
                 kde=False,
                 stat="count",
                 alpha=0.6,
                 element="step",
                 ax=axes[1, 1],
                 )
    sns.histplot(data=df_dir, 
                 x="b2d", 
                 legend="Base to drone", 
                 weights=weights_dir, 
                 bins=100,
                 kde=False,
                 stat="count",
                 alpha=0.6,
                 element="step",
                 ax=axes[1, 1]
                 )
    plt.tight_layout()
    
    axes[1,0].legend(["Drone to base", "Base to drone"])
    axes[1,0].set_ylabel("Time [s]")
    axes[1,0].set_xlim(-30, 10)
    axes[1,0].set_ylim(0, 10)

    
    axes[1,1].legend(["Drone to base", "Base to drone"])
    axes[1,1].set_ylabel("Time [s]")
    axes[1,1].set_xlim(-30, 10)
    axes[1,1].set_ylim(0, 10)
    

    
    axes[0,0].grid(True)
    axes[0,1].grid(True)
    axes[0,0].set_ylim(-30, 15)
    axes[0,1].set_ylim(-30, 15)


    axes[1,0].grid(True)
    axes[1,1].grid(True)

    plt.ylabel("Time [s]")

    plt.show()
