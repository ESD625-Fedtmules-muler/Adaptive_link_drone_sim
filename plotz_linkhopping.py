import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    df_dir = pd.read_csv("Directional_path.csv")
    df_hop = pd.read_csv("Directional_link_hopping.csv")


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    df_hop = df_hop[(df_hop["time"] > 120) & (df_hop["time"] < 320)]
    df_dir = df_dir[(df_dir["time"] > 120) & (df_dir["time"] < 320)]

    sns.lineplot(
        data=df_hop, 
        x="time", 
        y="d2b", 
        label="Drone to base", 
        errorbar=("sd", 2),
        marker=",",      # pixel marker
        markersize=2,    # very small
        ax=axes[0,0]
        )
    sns.scatterplot(
        data=df_hop, 
        x="time", 
        y="d2b", 
        label="Sample", 
        s=1,
        ax=axes[0,0]
        )

    sns.lineplot(
        data=df_hop, 
        x="time", 
        y="b2d", 
        label="Base to drone", 
        errorbar=("sd", 2),
        marker=",",      # pixel marker
        markersize=2,    # very small
        ax=axes[0,0]
    )
    sns.scatterplot(
        data=df_hop, 
        x="time", 
        y="b2d", 
        label="Sample", 
        s=1,
        ax=axes[0,0]
        )

    sns.lineplot(
        data=df_dir, 
        x="time", 
        y="d2b", 
        label="Drone to base", 
        errorbar=("sd", 2),
        marker=",",      # pixel marker
        markersize=2,    # very small
        ax=axes[0,1]
        )
    sns.scatterplot(
        data=df_dir, 
        x="time", 
        y="d2b", 
        label="Sample", 
        s=1,
        ax=axes[0,1]
        )
    sns.lineplot(
        data=df_dir, 
        x="time", 
        y="b2d", 
        label="Base to drone", 
        errorbar=("sd", 2),
        marker=",",      # pixel marker
        markersize=2,    # very small
        ax=axes[0,1]
        )
    sns.scatterplot(
        data=df_dir, 
        x="time", 
        y="b2d", 
        label="Sample", 
        s=1,
        ax=axes[0,1]
        )
    

    sns.histplot(data=df_hop, x="d2b", 
                legend="Drone to base", 
                bins=100,
                kde=False,
                stat="count",
                alpha=0.6,
                element="step",
                ax=axes[1, 0],
                )
    
    sns.histplot(data=df_hop, x="b2d", 
                legend="Base to drone",
                bins=100,
                kde=False,
                stat="count",
                alpha=0.6,
                element="step",
                ax=axes[1, 0],
                )


    sns.histplot(data=df_dir, x="d2b", 
                legend="Base to drone", 
                bins=100,
                kde=False,
                stat="count",
                alpha=0.6,
                element="step",
                ax=axes[1, 1],
                )
    
    sns.histplot(data=df_dir, x="b2d", 
                legend="Base to drone",
                bins=100,
                kde=False,
                stat="count",
                alpha=0.6,
                element="step",
                ax=axes[1, 1],
                )



    axes[0,0].set_title("Directional link hopping")
    axes[0,1].set_title("Directional direct path")
    axes[0,0].grid(True)
    axes[0,1].grid(True)
    axes[0,0].set_ylim(-20, 15)
    axes[0,1].set_ylim(-20, 15)
    axes[1,0].set_xlim(-20, 20)
    axes[1,1].set_xlim(-20, 20)



    plt.show()