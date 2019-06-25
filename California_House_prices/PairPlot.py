#                  ____       _              _       _
#                 |  _ \ __ _(_)_ __   _ __ | | ___ | |_ ___
#                 | |_) / _` | | '__| | '_ \| |/ _ \| __/ __|
#                 |  __/ (_| | | |    | |_) | | (_) | |_\__ \
#                 |_|   \__,_|_|_|    | .__/|_|\___/ \__|___/
#                                     |_|
# Customized pair plots, useful for data exploration
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd


def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    p_stars = ""
    if p <= 0.05:
        p_stars = "*"
    if p <= 0.01:
        p_stars = "**"
    if p <= 0.001:
        p_stars = "***"
    ax = plt.gca()
    ax.annotate(r'$\rho$ = {:.2f} '.format(r) + p_stars, xy=(0.05, 0.9), xycoords=ax.transAxes)


def corrdot(*args, **kwargs):
    corr_r = args[0].corr(args[1], 'pearson')
    corr_text = f"{corr_r:2.2f}".replace("0.", ".")
    ax = plt.gca()
    ax.set_axis_off()
    marker_size = abs(corr_r) * 10000
    ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.6, cmap="coolwarm",
               vmin=-1, vmax=1, transform=ax.transAxes)
    font_size = abs(corr_r) * 40 + 5
    ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                ha='center', va='center', fontsize=font_size)


def annotate_colname(x, **kws):
    print(x, kws)
    ax = plt.gca()
    ax.annotate(kws.get('label'), xy=(0.05, 0.9), xycoords=ax.transAxes, fontweight="bold")


def cor_matrix(df):
    g = sns.PairGrid(df, palette=["red"])
    # Use normal regplot as `lowess=True` doesn't provide CIs.
    g.map_upper(sns.regplot, scatter_kws={"s": 10})
    g.map_diag(sns.distplot)
    g.map_diag(annotate_colname)
    g.map_lower(sns.kdeplot, cmap="Blues_d")
    g.map_lower(corrfunc)
    # Remove axis labels, as they're in the diagonals.
    return g


def reorder_df(df, main_col, num_cols):
    if main_col in df.columns:
        main_col = 'median_house_value'
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index(main_col)))
        df = df.reindex(columns=cols)
        # Reorder with respect to main_col the columns with higher correlation
        abs_correlations = df.corr()[main_col].abs()
        abs_correlations_sorted = abs_correlations.sort_values(ascending=False)
        df = df.reindex(columns=abs_correlations_sorted.index)
        if num_cols and num_cols <= len(df.columns):
            df = df.iloc[:, :num_cols]
    else:
        print(f'WARNING: Column {main_col} not found in dataframe')
    return df


def cat_plot(df, main_col):
    cat_cols = df.select_dtypes(exclude=['number']).columns
    if main_col not in cat_cols:  # if main col is not categorical, but numerical => OK
        for cat_col in cat_cols:
            # Bar plot ordered by count
            sns.catplot(x=cat_col, kind='count', data=df, height=5,
                        order=df[cat_col].value_counts().index)
            plt.title(f'Distribution of {cat_col}')
            plt.show()
            # Vioilin plot: main_col by cat_col
            sns.catplot(x=cat_col, y=main_col, data=df, kind='violin', height=8, orient='v')
            plt.title(f'{main_col} distribution by {cat_col}')
            plt.show()


def pair_plot(df_in, sample=None, target=None, num_cols=None, hue=None, remove_names=False):
    # Prepare data
    df_sampled = df_in.dropna().sample(sample) if sample is not None else df_in
    df = reorder_df(df_sampled, target, num_cols) if target else df_sampled
    # Plot
    g = sns.PairGrid(df, aspect=1.4, diag_sharey=False, dropna=True, height=4)
    g.map_upper(corrfunc)
    g.map_upper(sns.regplot, scatter_kws={"s": 12}, line_kws={'color': 'tab:grey'})
    g.map_upper(sns.kdeplot, cmap="Reds", alpha=0.3)
    g.map_diag(sns.distplot, hist_kws={'edgecolor': 'w'})
    g.map_lower(corrdot)
    # Remove axis labels
    [(ax.set_ylabel(''), ax.set_xlabel('')) for ax in g.axes.flatten()] if remove_names else None
    # Add titles to the diagonal axes/subplots
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, fontsize=16)
    # Call cat_plot
    cat_plot(df_sampled, target)


def joint_plot(df_in, x, y, sample=None, height=10):
    df = df_in.sample(sample) if sample else df_in
    g = sns.JointGrid(x=x, y=y, data=df, height=height)
    g.plot_joint(corrfunc)  # pearson correlation coeficient
    g.plot_joint(sns.regplot, scatter_kws={"s": 12}, line_kws={'color': 'tab:grey'})
    g.plot_joint(sns.kdeplot, cmap="Reds", alpha=0.4)
    g.plot_marginals(sns.distplot, hist_kws={'edgecolor': 'w'})


def hist_plots(df, col_wrap=3):
    num_cols = df.select_dtypes(['number']).columns
    df_melted = pd.melt(df, value_vars=num_cols)
    g = sns.FacetGrid(df_melted, col="variable",  col_wrap=col_wrap, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
