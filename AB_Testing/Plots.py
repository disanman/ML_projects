#                         ____  _       _
#                        |  _ \| | ___ | |_ ___
#                        | |_) | |/ _ \| __/ __|
#                        |  __/| | (_) | |_\__ \
#                        |_|   |_|\___/ \__|___/
#
# Customized pair plots, useful for data exploration
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, norm
import warnings


# Style settings
sns.set_style('dark')
plt.rc('font', size=10)
plt.rc('figure', titlesize=12)
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)


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
    ax.annotate(corr_text, [.5, .5],  xycoords="axes fraction",
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


def reorder_df(df, target, num_cols):
    if target in df.columns:
        cols = list(df.columns)
        cols.insert(0, cols.pop(cols.index(target)))
        df = df.reindex(columns=cols)
        # Reorder with respect to main_col the columns with higher correlation
        abs_correlations = df.corr()[target].abs()
        abs_correlations_sorted = abs_correlations.sort_values(ascending=False)
        df = df.reindex(columns=abs_correlations_sorted.index)
        if num_cols and num_cols <= len(df.columns):
            df = df.iloc[:, :num_cols]
    else:
        print(f'WARNING: Column {target} not found in dataframe')
    return df


def cat_plot(df, target):
    cat_cols = df.select_dtypes(exclude=['number']).columns
    if target not in cat_cols:  # if main col is not categorical, but numerical => OK
        for cat_col in cat_cols:
            # Bar plot ordered by count
            sns.catplot(x=cat_col, kind='count', data=df, height=5,
                        order=df[cat_col].value_counts().index)
            plt.title(f'Distribution of {cat_col}')
            plt.show()
            # Vioilin plot: main_col by cat_col
            sns.catplot(x=cat_col, y=target, data=df, kind='violin', height=8, orient='v')
            plt.title(f'{target} distribution by {cat_col}')
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
    plt.show()


def hist_plots(df, col_wrap=3):
    num_cols = df.select_dtypes(['number']).columns
    df_melted = pd.melt(df, value_vars=num_cols)
    g = sns.FacetGrid(df_melted, col="variable",  col_wrap=col_wrap, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")


def corr_plot(data, target=None, figsize=(7,  7)):
    ''' Creates a correlation plot of all the variables present in the input dataframe
    Args:
       - data (DataFrame): contains all the data to be plotted
       - target (str):     name of the target column, if given, a second correlation plot will be made
                           by sorting all the other columns from higher to lower correlation with respect
                           to this target variable
    '''
    corr_matrix = data.corr()
    # Create mask: used to show only the lower part of the correlation matrix
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=figsize)
    if target is not None and target in data.columns:
        cols = corr_matrix[target].sort_values(ascending=False).index.values
        sns.heatmap(corr_matrix[cols].sort_values(by=target, ascending=False), cbar=True, annot=True, fmt='.2f',
                    annot_kws={'size': 7}, yticklabels=cols, xticklabels=cols,
                    cmap="RdBu_r", mask=mask)
        plt.title(f'Correlation plot: sorted by {target}')
    else:
        sns.heatmap(corr_matrix, cbar=True, annot=True, fmt='.2f',
                    annot_kws={'size': 7}, cmap="RdBu_r", mask=mask)
        plt.title('Correlation plot')
    plt.show()


def get_stars(rho_in):
    ''' Gets number of stars related with correlation '''
    rho = np.abs(rho_in)
    stars = ''
    if rho <= 0.05:
        stars = "*"
    if rho <= 0.01:
        stars = "**"
    if rho <= 0.001:
        stars = "***"
    return stars


def reg_plots(data, target, num_cols=3, min_abs_rho=0, figsize=(4, 4)):
    ''' Creates a matrix of regression plots for each column in data with respect to target
    Args:
       - data (DataFrame):     contains all the data to be plotted
       - target (str):         name of the target column, all the reg_plots will be done against this variable
       - min_abs_rho (float):  minimum correlation threshold against the target a variable should have to be plotted
       - figsize (tuple):      figure size of each individual plot
    '''
    warnings.simplefilter('ignore')
    assert target in data.columns, 'Target column not found in input DataFrame'
    corr_matrix = data.corr()
    correlations = corr_matrix[target].sort_values(ascending=False).drop(target)
    correlations = correlations[correlations.abs() >= min_abs_rho]
    # Define figure
    num_rows = int(np.ceil(len(correlations) / num_cols))
    fig_size = (num_cols * figsize[0], num_rows * figsize[1])
    plot, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    axes = axes.ravel()
    for idx, col in enumerate(correlations.index):   # iterates over columns
        p = correlations[col]  # correlation of column col vs. target
        sns.regplot(data[col], data[target], ax=axes[idx],  line_kws={'color': 'tab:grey'})
        sns.kdeplot(data[col], data[target], cmap='Reds', alpha=0.3, ax=axes[idx])
        anotation = r'$\rho$ = ' + f'{p:.2f}'
        # axes[idx].annotate(anotation, xy=(0.75, 0.9), xycoords=axes[idx].transAxes)
        axes[idx].set_title(f'{target} vs. {col}, {anotation}')
    plot.tight_layout()  # so titles won't overlap x_labels
    plt.show()


def distplot_stats(x, fit=None, title=''):
    sns.distplot(x, fit=fit)
    plt.title(f'Skewness: {skew(x):.2f}, Kurtosis: {kurtosis(x):.2f}')
    plt.suptitle(title)
    plt.show()


def get_value_counts(df, cols, verbose=0):
    counts = {}
    for col in cols:
        value_counts = pd.DataFrame(df[col].value_counts())
        value_counts['freq'] = value_counts / value_counts.sum() * 100
        value_counts['freq'] = value_counts['freq'].map('{:.2f}%'.format)   # map a text format
        value_counts = value_counts.reset_index().rename(columns={col: 'count', 'index': 'value'})
        if verbose > 0:
            print(f'Value counts for column \'{col}\':')
            print(f'There are {len(value_counts)} unique cases, top 5: ', value_counts[:5],
                  sep='\n', end='\n------------------------\n\n')
        counts[col] = value_counts
    return counts


def plot_value_counts(df, cols, top=5, num_cols=3, figsize=(4, 3)):
    counts = get_value_counts(df, cols)
    # Define figure
    num_rows = int(np.ceil(len(cols) / num_cols))
    fig_size = (num_cols * figsize[0], num_rows * figsize[1])
    plot, axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
    axes = axes.ravel()
    for idx, (col, count) in enumerate(counts.items()):   # iterates over columns
        sns.barplot(x='count', y='value', data=count[:top], order=count.loc[:top - 1, 'value'],
                    orient='h', palette='Blues_d', ax=axes[idx])
        axes[idx].set_title(f'Column: \'{col}\'\ntop {min(top, len(count))} of {len(count)} unique cases')
        axes[idx].set(xlim=(0, count.loc[0, 'count'] * 1.3))   # add some space for the text labels
        axes[idx].set(xticks=[])   # remove xticks
        # Annotate bars (text labels)
        for p in axes[idx].patches:
            width = p.get_width()
            axes[idx].text(p.get_width(), p.get_y() + 0.55 * p.get_height(),
                           f'{width:.0f}\n({width/sum(count["count"]) * 100:2.2f})%', ha='left', va='center', size=7)
    plot.tight_layout()  # so titles won't overlap x_labels
    plt.suptitle(f'Top {top} unique cases for each column in the dataset', y=1.02)
    plt.show()


def count_plot(array, title='', add_labels=True, ax=None, show=True):
    ''' Creates a bar plot with the counts of an array
    Args:
      - array (list, np.array1d or series): input array
      - title (str): title to the graph
      - add_labels (bool): wheter to add text labels (counts and percentage)
      - ax: in case this plot is used as a subplot, pass the ax object where to plot it (check plt.subplots help)
      - show: when doin subplots, in that case the plt.show() should be called after all subplots have been plotted
    '''
    plot = sns.countplot(array, color='steelblue', ax=ax)
    if add_labels:
        max_height = 0
        for p in plot.patches:
            height = p.get_height()
            max_height = max(max_height, height)
            bar_text = f'{height:.0f}\n({height / len(array) * 100:.2f}%)'
            plot.annotate(bar_text, (p.get_x() + p.get_width() / 2., p.get_height()), size=7,
                          ha='center', va='center', xytext=(0, 10), textcoords='offset points')
        plot.set(ylim=(0, max_height * 1.1), yticks=[])   # add some space for the text labels, remove y ticks
    ax.set_title(title) if ax else plt.title(title)   # set plot title for subplot or normal plot
    plt.show() if show else None


def zplot(area=0.95, two_tailed=True, align_right=False):
    """Plots a z distribution with common annotations
    Example:
        zplot(area=0.95)
        zplot(area=0.80, two_tailed=False, align_right=True)
    Parameters:
        area (float): The area under the standard normal distribution curve.
        align (str): The area under the curve can be aligned to the center
            (default) or to the left.
    Returns:
        None: A plot of the normal distribution with annotations showing the
        area under the curve and the boundaries of the area.
    """
    # create plot object
    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()
    # create normal distribution
    x = np.linspace(-5, 5, 1000)
    y = norm.pdf(x)
    ax.plot(x, y)
    # ------------------------------------------------
    # code to fill areas:
    # for two-tailed tests
    if two_tailed:
        left = norm.ppf(0.5 - area / 2)
        right = norm.ppf(0.5 + area / 2)
        ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
        ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
        ax.fill_between(x, 0, y, color='grey', alpha='0.25', where=(x > left) & (x < right))
        plt.xlabel('z')
        plt.ylabel('PDF')
        plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left), fontsize=12, rotation=90, va="bottom", ha="right")
        plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right), fontsize=12, rotation=90, va="bottom", ha="left")
    # for one-tailed tests
    else:
        # align the area to the right
        if align_right:
            left = norm.ppf(1-area)
            ax.vlines(left, 0, norm.pdf(left), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25', where=x > left)
            plt.text(left, norm.pdf(left), "z = {0:.3f}".format(left),
                     fontsize=12, rotation=90, va="bottom", ha="right")
        # align the area to the left
        else:
            right = norm.ppf(area)
            ax.vlines(right, 0, norm.pdf(right), color='grey', linestyle='--')
            ax.fill_between(x, 0, y, color='grey', alpha='0.25', where=x < right)
            plt.text(right, norm.pdf(right), "z = {0:.3f}".format(right),
                     fontsize=12, rotation=90, va="bottom", ha="left")
    # annotate the shaded area
    plt.text(0, 0.1, "shaded area = {0:.3f}".format(area), fontsize=12, ha='center')
    # axis labels
    plt.xlabel('z')
    plt.ylabel('PDF')
    plt.show()


def plot_norm_dist(ax, mu=0, std=1, label=None):
    ''' Plots a normal distribution to the axis provided
    Args:
        - ax: (matplotlib axes)
        - mu: (float) mean of the normal distribution
        - std: (float) standard deviation of the normal distribution
        - label: (string or None) label name for the plot'''
    x = np.linspace(mu - 12 * std, mu + 12 * std, 1000)
    y = norm(mu, std).pdf(x)
    ax.plot(x, y, label=label)
