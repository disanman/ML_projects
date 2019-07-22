#                 ____                              _
#                |  _ \ ___  __ _ _ __ ___  ___ ___(_) ___  _ __
#                | |_) / _ \/ _` | '__/ _ \/ __/ __| |/ _ \| '_ \
#                |  _ <  __/ (_| | | |  __/\__ \__ \ | (_) | | | |
#                |_| \_\___|\__, |_|  \___||___/___/_|\___/|_| |_|
#                           |___/
#     ____  _                             _   _              _       _
#    |  _ \(_) __ _  __ _ _ __   ___  ___| |_(_) ___   _ __ | | ___ | |_ ___
#    | | | | |/ _` |/ _` | '_ \ / _ \/ __| __| |/ __| | '_ \| |/ _ \| __/ __|
#    | |_| | | (_| | (_| | | | | (_) \__ \ |_| | (__  | |_) | | (_) | |_\__ \
#    |____/|_|\__,_|\__, |_| |_|\___/|___/\__|_|\___| | .__/|_|\___/ \__|___/
#                   |___/                             |_|
# This is a modified version of the code found here:
# https://robert-alvarez.github.io/2018-06-04-diagnostic_plots/
# The code was adjusted so the four plots can be visualized in a complete plot + subplots
# To understand better the diagnostic plots, check: https://data.library.virginia.edu/diagnostic-plots/
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.gofplots import ProbPlot
import warnings


# Style settings
sns.set_style('dark')
plt.rc('font', size=10)
plt.rc('figure', titlesize=12)
plt.rc('axes', labelsize=10)
plt.rc('axes', titlesize=10)


def diagnostic_plots(model, figsize=(12, 7)):
    warnings.simplefilter('ignore')
    # Defining subplots
    plot, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()

    # Plot 1
    axes[0] = sns.residplot(
        model.fittedvalues,
        model.resid + model.fittedvalues,    # residuals = target - predicted => target = resid + predicted
        lowess=True,
        scatter_kws={"alpha": 0.5},
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=axes[0]
    )
    axes[0].set_title("Residuals vs Fitted")
    axes[0].set_xlabel("Fitted values")
    axes[0].set_ylabel("Residuals")
    # annotations
    abs_resid = model.resid.abs().sort_values(ascending=False)
    for i in abs_resid[:3].index:
        axes[0].annotate(i, xy=(model.fittedvalues[i], model.resid[i]))

    # Plot 2
    # normalized residuals
    model_norm_residuals = model.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    QQ = ProbPlot(model_norm_residuals)
    QQ.qqplot(line="45", alpha=0.5, color="#4C72B0", lw=1, ax=axes[1])
    axes[1].set_title("Normal Q-Q")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Standardized Residuals")
    # annotations
    abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)
    abs_norm_resid_top_3 = abs_norm_resid[:3]
    for r, i in enumerate(abs_norm_resid_top_3):
        axes[1].annotate(
            i, xy=(np.flip(QQ.theoretical_quantiles, 0)[r], model_norm_residuals[i])
        )

    # Plot 3
    axes[2] = sns.scatterplot(model.fittedvalues, model_norm_residuals_abs_sqrt, alpha=0.5, ax=axes[2])
    axes[2] = sns.regplot(
        model.fittedvalues,
        model_norm_residuals_abs_sqrt,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=axes[2]
    )
    axes[2].set_title("Scale-Location")
    axes[2].set_xlabel("Fitted values")
    axes[2].set_ylabel("$\sqrt{|Standardized Residuals|}$")
    # annotations
    abs_sq_norm_resid = np.flip(np.argsort(model_norm_residuals_abs_sqrt), 0)
    abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
    for i in abs_sq_norm_resid_top_3:
        axes[2].annotate(
            i, xy=(model.fittedvalues[i], model_norm_residuals_abs_sqrt[i])
        )

    # Plot 4
    # leverage, from statsmodels internals
    model_leverage = model.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model.get_influence().cooks_distance[0]
    sns.scatterplot(model_leverage, model_norm_residuals, alpha=0.5, ax=axes[3])
    sns.regplot(
        model_leverage,
        model_norm_residuals,
        scatter=False,
        ci=False,
        lowess=True,
        line_kws={"color": "red", "lw": 1, "alpha": 0.8},
        ax=axes[3]
    )
    axes[3].set_xlim(0, max(model_leverage) + 0.01)
    axes[3].set_ylim(-3, 5)
    axes[3].set_title("Residuals vs Leverage")
    axes[3].set_xlabel("Leverage")
    axes[3].set_ylabel("Standardized Residuals")

    # annotations
    leverage_top_3 = np.flip(np.argsort(model_cooks), 0)[:3]
    for i in leverage_top_3:
        axes[3].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]))

    p = len(model.params)  # number of model parameters

    # line 1
    x = np.linspace(0.001, max(model_leverage), 50)
    f = lambda x: np.sqrt((0.5 * p * (1 - x)) / x)
    y = f(x)
    sns.lineplot(x, y, label="Cook's distance", ax=axes[3], color='red', dashes=True)
    axes[3].lines[1].set_linestyle("--")
    # line 2
    x = np.linspace(0.001, max(model_leverage), 50)
    f = lambda x: np.sqrt((1 * p * (1 - x)) / x)
    y = f(x)
    sns.lineplot(x, y, ax=axes[3], color='red', dashes=True)
    axes[3].legend(loc='upper right')

    plot.tight_layout()  # so titles won't overlap x_labels
    plot.show()
