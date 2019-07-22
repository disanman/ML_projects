#                     _____                                _
#                    |  ___|__  _ ____      ____ _ _ __ __| |
#                    | |_ / _ \| '__\ \ /\ / / _` | '__/ _` |
#                    |  _| (_) | |   \ V  V / (_| | | | (_| |
#                    |_|  \___/|_|    \_/\_/ \__,_|_|  \__,_|
#
#                     ____       _           _   _
#                    / ___|  ___| | ___  ___| |_(_) ___  _ __
#                    \___ \ / _ \ |/ _ \/ __| __| |/ _ \| '_ \
#                     ___) |  __/ |  __/ (__| |_| | (_) | | | |
#                    |____/ \___|_|\___|\___|\__|_|\___/|_| |_|
# Forward selection using Statsmodels formula
import statsmodels.formula.api as smf
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DiagnosticPlots import diagnostic_plots
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


class ForwardSelector():

    def __init__(self, data, target, poly_grade=0, poly_features=[], other_features=[]):
        assert target in data, f'Target variable {target} not present in input DataFrame'
        self.data = data
        self.target = target
        self.remaining_features = list(data.columns.difference([target]))
        self.remaining_features.extend(other_features)
        self.removed_features = []
        if poly_grade > 1:
            self._get_poly_features(poly_grade, poly_features)
        # train M_0 model
        self.formula = f'{target} ~ 1'
        self.models = []
        lm = smf.ols(formula=self.formula, data=self.data).fit()   # ols: Ordinary Least Squares linear reg.
        self.models.append(lm)
        # Initialize best RMSE
        self.best_RMSE = np.sqrt(lm.mse_resid)
        # Initialize model stats historic
        self.stats = dict(rmse=[], bic=[], rsquared=[])
        self._get_model_stats(lm, verbose=0)

    def poly(self, col, grade=0):
        return [f'np.power({col},' + str(i) + ')' for i in range(2, grade + 1)]

    def _get_poly_features(self, poly_grade, poly_features):
        if poly_features == []:
            poly_features = self.remaining_features.copy()
        for feature in poly_features:
            self.remaining_features.extend(self.poly(feature, poly_grade))
        print('Extending polinomials, list of features to test: ')
        print(self.remaining_features)

    def _get_model_stats(self, model, verbose=1):
        rmse = np.sqrt(model.mse_resid)
        if verbose > 0:
            print(f'  BIC:  {model.bic:4.2f}')
            print(f'  RMSE: {rmse:4.2f}')
            print(f'  R^2:  {model.rsquared * 100:4.2f}%')
        # save stats()
        self.stats['rmse'].append(rmse)
        self.stats['bic'].append(model.bic)
        self.stats['rsquared'].append(model.rsquared)

    def _get_vif(self, formula, data):
        # get y and X dataframes based on the formula previously defined:
        y, X = dmatrices(formula, data, return_type='dataframe')
        # For each X, calculate VIF and save in dataframe
        vif = pd.DataFrame()
        vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif["features"] = X.columns
        return vif

    def print_vif(self):
        vif = self._get_vif(self.final_formula, self.data)
        print('\nVariance Inflation Factors (VIF): \n', vif)

    def forward_select_step(self, verbose=1, remove_low_pvalue=True):
        print(f'Initial formula: {self.formula}') if verbose > 1 else None
        # over all the features, select the one that improves the most the residual standard error
        best_feature = None
        best_model = None
        for feature in self.remaining_features:
            formula = self.formula + f' + {feature}'
            print(f'Formula: {formula}') if verbose > 1 else None
            lm = smf.ols(formula=formula, data=self.data).fit()   # ols: Ordinary Least Squares linear reg.
            rmse = np.sqrt(lm.mse_resid)
            if rmse < self.best_RMSE:
                print(f'  Best RMSE found: {rmse}') if verbose > 1 else None
                self.best_RMSE = rmse
                best_model = lm
                best_feature = feature
        if best_feature is None:
            print('No further feature improves RMSE') if verbose > 0 else None
            return False
        else:
            best_formula = self.formula + f' + {best_feature}'
            # Check if all p-values are significant, if not => remove the variable (don't use it)
            if not all(best_model.pvalues[1:] < 0.05):   # not using the p-value of the intercept here
                if verbose > 0:
                    print('Warning!: not all p-values are significative, this might be due to multicollinearity')
                    print(best_model.pvalues[best_model.pvalues > 0.05])
                    print('Variance Inflation Factors (VIF): \n', self._get_vif(best_formula, self.data))
                if remove_low_pvalue:
                    print(f'Feature {best_feature} was removed due to low p-value!')
                    self.removed_features.append(best_feature)
                    return True
            self.formula = best_formula
            print(f'Updated model: {self.formula}') if verbose > 0 else None
            self.models.append(best_model)
            self._get_model_stats(best_model, verbose)
            self.remaining_features.remove(best_feature)
            return True

    def plot_historic(self, figsize=(15, 6)):
        plot, axes = plt.subplots(1, 3, figsize=figsize)
        axes = axes.ravel()
        x = range(len(self.models))
        # Plot 1: RMSE
        sns.lineplot(x, self.stats['rmse'], ax=axes[0])
        axes[0].set_title("RMSE")
        axes[0].set_xlabel("Number of variables used")
        axes[0].set_ylabel("RMSE")
        # Plot 2: BIC
        sns.lineplot(x, self.stats['bic'], ax=axes[1])
        axes[1].set_title("BIC")
        axes[1].set_xlabel("Number of variables used")
        axes[1].set_ylabel("BIC")
        # Plot 3: Rsquared
        sns.lineplot(x, self.stats['rsquared'], ax=axes[2])
        axes[2].set_title("$R^2$")
        axes[2].set_xlabel("Number of variables used")
        axes[2].set_ylabel("R^2")

    def print_anova_results(self):
        # Print anova results
        anova_results = sm.stats.anova_lm(*self.models)
        print('\nANOVA results:')
        anova_results['Pr(>F)'] = anova_results['Pr(>F)'].map('{:,.3f}'.format)  # format pvalues
        print(anova_results)

    def print_summary(self, figsize=(15, 10)):
        # Print summary of the best model found (the last one)
        print('\nBest model found: ', self.final_formula)
        print('\nNot used features: ', self.remaining_features)
        if len(self.removed_features) > 0:
            print('\nRemoved features (due to low p-values): ', self.removed_features)
        print('\n\n', self.final_model.summary())
        diagnostic_plots(self.final_model, figsize=figsize)
        self.plot_historic()
        self.print_vif()
        self.print_anova_results()

    def forward_select(self, verbose=1, remove_low_pvalue=True):
        loop = True
        while loop:
            loop = self.forward_select_step(verbose=verbose, remove_low_pvalue=True)
        self.final_model = self.models[-1]
        self.final_formula = self.formula
        if verbose > 0:
            self.print_summary()
