#                    _    ____    _____         _
#                   / \  | __ )  |_   _|__  ___| |_ ___ _ __
#                  / _ \ |  _ \    | |/ _ \/ __| __/ _ \ '__|
#                 / ___ \| |_) |   | |  __/\__ \ ||  __/ |
#                /_/   \_\____/    |_|\___||___/\__\___|_|
# References
# Google-Udacity AB Testing course
# https://towardsdatascience.com/the-math-behind-a-b-testing-with-example-code-part-1-of-2-7be752e1d06f
# https://stackoverflow.com/questions/15204070/is-there-a-python-scipy-function-to-determine-parameters-needed-to-obtain-a-ta/18379559#18379559
import numpy as np
from scipy.stats import norm, binom
import logging
import statsmodels.stats.api as sms
# import statsmodels.stats.power as smp   # not used
import matplotlib.pyplot as plt
import seaborn as sns
from Plots import plot_norm_dist
from collections import namedtuple


# logger.setLevel(logging.DEBUG)
LOG_FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


class ABTester:
    ''' Class AB Tester, used to perform AB Tests and plot the results! '''

    def __init__(self, A, B=None, significance=0.05, power=0.8, two_sided=True):
        ''' Initializes the AB Tester object
        Args:
            - A, B: (dict) dictionaries with two key-values: conversions and impressions (both int)
            - significance level: (float) alpha
            - power: (float) 1-beta
            - two_sided: (bool) wheter a two sided test is used or not (single-sided)
        '''
        self.A = A
        self.B = B
        self.significance = significance
        self.power = power
        self.two_sided = two_sided
        self.A['p_hat'] = self.A['conversions'] / self.A['impressions']
        if B:
            self.B['p_hat'] = self.B['conversions'] / self.B['impressions']
            self.AB_stats = self._get_AB_test_stats()    # it will be a namedtuple for easier access with . notation

    def _get_AB_test_stats(self):
        '''Returns some stats used in an AB test:
              - d_hat: estimated difference among the groups
              - pooled_prob: pooled probability of the two samples
              - pooled_se: pooled standard error of the two samples
              - confidence_interval: at the significance level set in initialization '''
        pooled_prob = (self.A['conversions'] + self.B['conversions']) / (self.A['impressions'] + self.B['impressions'])
        pooled_standard_error = np.sqrt(pooled_prob * (1 - pooled_prob) *
                                        (1 / self.A['impressions'] + 1 / self.B['impressions']))
        d_hat = self.B["p_hat"] - self.A["p_hat"]
        # Finding confidence interval
        prob = self.significance / 2 if self.two_sided else self.significance   # alpha/2 if two-sided
        z = -norm.ppf(prob)
        margin_of_error = z * pooled_standard_error
        left = d_hat - margin_of_error
        right = d_hat + margin_of_error
        confidence_interval = (left, right)
        # Get p-value of the AB test
        p_value = binom(self.A['impressions'], self.A['p_hat']).pmf(self.B['impressions'] * self.B['p_hat'])
        # Gather result stats
        Stats = namedtuple(typename='Stats', field_names='d_hat, pooled_prob, pooled_se, confidence_interval, p_value')
        return Stats(d_hat, pooled_prob, pooled_standard_error, confidence_interval, p_value)

    def print_AB_results(self):
        logger.info(f'Estimated difference, d_hat: {self.AB_stats.d_hat}')
        # logger.info(f'pooled_prob: {pooled_prob:.2%}, z: {z:.2f}, margin of error: {margin_of_error:.2%}')
        left, right = self.AB_stats.confidence_interval
        logger.info(f'Confidence interval: ({left:.2%}, {right:.2%})')
        print(self.AB_stats)

    def __str__(self):
        return 'string representation'   # TODO

    def get_variant_confidence_interval(self, variant='A'):
        ''' Calculates the confidence interval of a Binomial distribution of a variant defined by:
        Args:
            - variant: (string) 'A' or 'B'. Represents the variant defined in object initialization
                        the variant contains: impressions and conversions
        Returns:
            - confidence_interval: (tuple of 2 floats) with min and max interval values
        '''
        data = self.A if variant == 'A' else self.B
        impressions, p_hat = data['impressions'], data['p_hat']
        if impressions * p_hat < 5:
            logger.warning('Warning: the normal approximation does not hold')
        standard_error = np.sqrt(p_hat * (1-p_hat) / impressions)     # assumes it is distributed as a binomial
        z = self._get_z_val()
        margin_of_error = z * standard_error   # assumes it is normally distributed
        left = p_hat - margin_of_error
        right = p_hat + margin_of_error
        logger.info(f'p_hat: {p_hat:.2f}, z: {z:.2f}, margin of error: {margin_of_error:.3f}')
        logger.info(f'Confidence interval: ({left:.3f}, {right:.3f})')
        confidence_interval = (left, right)
        return confidence_interval

    def get_sample_size1(self, min_detectable_effect=0.2):
        """ Based on https://www.evanmiller.org/ab-testing/sample-size.html
        Explanation here: http://www.alfredo.motta.name/ab-testing-from-scratch/
        and here, two sided or one sided: https://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm
        Args:
            - min_detectable_effect (float): Minimum detectable effect, relative to base conversion rate.
        - TODO: add option to select wether the min_detectable_effect is absolute or relative
        """
        p_hat = self.A['p_hat']
        delta = p_hat * min_detectable_effect
        if self.two_sided:
            t_alpha = norm.ppf(1.0 - self.significance / 2)  # two-sided interval
        else:
            t_alpha = norm.ppf(1.0 - self.significance)  # one-sided interval
        t_beta = norm.ppf(self.power)
        sd1 = np.sqrt(2 * p_hat * (1.0 - p_hat))
        sd2 = np.sqrt(p_hat * (1.0 - p_hat) + (p_hat + delta) * (1.0 - p_hat - delta))
        sample_size = ((t_alpha * sd1 + t_beta * sd2) / delta) ** 2
        logger.info(f'A minimum sample size of {sample_size:.0f} is needed\
                     \n  to detect a change of {min_detectable_effect:.0%} of a base CTR of {self.A["p_hat"]:.0%}\
                     \n  with a power of {self.power:.0%}')
        return sample_size

    def get_sample_size2(self, min_detectable_effect=0.2, verbose=True):
        '''
        Sample size needed for a two-sided test, given a minimun dettectable effect
        Args:
            - min_detectable_effect (float): Minimum detectable effect, relative to base conversion rate.

        Matches the results obtained in R by using:
        # power.prop.test(p1=.1, p2=.12, power=0.8, alternative='two.sided', sig.level=0.05, strict=T)
        '''
        effect_size = sms.proportion_effectsize(self.A['p_hat'], self.A['p_hat'] * (1 + min_detectable_effect))
        sample_size = sms.NormalIndPower().solve_power(effect_size, power=self.power, alpha=self.significance, ratio=1)
        # alternative way of calculating it:
        # smp.zt_ind_solve_power(nobs1=None, effect_size=es, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
        if verbose:
            logger.info(f'A minimum sample size of {sample_size:.0f} is needed\
                         \n  to detect a change of {min_detectable_effect:.0%} of a base CTR of {self.A["p_hat"]:.0%}\
                         \n  with a power of {self.power:.0%}')
        return sample_size

    def get_sample_size3(self, min_detectable_effect):
        """Returns the minimum sample size to set up a split test
        Arguments:
            mde (float): minimum change in measurement between control
            group and test group if alternative hypothesis is true, sometimes
            referred to as minimum detectable effect
        Returns:
            min_N: minimum sample size (float)
        References:
            Stanford lecture on sample sizes
            http://statweb.stanford.edu/~susan/courses/s141/hopower.pdf
        """
        bcr = self.A['p_hat']   # base conversion rate
        # standard normal distribution to determine z-values
        standard_norm = norm(0, 1)

        # find Z_beta from desired power
        Z_beta = standard_norm.ppf(self.power)

        # find Z_alpha
        Z_alpha = standard_norm.ppf(1-self.significance/2)  # two-sided

        # average of probabilities from both groups
        pooled_prob = (bcr + bcr * (1 + min_detectable_effect)) / 2

        min_N = (2 * pooled_prob * (1 - pooled_prob) * (Z_beta + Z_alpha)**2 / min_detectable_effect**2)
        return min_N

    def compute_power(self, min_detectable_effect, sample_size=100):
        ''' Calculates the power of an AB test when control is defined as variant A, and given the inputs:
        Args:
            - min_detectable_effect (float): Minimum detectable effect, relative to base conversion rate.
            - sample_size: (int) number of data points that are available for the test '''
        effect_size = sms.proportion_effectsize(self.A['p_hat'], self.A['p_hat'] * (1 + min_detectable_effect))
        power = sms.NormalIndPower().solve_power(effect_size, nobs1=sample_size, alpha=self.significance, ratio=1)
        return power

    def plot_sample_size_vs_diff(self, min_diff=0.01, max_diff=0.03, step_diff=0.001):
        ''' Plots the sample size needed vs. minimum detectable difference in the input range given:
        Args:
            - min_diff: (float) minimum detectable difference, lower bound in the graph (x-axis)
            - max_diff: (float) maximum detectable difference, higher bound in the graph (x-axis)
            - step_diff: (float) step size for the minimum detectable difference within the previously defined range
        Note: Uses get_sample_size2 to get the sample size needed and the power defined in the objet initialization'''
        min_diffs = np.arange(min_diff, max_diff, step_diff)   # range definition
        sample_sizes = [self.get_sample_size2(min_detectable_effect=min_diff, verbose=False) for min_diff in min_diffs]
        plot = sns.lineplot(min_diffs, sample_sizes)
        plot.set(xlabel='Minimum detectable difference (% of CTR base)', ylabel='Sample size',
                 title=fr'Significance level: {self.significance:.0%} ($\alpha$), power: {self.power:.0%} (1-$\beta$)')
        plot.set_xticklabels(['{:,.0%}'.format(x) for x in plot.get_xticks()])
        plt.suptitle('Sample size required for a minimum detectable difference')  # actual title
        plt.show()

    def plot_power_vs_sample_size(self, min_sample_size=200, max_sample_size=20000, step_sample_size=100,
                                  min_diffs=[0.01, 0.02, 0.05]):
        ''' Plots the power of the AB Test vs. sample size in the input range:
        Args:
            - min_sample_size: (int) minimum sample size in plot (x-axis)
            - max_sample_size: (int) maximum sample size in plot (x-axis)
            - step_sample_size: (int) '''
        sample_sizes = np.arange(min_sample_size, max_sample_size, step_sample_size)
        for min_diff in reversed(min_diffs):
            powers = [self.compute_power(min_detectable_effect=min_diff, sample_size=sample_size)
                      for sample_size in sample_sizes]
            plt.plot(sample_sizes, powers, label=f'{min_diff:.0%}')
        plt.suptitle('Statistical power by sample size and minimum detectable difference')
        plt.title(fr'Significance level: {self.significance:.0%} ($\alpha$)')
        plt.legend(title='Min detectable difference\n           (% of base CTR)')
        plt.xlabel('Sample size')
        # Format y-labels as %
        locs, _ = plt.yticks()
        plt.yticks(locs, [f'{y:.0%}' for y in locs])
        plt.ylabel(r'Power (1 - $\beta$)')
        # add a horizontal line indicating 80% power
        plt.hlines(y=0.8, xmin=min_sample_size, xmax=max_sample_size, colors='k', linestyles='dotted', label='80%')
        plt.show()

    def _get_z_val(self):
        ''' Returns the z value for a given significance level '''
        # Find the z value accordingly with one_sided or two_sided test:
        prob = (self.significance / 2) if self.two_sided else (self.significance)   # alpha/2 if two-sided
        z = -norm.ppf(prob)   # finds the z-value for the given probability
        return z

    def _get_confidence_interval(self, sample_mean=0, sample_std=1, sample_size=1):
        ''' Returns the confidence interval of a normal distribution as input, using the significance level
        self.significance
        Args:
            - sample_mean: (float)
            - sample_std: (float)
            - sample_size: (int)'''
        z = self._get_z_val()
        left = sample_mean - z * sample_std / np.sqrt(sample_size)
        right = sample_mean + z * sample_std / np.sqrt(sample_size)
        return (left, right)

    def _plot_norm_dist(self, ax, mu=0, std=1, label=None, with_CI=False):
        ''' Plots a normal distribution to the axis provided
        Args:
            - ax: (matplotlib axes)
            - mu: (float) mean of the normal distribution
            - std: (float) standard deviation of the normal distribution
            - label: (string or None) label name for the plot
            - with_CI: (bool) if True, adds confidence interval to the plot '''
        x = np.linspace(mu - 12 * std, mu + 12 * std, 1000)
        y = norm(mu, std).pdf(x)
        ax.plot(x, y, label=label)
        if with_CI:
            self._plot_confidence_interval(ax, mu, std)

    def _plot_confidence_interval(self, ax, mu, std, color='grey'):
        ''' Calculates the confidence interval of a normal distribution and plots it in the ax object
        Args:
            - ax: (matplotlib axes)
            - mu: (float) mean of the normal distribution
            - std: (float) standard deviation of the normal distribution '''
        left, right = self._get_confidence_interval(mu, std)
        ax.axvline(left, c=color, linestyle='--', alpha=0.5)
        ax.axvline(right, c=color, linestyle='--', alpha=0.5)

    def _plot_null(self, ax, with_CI=True):
        """ Plots the null hypothesis distribution: if there is no real change (d_hat=mu=0), the distribution of the
        differences between the test and the control groups will be normally distributed.
        The confidence band is also plotted.
        Args:
            - ax (matplotlib axes) where to plot
        """
        std = self.AB_stats.pooled_se
        self._plot_norm_dist(ax, mu=0, std=std, label='Null', with_CI=with_CI)

    def _plot_alt(self, ax, with_CI=False):
        ''' Plots the alternative hypothesis distribution, where if there is a real change, the distribution of the
        differences between the test and the control groups will be normally distributed and centered around d_hat
        Args:
            - ax: (matplotlib axes) '''
        self._plot_norm_dist(ax, mu=self.AB_stats.d_hat, std=self.AB_stats.pooled_se,
                             label='Alternative', with_CI=with_CI)

    def _get_distribution(self, variant='control'):
        ''' Returns a normal distribution object depending on the selected variant
        Uses the pooled standard error as standard deviation and the mean depends on the selected variant
        Args:
            - variant: (string) Options: control or test
        Return:
            a scipy.stats.norm distribution object '''
        sample_mean = 0 if variant == 'control' else self.AB_stats.d_hat
        std = self.AB_stats.pooled_se
        return norm(sample_mean, std)

    def _fill_area(self, ax, show, show_pvalue=True):
        ''' Fill areas in an AB results plot depending on the show input:
        Args:
            - show: (str) Options: power, alpha, beta, p-value '''
        se = self.AB_stats.pooled_se
        left, right = self._get_confidence_interval(sample_mean=0, sample_std=se)   # TODO: why not get the pooled_SE from AB_stats???
        x = np.linspace(-12 * se, 12 * se, 1000)
        null_dist = self._get_distribution('control')
        alternative_dist = self._get_distribution('test')
        # Fill areas:
        if show == 'power':    # fill between upper significance boundary and distribution for alternative hypothesis
            ax.fill_between(x, 0, alternative_dist.pdf(x), color='green', alpha='0.25', where=(x > right))
            ax.text(-3 * se, null_dist.pdf(0), f'power = {1 - alternative_dist.cdf(right):.2%}',
                    fontsize=12, ha='right', color='k')
        elif show == 'alpha':   # Fill between upper significance boundary and distribution for null hypothesis
            ax.fill_between(x, 0, null_dist.pdf(x), color='green', alpha='0.25', where=(x > right))
            ax.text(-3 * se, null_dist.pdf(0), 'alpha = {0:.2%}'.format(1 - null_dist.cdf(right)),
                    fontsize=12, ha='right', color='k')
        elif show == 'beta':   # Fill between distribution for alternative hypothesis and upper significance boundary
            ax.fill_between(x, 0, alternative_dist.pdf(x), color='green', alpha='0.25', where=(x < right))
            ax.text(-3 * se, null_dist.pdf(0), 'beta = {0:.2%}'.format(alternative_dist.cdf(right)),
                    fontsize=12, ha='right', color='k')
        if show_pvalue:   # based on the binomial distributions for the two groups
            ax.text(3 * se, null_dist.pdf(0), f'p-value = {self.AB_stats.p_value:.2%}', fontsize=12, ha='left')
        plt.legend()

    def AB_plot(self, show=None, figsize=(10, 6)):
        ''' Plots the AB test analysis results
        Args:
            - show: (str) Options: power, alpha, beta, p-value, or None '''
        fig, ax = plt.subplots(figsize=figsize)   # create the plot object
        # plot the distribution of the null and alternative hypothesis
        self._plot_null(ax)
        self._plot_alt(ax)
        if show:
            self._fill_area(ax, show)
        # set extent of plot area
        # ax.set_xlim(-3 * d_hat, 3 * d_hat)
        plt.xlabel('mean difference')
        plt.ylabel('PDF')
        plt.show()


# Alternatives TODO:
# finding the power of a t-test (non-independent????)
# smp.ttest_power(effect_size=0.2, nobs=60, alpha=0.1, alternative='two-sided')
# p = pwr.t.test(d=0.2, n=60, sig.level=0.10, type="one.sample", alternative="two.sided"); p$power    # 0.45 in R

# Another way of finding the power of a t-test:
# smp.tt_solve_power(nobs=60, effect_size=0.2, alpha=0.1, alternative='two-sided')  # same as R's pwr:


# finding the power of a t-test (independent)
# smp.tt_ind_solve_power(nobs1=120.2232, effect_size=0.3, ratio=1, alpha=0.05, alternative='larger')  # same as R's pwr:  0.74




# Finding the min sample size needed
# smp.tt_ind_solve_power(nobs1=None, effect_size=0.3, power=0.75, ratio=1, alpha=0.05, alternative='larger')  # same as R's pwr:
# p = pwr.t.test(d=0.3, power=0.75, sig.level=0.05, type="two.sample", alternative="greater"); p$n   # 120.22 in R

# Other way of calculating it in Python:
# smp.TTestIndPower().solve_power(nobs1=None, effect_size=0.3, power=0.75,
#                                 ratio=1, alpha=0.05, alternative='larger')   # Same as R pwr
