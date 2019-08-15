#                    _    ____    _____         _
#                   / \  | __ )  |_   _|__  ___| |_ ___ _ __
#                  / _ \ |  _ \    | |/ _ \/ __| __/ _ \ '__|
#                 / ___ \| |_) |   | |  __/\__ \ ||  __/ |
#                /_/   \_\____/    |_|\___||___/\__\___|_|
# References
# Google-Udacity AB Testing course
# https://stackoverflow.com/questions/15204070/is-there-a-python-scipy-function-to-determine-parameters-needed-to-obtain-a-ta/18379559#18379559
import numpy as np
from scipy.stats import norm
import logging
import statsmodels.stats.api as sms
# import statsmodels.stats.power as smp   # not used
import matplotlib.pyplot as plt
import seaborn as sns


# logger.setLevel(logging.DEBUG)
LOG_FORMAT = '%(message)s'
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger()


class ABTester:
    ''' Class AB Tester, used to perform AB Tests and plot the results! '''

    def __init__(self, A, B=None, significance=0.05, power=0.8, two_sided=True):
        ''' Initializes the object
        Args:
            - A, B: (dict) dictionaries with two key-values: conversions and impressions (int)
            - significance: (float) alpha
            - power: (float) 1-beta
            - two_sided: (bool) wheter a two sided test is used or not (single-sided)
        '''
        self.A = A
        self.B = B
        self.significance = significance
        self.power = power
        self.two_sided = two_sided
        self.A['p_hat'] = self.A['conversions'] / self.A['impressions']
        if B is not None:
            self.B['p_hat'] = self.B['conversions'] / self.B['impressions']

    def get_confidence_interval(self, variant='A'):
        ''' Calculates the confidence interval, uses the normal approximation for the
        binomial distribution
        Args:
            - variant: (string) 'A' or 'B'
        Returns:
            - confidence_interval: (tuple of 2 floats) with min and max interval values
        '''
        conversions = self.A['conversions'] if variant == 'A' else self.B['conversions']
        impressions = self.A['impressions'] if variant == 'A' else self.B['impressions']
        p_hat = conversions / impressions
        if impressions * p_hat < 5:
            logging.warning('Warning: the normal approximation does not hold')
        standard_error = np.sqrt(p_hat * (1-p_hat) / impressions)
        # Find the z value accordingly with one_sided or two_sided test:
        prob = self.significance / 2 if self.two_sided else self.significance   # alpha/2 if two-sided
        z = -norm.ppf(prob)
        margin_of_error = z * standard_error
        confidence_interval = (p_hat - margin_of_error, p_hat + margin_of_error)
        logger.info(f'p_hat: {p_hat:.2f}, z: {z:.2f}, margin of error: {margin_of_error:.3f}')
        logger.info(f'Confidence interval: ({confidence_interval[0]:.3f}, {confidence_interval[1]:.3f})')
        return p_hat, confidence_interval

    def get_sample_size1(self, min_detectable_effect=0.2):
        """ Based on https://www.evanmiller.org/ab-testing/sample-size.html
        Explanation here: http://www.alfredo.motta.name/ab-testing-from-scratch/
        Better here, two sided or one sided: https://www.itl.nist.gov/div898/handbook/prc/section2/prc242.htm
        Args:
            alpha (float): How often are you willing to accept a Type I error (false positive)?
            p (float): Base conversion rate
            min_detectable_effect (float): Minimum detectable effect, relative to base conversion rate.
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
                     \n  to detect a change of {min_detectable_effect:.0%} of a base CR of {self.A["p_hat"]:.0%}\
                     \n  with a power of {self.power:.0%}')
        return sample_size

    def get_sample_size2(self, min_detectable_effect=0.2, verbose=True):
        '''
        Matches the results obtained in R:
        # power.prop.test(p1=.1, p2=.12, power=0.8, alternative='two.sided', sig.level=0.05, strict=T)
        '''
        effect_size = sms.proportion_effectsize(self.A['p_hat'], self.A['p_hat'] * (1 + min_detectable_effect))
        sample_size = sms.NormalIndPower().solve_power(effect_size, power=self.power, alpha=self.significance, ratio=1)
        # alternative way of calculating it:
        # smp.zt_ind_solve_power(nobs1=None, effect_size=es, alpha=0.05, power=0.8, ratio=1, alternative='two-sided')
        if verbose:
            logger.info(f'A minimum sample size of {sample_size:.0f} is needed\
                         \n  to detect a change of {min_detectable_effect:.0%} of a base CR of {self.A["p_hat"]:.0%}\
                         \n  with a power of {self.power:.0%}')
        return sample_size

    def compute_power(self, min_detectable_effect, sample_size=100):
        effect_size = sms.proportion_effectsize(self.A['p_hat'], self.A['p_hat'] * (1 + min_detectable_effect))
        power = sms.NormalIndPower().solve_power(effect_size, nobs1=sample_size, alpha=self.significance, ratio=1)
        return power

    def plot_sample_size_vs_diff(self, min_diff=0.01, max_diff=0.03, step_diff=0.001):
        min_diffs = np.arange(min_diff, max_diff, step_diff)
        sample_sizes = [self.get_sample_size2(min_detectable_effect=min_diff, verbose=False) for min_diff in min_diffs]
        print(min_diffs, sample_sizes)
        plot = sns.lineplot(min_diffs, sample_sizes)
        plot.set(xlabel='Minimum detectable difference', ylabel='Sample size',
                 title=fr'Significance level: {self.significance:.0%} ($\alpha$), power: {self.power:.0%} (1-$\beta$)')
        plot.set_xticklabels(['{:,.1%}'.format(x) for x in plot.get_xticks()])
        plt.suptitle('Sample size required for a minimum detectable difference')
        plt.show()

    def plot_power_vs_sample_size(self, min_sample_size=200, max_sample_size=20000, step_sample_size=100,
                                  min_diffs=[0.01, 0.02, 0.05]):
        sample_sizes = np.arange(min_sample_size, max_sample_size, step_sample_size)
        for min_diff in reversed(min_diffs):
            powers = [self.compute_power(min_detectable_effect=min_diff, sample_size=sample_size)
                      for sample_size in sample_sizes]
            plt.plot(sample_sizes, powers, label=f'{min_diff:.1%}')
            # plt.yticklabels(['{:,.1%}'.format(y) for y in plt.get_yticks()])
        plt.suptitle('Statistical power by sample size and minimum detectable difference')
        plt.title(fr'Significance level: {self.significance:.0%} ($\alpha$)')
        plt.legend(title='Min detectable difference')
        plt.xlabel('Sample size')
        locs, _ = plt.yticks()
        plt.yticks(locs, [f'{y:.0%}' for y in locs])
        plt.ylabel(r'Power (1 - $\beta$)')
        plt.hlines(y=0.8, xmin=min_sample_size, xmax=max_sample_size, colors='k', linestyles='dotted', label='80%')
        plt.show()
