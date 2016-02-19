"""
This code is part of Data Science Dojo's bootcamp
Copyright (C) 2016

Objective: Understand AB Testing by two simple examples
Data source: Dwell_Time_VersionA.csv and Dwell_Time_VersionB.csv at:
    https://github.com/datasciencedojo/bootcamp/tree/master/Online_Experimentation_and_AB_Testing
Python Version: 3.4+
Packages: pandas, statsmodels, matplotlib
"""

from statsmodels.stats.proportion import proportions_chisquare, proportion_effectsize
from statsmodels.stats.power import NormalIndPower, tt_ind_solve_power
from statsmodels.stats.weightstats import ttest_ind
import pandas as pd
import matplotlib.pyplot as plt

###################################################################################
## 1. AB Test of Proportions
## 
## Basic concepts:
##
## Conversion rate -- In electronic commerce, The conversion rate is the proportion
## of visitors to a website who take action to beyond a casual content view or
## website visit, as a result of subtle or direct requests from marketers, 
## advertisers, and content creators. The formula is:
## Conversion rate = Number of Goal Achievements / Number of Visitors
## Successful conversions are defined differently by individual marketers,
## advertisers, and content creators.
## Check out the wikipedia page http://en.wikipedia.org/wiki/Conversion_marketing
## for more details of conversion rate and conversion marketing.
##
## Significance level -- the probability of the null hypothesis being true.
## 
## Confidence level = 1 - Significance level -- the probability of the null hypothesis being false
##
## Type 1 error -- Incorrectly rejecting the null hypothesis. Rate is connected to significance level
## Type II error -- Incorrectly accepting the null hypothesis. This is a subtler and harder to
##                  measure error.
##
## Power =  1 - type II error rate -- The probability that we do not incorrectly accept the null hypothesis
## 
## Scenario:
## Your team is developing and maintaining a e-comerce website. Last week, You did
## an AB test to compare two version of the main landing page (versions A & B).
## This week, you are going to analyze the results of last week's AB test. Your
## team has agreed on conversion rate as the metric.
## Data:
## 1. There are 298,234 visiting times of the website last week. The server showed A
## and B versions with equal times among all the visitings.
## 2. There are 8365 successful conversions in version A, and 8604 successful
## conversions in version B.
## Question:
## Do A and B versions of the website give the website the same
## conversion rate? What is the significance level?
###################################################################################

## Assign some variables
visits_per_group = 298234/2
success_A = 8365
success_B = 8604
conversion_rate_A = success_A / visits_per_group
conversion_rate_B = success_B / visits_per_group

## Perform a proportional hypothesis test
## H0 (null hypothesis): conversion rates of version A and B are the same
## H1 (alternative hypothesis): conversion rates of version A and B are different
(chi2, chi2_p_value, expected) = proportions_chisquare(count=[success_A, success_B],
                                                  nobs=[visits_per_group, visits_per_group])
print("chi2: %f \t p_value: %f" % (chi2, chi2_p_value))

## Examine the output of the chi2_contingency function.
## If the target p_value is 0.05, what is your conclusion? Do you accept or reject H0?

## Note that this test only tells you whether A & B have different conversion rates, not
## which is larger. In this case, since A & B had the same number of visits, this is easy to 
## determine. However, if you only showed B to 10% of your visitors, you may want to use a
## one-sided test instead.

## Your team also wants to know the "power" of the above results. Since they want to
## know if H1 is true, what is the possiblity that we accept H0 when H1 is true?
## The power can be obtained using the GofChisquarePower.solve_power function
effect_size = proportion_effectsize(prop1=conversion_rate_A, prop2=conversion_rate_B)
proportion_test_power = NormalIndPower().solve_power(effect_size=effect_size, nobs1=visits_per_group, alpha=0.05)

###################################################################################
## 2. AB Test of Means
## Scenario:
## Your team's manager asks you about dwell time differences between versions A and B
## Question: Is the customers' time spent on page different between version A
## and B of the website?
###################################################################################
## Load the data
## Remember to set your working directory to the bootcamp base folder
dwell_time_A = pd.read_csv('Datasets/Dwell_Time/Dwell_Time_VersionA.csv')
dwell_time_B = pd.read_csv('Datasets/Dwell_Time/Dwell_Time_VersionB.csv')

## Visualize the data
## Calculate mean and standard deviation (sd) of dwell time on the web pages
mean_A = round(dwell_time_A['dwellTime'].mean(), 2)
sd_A = round(dwell_time_A['dwellTime'].std(), 2)
mean_B = round(dwell_time_B['dwellTime'].mean(), 2)
sd_B = round(dwell_time_B['dwellTime'].std(), 2)
mean_sd_AB = pd.DataFrame({'mean': [mean_A, mean_B], 'sd': [sd_A, sd_B]}, index=['A', 'B'])
print(mean_sd_AB)

## Plot the densities of dwell times A and B
dwell_times = pd.DataFrame({'A': dwell_time_A['dwellTime'], 'B': dwell_time_B['dwellTime']})
dwell_times.plot(kind='kde', color=['r', 'b'])
plt.show()

## For this question, we use a t-test.
(tstat, t_p_value, t_df) = ttest_ind(x1=dwell_time_A, x2=dwell_time_B, alternative='two-sided')
print("tstat: %f \t p_value: %e" % (tstat[1], t_p_value[1]))
## Is the dwell time different between version A and B (with significance level 0.05)?
## What is the power of this conclusion? Use the tt_ind_solve_power function to find out.

###################################################################################
## EXERCISE:
## After you finish the above analysis, an engineer in your team notifies you that
## 3430 of the records in group B with unsuccessful conversion are fake data
## automatically filled in by computer. The number of visitors for version B is thus
## 298234/2-3430, but there were still 8604 successful conversions for version B.
## Does this revelation change your conclusion from section 1?
###################################################################################
