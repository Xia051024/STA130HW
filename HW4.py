#!/usr/bin/env python
# coding: utf-8

# # 1.
# 
# Bootstrapping is used to estimate the sampling distribution of a statistic by resampling data with replacement. By repeatedly sampling from the original dataset and calculating the desired statistic for each resample,  bootstraping could give an empirical distribution of that statistic.
# 
# Standard Error of the Mean (SEM) measures the standard error of the mean measures the precision with which the sample mean estimates the population mean. It quantifies how much the sample mean would vary from sample to sample if we repeatedly drew samples from the population.
# 
# Standard deviation of the original data measures the spread or dispersion of individual data points in the original dataset. It tells how much the data points tend to deviate from the mean of the dataset.
# 
# The difference between SEM and Standard deviation is that SEM get statistics through resampling data with replacement rather than using original data. SEM will applicate bootstrapping, estimating from the variation in the sample means across the bootstrap samples. However, for standard deviation of the original data, the statistics are not estimated from sample but from original true condition.
# 
# CHAT summary:
# https://chatgpt.com/share/66fd7c32-c7a4-8008-8596-8a21fe37a736
# 1. We explored **bootstrapping**, a statistical method that involves resampling with replacement to estimate the sampling distribution of a statistic, often used for estimating confidence intervals or the standard error when theoretical distributions are unknown.
# 
# 2. We clarified the difference between the **standard deviation (SD)** and the **standard error of the mean (SEM)**:
#    - **SD** measures the spread of data points within the sample.
#    - **SEM** estimates the precision of the sample mean as a reflection of the population mean and decreases with larger sample sizes.
# 
# 3. We discussed how **SEM can be estimated using bootstrapping**, providing a non-parametric approach for calculating SEM when traditional assumptions (e.g., normality) don’t hold.
# 
# 4. We framed the distinction between SD and SEM in terms of **sample vs. population**:
#    - SD is a **descriptive statistic** based on the sample.
#    - SEM is an **inferential statistic** aimed at understanding the accuracy of the sample mean in estimating the true population mean.

# # 2.
# 
# First,start with original sample data of size n.
#    
# Second, use bootstrap resampling. With replacement from the original sample 10000 time. Each resample should be the same size as the original sample n.
# 
# Third, calculate Standard Error of the Mean (SEM). Calculate the sample mean of each boost sample. 
# 
# Fourth, determine the Critical Value for 95% confidence. For a 95% confidence interval,use the critical value from the standard normal distribution, which is about 2 for 95% confidence.
# 
# Fifth, calculate the confidence interval. 
# CI = [The mean of the bootstrapped sample means - 2 x SEM, The mean of the bootstrapped sample means + 2 x SEM]
# SEM is the standard error of the bootstrapped means.
# This formula will give a confidence interval that covers 95% of the bootstrapped sample means.
# 
# Finally, interpret the Results. The resulting interval reflects where 95% of the bootstrapped sample means are expected to lie, providing an empirical confidence interval for the population mean based on the original sample.
# 
# CHAT summary:
# https://chatgpt.com/share/66fd7eca-8d2c-8008-bb96-3448fafd2f9c
# 
# 1. **Collect data** from your original sample.
# 2. **Bootstrap resampling**: Resample the data multiple times, calculate the mean for each resample, and create a distribution of sample means.
# 3. **Calculate SEM**: Use the standard deviation of the bootstrapped sample means as the SEM.
# 4. **Determine the critical value** (1.96 for 95% confidence) from the standard normal distribution.
# 5. **Calculate the confidence interval**
# 6. **Interpret** the results: The interval covers 95% of the bootstrapped sample means, estimating the population mean.
# 

# # 3.
# 
# Firstly, load my sample data using pd.read_csv().
# 
# Second, set the key parameters of simulation. Set our sample size and how many times we want to repeat the sampling. For exapmle, set rps = 10000(repeat times). Set our sample size equals to our original sample n.
# 
# Thirdly, use np.zeros() to create an array to save values.
# 
# Fourthly, use np.random.seed() to set original random value to make sure the results are the same while we are resampling our data. 
# 
# Fifthly, use np.random choice() sampling from our original sample with replacement.
# 
# Sixthly, use sample.mean() to calculate the semple mean of each bootstrap.
# 
# Finally, use np.quantile() to generate the lower confidence limit and the upper confidence limit from 1000 bootstrap sample mean, between the percentage from 0.025 to 0.0975.
# 

# In[4]:


# 4.
import numpy as np
import pandas as pd

# Step 1: Load the sample data
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
data = pd.read_csv(url)

# Select the feature for which we want to calculate the bootstrap CI
feature = 'sepal_length'
original_sample = data[feature].values  # Convert the feature column to a NumPy array

# Step 2: Set the key parameters for simulation
n = len(original_sample)  # Sample size equals to the original sample size
rps = 10000  # Number of times to repeat the sampling

# Step 3: Use np.zeros() to create an array to save values
bootstrap_medians = np.zeros(rps)

# Step 4: Set random seed for reproducibility
np.random.seed(333)

# Step 5: Perform bootstrap sampling
for i in range(rps):
    bootstrap_sample = np.random.choice(original_sample, size=n, replace=True)
    
    # Step 6: Calculate the sample median of each bootstrap sample
    bootstrap_medians[i] = np.median(bootstrap_sample)

# Step 7: Calculate the confidence intervals using np.quantile()
lower_ci = np.quantile(bootstrap_medians, 0.025)  # Lower confidence limit
upper_ci = np.quantile(bootstrap_medians, 0.975)  # Upper confidence limit

# Output the results
print("95% Bootstrap CI for the population median of sepal length:", (lower_ci, upper_ci))


# #Chat summary: https://chatgpt.com/share/66fd8946-0c38-8008-aa78-8082953f9452
# 
# 1. **Bootstrap Confidence Interval Code**:
#    - You requested code to produce a 95% bootstrap confidence interval for a population mean and later specified to focus on the population median.
#    - I provided a detailed code example that used the bootstrap method to calculate confidence intervals for both the mean and median, using the `sepal_length` from the Iris dataset.
# 
# 2. **Step-by-Step Instructions**:
#    - You specified a series of steps for implementing the bootstrap confidence interval, which included loading data, setting parameters, sampling, and calculating the median.
#    - I adapted the code to follow your specific instructions, including setting the random seed to ensure reproducibility.
# 
# 3. **Final Code Implementation**:
#    - The final code was provided using the `sepal_length` feature from the Iris dataset, with the random seed set to `333`.
#    - The code effectively demonstrated how to compute the 95% bootstrap confidence interval for the population median.

# # 5.
# 
# Population Parameter is a value that describes a characteristic of the entire population. It is usually unknown and fixed. Sample Statistic is a value calculated from a sample drawn from the population. It varies from sample to sample.
# Confidence intervals are used to estimate the population parameter based on the sample statistic. The sample statistic serves as a point estimate of the population parameter, but because of sampling variability, it may not equal the true population parameter. 
# The confidence interval quantifies the uncertainty around the sample statistic as an estimate of the population parameter. It provides a range within which we expect the population parameter to lie, with a certain level of confidence. We could use confidence interval to estimate the statistics we want from the entire population as population parameter cannot be calculated in real life. 
# 
# chat summary:
# https://chatgpt.com/share/66fd8dd1-5608-8008-bd94-5b3c1290373c

# # 6.
# What is the process of bootstrapping?
# The process of bootstapping is to get the statistics through resampling data with replacement.
# 
# What is the main purpose of bootstrapping?
# To estimate those population parameter by using boostrapped samples.
# 
# If you had a (hypothesized) guess about what the average of a population was, and you had a sample of size n from that population, how could you use bootstrapping to assess whether or not your (hypothesized) guess might be plausible?
# I need to check whether the (hypothesized) guess is in the CI. If it is in the range of CI, then this suggests that  hypothesized average is plausible given the sample data, meaning there isn't enough evidence to reject it. If it is not in (outside) the range of CI, then I would reject the guess since it is not supported by the sample data.

# # 7.
# If the range of CI includes 0 which indicates that the estimated range of plausible values for the population effect includes no effect (zero), then the fact "mean statistic itself is zero" should be included, which is true, failng to reject null hypothesis.
# It won't lead to the opposite conclusion in this context, because there is no condition to reject null hypothesis. Under 95% of the condition, the CI contains zero, which means under 95% null hypothesis "drug has no effect" is included. This means null hypothesis are mostly correct. I cannot reject null hypothesis, failing to reject null hypothesis. Fail to reject null hypothesis conclude rejecting alternative hypothesis. I don't have enough evidence to state alternative hypothesis is correct, meaning drugs have no effect.

# In[19]:


# 8.
import pandas as pd
import numpy as np

patient_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ages = [45, 34, 29, 52, 37, 41, 33, 48, 26, 39]
genders = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F']
initial_health_scores = [84, 78, 83, 81, 81, 80, 79, 85, 76, 83]
final_health_scores = [86, 86, 80, 86, 84, 86, 86, 82, 83, 84]

data = {
    "PatientID": patient_ids,
    "Age": ages,
    "Gender": genders,
    "InitialHealthScore": initial_health_scores,
    "FinalHealthScore": final_health_scores
}
df = pd.DataFrame(data)

df['HealthChange'] = df['FinalHealthScore'] - df['InitialHealthScore']

def bootstrap_means(data, num_samples=1000):
    boot_means = []
    for _ in range(num_samples):
        sample = np.random.choice(df['HealthChange'], size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    return np.array(boot_means)

np.random.seed(199)

bootstrap_results = bootstrap_means(df['HealthChange'].values)

mean_change = np.mean(df['HealthChange'])
lower_bound = np.percentile(bootstrap_results, 2.5)
upper_bound = np.percentile(bootstrap_results, 97.5)

print(f'Mean Change: {mean_change:.2f}')
print(f'95% Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}]')


# Set U be the population mean of the difference between final health score and initial health score.
# 
# Null hypothesis: U = 0 (vaccine has no effect)
# Alternative hypothesis : U != 0 (vaccine has some effect)
# 
# As the CI is between 0.70 and 5.60, U = 0 is not in the interval, meaning rejecting the null hypothesis. U!=0 is included in the CI, so we have enough evidence to be fail to reject the alternative hypothesis, which means the vaccine has some effect. 

# # 9. no
