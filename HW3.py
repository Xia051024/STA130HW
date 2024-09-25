#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1.
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pandas as pd

# Load penguins dataset
penguins = sns.load_dataset("penguins")

# Define function to add annotations
def add_annotations(fig, species_df):
    mean = species_df['flipper_length_mm'].mean()
    median = species_df['flipper_length_mm'].median()
    min_val = species_df['flipper_length_mm'].min()
    max_val = species_df['flipper_length_mm'].max()
    q1 = species_df['flipper_length_mm'].quantile(0.25)
    q3 = species_df['flipper_length_mm'].quantile(0.75)
    std = species_df['flipper_length_mm'].std()
    
    # Add vertical lines for mean and median
    fig.add_vline(x=mean, line_dash="dash", line_color="blue", annotation_text="Mean", annotation_position="top left")
    fig.add_vline(x=median, line_dash="dash", line_color="green", annotation_text="Median", annotation_position="top right")
    
    # Add rectangles for range, IQR, and 2-standard-deviation range
    fig.add_vrect(x0=min_val, x1=max_val, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="Range", annotation_position="top left")
    fig.add_vrect(x0=q1, x1=q3, fillcolor="purple", opacity=0.1, line_width=0, annotation_text="IQR", annotation_position="top right")
    fig.add_vrect(x0=mean - 2*std, x1=mean + 2*std, fillcolor="red", opacity=0.1, line_width=0, annotation_text="2 Std Dev", annotation_position="top right")

# Create histograms for each species
for species in penguins['species'].unique():
    species_df = penguins[penguins['species'] == species]
    
    fig = px.histogram(species_df, x='flipper_length_mm', title=f"Flipper Length Distribution for {species}")
    
    # Add lines and rectangles to indicate statistics
    add_annotations(fig, species_df)
    
    fig.show()


# In[2]:


#2.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load penguins dataset
penguins = sns.load_dataset("penguins")

# Define a function to add annotations (mean, median, etc.)
def add_annotations(ax, species_df):
    mean = species_df['flipper_length_mm'].mean()
    median = species_df['flipper_length_mm'].median()
    min_val = species_df['flipper_length_mm'].min()
    max_val = species_df['flipper_length_mm'].max()
    q1 = species_df['flipper_length_mm'].quantile(0.25)
    q3 = species_df['flipper_length_mm'].quantile(0.75)
    std = species_df['flipper_length_mm'].std()

    # Add vertical lines for mean and median
    ax.axvline(mean, color='blue', linestyle='--', label='Mean')
    ax.axvline(median, color='green', linestyle='--', label='Median')

    # Add shaded areas for range, IQR, and 2 standard deviations
    ax.axvspan(min_val, max_val, color='orange', alpha=0.1, label='Range')
    ax.axvspan(q1, q3, color='purple', alpha=0.1, label='IQR')
    ax.axvspan(mean - 2*std, mean + 2*std, color='red', alpha=0.1, label='2 Std Dev')

# Get unique species
species_unique = penguins['species'].dropna().unique()

# Create a figure with 3 subplots (one row, three columns)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Loop through each species and plot KDE
for i, species in enumerate(species_unique):
    species_df = penguins[penguins['species'] == species]
    
    # KDE plot
    sns.kdeplot(data=species_df, x='flipper_length_mm', ax=axes[i], fill=True, color='lightblue')
    
    # Add title and labels
    axes[i].set_title(f"Flipper Length for {species}")
    axes[i].set_xlabel('Flipper Length (mm)')
    
    # Add mean, median, range, etc. annotations
    add_annotations(axes[i], species_df)

    # Add legend for the first plot
    if i == 0:
        axes[i].legend()

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


# #3
# ### **Box Plots**
# - **Purpose**: Box plots are used to visualize the spread and summary statistics of a dataset, such as the median, quartiles (Q1, Q3), interquartile range (IQR), and potential outliers.
# - **How it works**: 
#   - The central line represents the median.
#   - The box spans from the first quartile (Q1) to the third quartile (Q3).
#   - The "whiskers" extend to the smallest and largest values within 1.5 times the IQR from Q1 and Q3.
#   - Points beyond the whiskers are considered outliers.
# 
# #### Pros:
# - **Summarizes key statistics** (median, quartiles, IQR, range) in one concise visual.
# - **Outlier detection**: Makes it easy to identify outliers.
# - **Compact**: Good for comparing distributions across multiple groups.
# - **No assumptions**: Doesn't assume any underlying distribution.
# 
# #### Cons:
# - **No insight into modality**: Doesn't show whether the distribution is unimodal, bimodal, etc.
# - **No detail on shape**: Doesn't give insight into the actual shape of the distribution (e.g., skewness, kurtosis).
# 
# ---
# 
# ###  **Histograms**
# - **Purpose**: Histograms display the frequency or count of data points in intervals (bins), providing a rough sense of the distribution's shape and spread.
# - **How it works**: 
#   - Data is divided into intervals (bins), and the height of each bar represents the count of observations in that bin.
# 
# #### Pros:
# - **Intuitive**: Easy to understand and interpret.
# - **Shows the distribution shape**: Can reveal whether the data is symmetric, skewed, bimodal, etc.
# - **Versatile**: Suitable for both continuous and discrete data.
# 
# #### Cons:
# - **Bin size sensitivity**: The shape of the histogram can change significantly based on the choice of bin size.
# - **Discontinuous**: Data is discretized into bins, which can lead to loss of detail.
# - **Less compact**: Takes more space, especially when comparing distributions across groups.
# 
# ---
# 
# ###  **Kernel Density Estimators (KDE)**
# - **Purpose**: KDE is a non-parametric way to estimate the probability density function of a continuous random variable, providing a smooth curve to visualize the distribution.
# - **How it works**: 
#   - KDE estimates the data's underlying distribution by placing a smooth kernel (often Gaussian) at each data point and summing the results.
# 
# #### Pros:
# - **Smooth visualization**: Provides a continuous, smooth estimate of the data distribution.
# - **Insight into modality**: Can reveal multiple peaks (unimodal, bimodal, etc.), which histograms might miss.
# - **Adjustable smoothness**: The bandwidth parameter allows control over the smoothness of the curve.
# 
# #### Cons:
# - **Bandwidth sensitivity**: The choice of bandwidth can significantly affect the smoothness of the curve. Too small a bandwidth can overfit (too many peaks), while too large a bandwidth can oversmooth (hide features).
# - **Harder to interpret**: More abstract than a histogram for beginners.
# - **Requires more data**: KDE can struggle with small datasets.
# 
# ### **When to Use Each:**
# - **Box Plot**: Use when you need a quick summary of data with clear outlier detection and comparison across categories.
# - **Histogram**: Best for a more granular view of data distribution, particularly when exploring a dataset's shape (e.g., skewness, symmetry).
# - **KDE**: Ideal for a smoother, more refined visualization of a data distribution, especially when you want to understand the underlying continuous density.
# 
# I would prefer histogram, as it can be read easily and intuitively. It shows clearly whether the graph is symmetrical, negative skewed or left skewed. The mode is clear in histogram. And the position of median and mean could tell a lot from the histogram.
# 
# Chats Summary:
# https://chatgpt.com/share/66f444f5-d2c0-8008-9c90-3d190d7219c8
# 
# 1. **Visualizing Flipper Length in Penguins Dataset**:
#    - Discussed how to create histograms using Plotly to visualize the distribution of `flipper_length_mm` for each species in the penguins dataset, including marking mean, median, range, interquartile range (IQR), and ±2 standard deviations.
#    - Provided a code snippet for generating the histograms with annotations for these statistics.
# 
# 2. **Using Seaborn for KDE Plots**:
#    - Explained how to produce Kernel Density Estimation (KDE) plots using Seaborn, displaying them in a row of three for each species.
#    - Shared code to create KDE plots with annotations for mean, median, IQR, and ±2 standard deviations.
# 
# 3. **Understanding Box Plots, Histograms, and KDEs**:
#    - Defined box plots, histograms, and kernel density estimators (KDEs).
#    - Outlined the pros and cons of each visualization method:
#      - **Box Plots**: Provide concise summaries and show outliers but lose detailed distribution information.
#      - **Histograms**: Show the shape of the distribution clearly but are sensitive to bin size.
#      - **KDEs**: Offer smooth estimates of distribution without binning but can be sensitive to bandwidth choice.

# In[3]:


# 4.
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

n = 1500
data1 = stats.uniform.rvs(0, 10, size=n)
data2 = stats.norm.rvs(5, 1.5, size=n)
data3 = np.r_[stats.norm.rvs(2, 0.25, size=int(n/2)), stats.norm.rvs(8, 0.5, size=int(n/2))]
data4 = stats.norm.rvs(6, 0.5, size=n)

fig = make_subplots(rows=1, cols=4)

fig.add_trace(go.Histogram(x=data1, name='A', nbinsx=30, marker=dict(line=dict(color='black', width=1))), row=1, col=1)
fig.add_trace(go.Histogram(x=data2, name='B', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=2)
fig.add_trace(go.Histogram(x=data3, name='C', nbinsx=45, marker=dict(line=dict(color='black', width=1))), row=1, col=3)
fig.add_trace(go.Histogram(x=data4, name='D', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=4)

fig.update_layout(height=300, width=750, title_text="Row of Histograms")
fig.update_xaxes(title_text="A", row=1, col=1)
fig.update_xaxes(title_text="B", row=1, col=2)
fig.update_xaxes(title_text="C", row=1, col=3)
fig.update_xaxes(title_text="D", row=1, col=4)
fig.update_xaxes(range=[-0.5, 10.5])

for trace in fig.data:
    trace.xbins = dict(start=0, end=10)
    
# This code was produced by just making requests to Microsoft Copilot
# https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/COP/SLS/0001_concise_makeAplotV1.md

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# 4.
# (1）Which datasets have similar means and similar variances
# A and B
# (2)Which datasets have similar means but quite different variances
# B and C
# (3)Which datasets have similar variances but quite different means
# D and B
# (4)Which datasets have quite different means and quite different variances
# C and D

# 5. 
# ### Mean and Median in Relation to Skewness
# 
# 1. **Right Skewness (Positive Skew)**
#    - **Characteristics**: In a right-skewed distribution, the tail on the right side (higher values) is longer or fatter than the left side.
#    - **Relationship**: 
#      - **Mean > Median**: Because the mean is influenced by extreme values in the right tail, it tends to be greater than the median. 
#      - **Example**: Consider income distributions in many economies; a few high earners can significantly raise the mean.
# 
# 2. **Left Skewness (Negative Skew)**
#    - **Characteristics**: In a left-skewed distribution, the tail on the left side (lower values) is longer or fatter than the right side.
#    - **Relationship**:
#      - **Mean < Median**: In this case, the mean is pulled down by the extreme low values, resulting in a mean that is less than the median.
#      - **Example**: Test scores where a few students perform poorly can lower the average score while the median remains higher.
# 
# 3. **Symmetrical Distribution**
#    - **Characteristics**: In a symmetrical distribution, the data values are evenly distributed around the mean.
#    - **Relationship**:
#      - **Mean = Median**: Both measures of central tendency will be equal, as the distribution does not favor one side over the other.
#      - **Example**: A normal distribution, which is perfectly symmetrical.
# 
# ### Why These Relationships Occur
# 
# - **Influence of Outliers**: The mean is sensitive to extreme values (outliers), while the median is resistant to them. In skewed distributions, the presence of outliers on one side will significantly impact the mean but not the median.
# - **Distribution Shape**: Skewness describes the asymmetry of the distribution. When a distribution is skewed, the tail on one side indicates the presence of extreme values that influence the mean more than the median.
# 
# ### Summary
# 
# - **Right Skewness**: Mean > Median
# - **Left Skewness**: Mean < Median
# - **Symmetrical**: Mean = Median

# In[16]:


import pandas as pd
import numpy as np
import plotly.express as px
from scipy import stats

# Generate a sample from a gamma distribution
sample1 = stats.gamma(a=2, scale=2).rvs(size=1000)

# Create a histogram of sample1
fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data", title='Histogram of Sample 1')
fig1.show(renderer="png")  # For GitHub and MarkUs submissions

# Calculate the mean and median of sample1
mean_sample1 = sample1.mean()
median_sample1 = np.quantile(sample1, [0.5])  # median

# Generate a second sample from a gamma distribution and negate it
sample2 = -stats.gamma(a=2, scale=2).rvs(size=1000)

# Optionally, you can create a histogram for sample2 if desired
fig2 = px.histogram(pd.DataFrame({'data': sample2}), x="data", title='Histogram of Sample 2')
fig2.show(renderer="png")  # For GitHub and MarkUs submissions

# Output mean and median
print(f"Mean of Sample 1: {mean_sample1}")
print(f"Median of Sample 1: {median_sample1}")


# 
# 
# ### 1. **Import Libraries**
# ```python
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from scipy import stats
# ```
# - **Pandas**: Used for data manipulation and analysis, especially for handling data in DataFrame format.
# - **NumPy**: Provides support for arrays and numerical operations, particularly useful for statistical calculations.
# - **Plotly Express**: A high-level interface for creating interactive visualizations.
# - **SciPy**: Contains statistical functions, including those for generating random variables from various distributions.
# 
# ### 2. **Generate the First Sample**
# ```python
# sample1 = stats.gamma(a=2, scale=2).rvs(size=1000)
# ```
# - **`stats.gamma(a=2, scale=2)`**: Creates a gamma distribution with shape parameter `a=2` and scale parameter `scale=2`.
# - **`.rvs(size=1000)`**: Generates 1000 random variates (samples) from this gamma distribution and stores them in `sample1`.
# 
# ### 3. **Create a Histogram of the First Sample**
# ```python
# fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data", title='Histogram of Sample 1')
# fig1.show(renderer="png")
# ```
# - **`pd.DataFrame({'data': sample1})`**: Converts the `sample1` array into a pandas DataFrame with a column named "data".
# - **`px.histogram(...)`**: Uses Plotly Express to create a histogram of the values in the "data" column.
# - **`fig1.show(renderer="png")`**: Displays the histogram. The `renderer="png"` option ensures it can be used in environments like GitHub or MarkUs, which may not support interactive plots.
# 
# ### 4. **Calculate Mean and Median of the First Sample**
# ```python
# mean_sample1 = sample1.mean()
# median_sample1 = np.quantile(sample1, [0.5])  # median
# ```
# - **`sample1.mean()`**: Calculates the mean (average) of the values in `sample1`.
# - **`np.quantile(sample1, [0.5])`**: Calculates the median, which is the 50th percentile of the distribution.
# 
# ### 5. **Generate the Second Sample**
# ```python
# sample2 = -stats.gamma(a=2, scale=2).rvs(size=1000)
# ```
# - This line generates a new sample from the same gamma distribution and negates it. 
# - **Important Note**: Negating a gamma distribution creates values that are always negative, which may not have a meaningful interpretation since the gamma distribution is typically used for modeling positive quantities.
# 
# ### 6. **Optional: Create a Histogram of the Second Sample**
# ```python
# fig2 = px.histogram(pd.DataFrame({'data': sample2}), x="data", title='Histogram of Sample 2')
# fig2.show(renderer="png")
# ```
# - Similar to the first histogram, this creates a histogram for `sample2` and displays it.
# 
# ### 7. **Output Mean and Median**
# ```python
# print(f"Mean of Sample 1: {mean_sample1}")
# print(f"Median of Sample 1: {median_sample1}")
# ```
# - Prints the calculated mean and median values of `sample1` to the console.
# 
# 
# 
# From my perspectives, mean would be equal to median only if the histogram is absolutely symmetric. Mean is average number, and median is the number of 50% of the data set. If the graph is "right" or "left" skewed, mean and median will not be qual. As the median would be positioned around the peak of the graph. For mean, because of the skewed part, the mean will be lefteard (if it is left skewed) or rightward (if it is right skewed) , resulting from the small number of relatively large or small data.
# From the sample 1 and 2, I could say the median of these two datas would be around the peak of the histogram. The median will not be equal to mean. For sample 1, median is smaller than mean. For sample 2, median is larger than mean.
# 
# 
# 
# Chat summary:
# https://chatgpt.com/share/66f44d27-d43c-8008-b47d-49f66e6935a3
# 
# 1. **Exploration of Mean, Median, and Skewness**:
#    - Discussed the relationship between the mean, median, and skewness of a distribution.
#    - Explained how in right-skewed distributions, the mean is greater than the median, while in left-skewed distributions, the mean is less than the median.
#    - Illustrated the concepts with visual representations and noted that outliers influence the mean more than the median.
# 
# 2. **Code Analysis**:
#    - Analyzed a code snippet that generates random samples from a gamma distribution.
#    - Described how the code imports necessary libraries, generates a right-skewed sample, creates a histogram, calculates the mean and median, and then generates a left-skewed sample by negating the values of the gamma distribution.
#    
#  https://chatgpt.com/share/66f44fa0-5304-8008-a44e-6195ff423703
# 
# 1. **Initial Code Fix**: You provided code that generated a sample from a gamma distribution, created a histogram, and calculated summary statistics (mean and median). You also attempted to generate a second sample by negating the gamma-distributed values. I helped you fix and improve the code.
# 
# 2. **Code Explanation**: I explained how the corrected code works step by step, covering:
#    - Importing necessary libraries.
#    - Generating a random sample from a gamma distribution.
#    - Creating a histogram to visualize the first sample.
#    - Calculating the mean and median of the first sample.
#    - Generating a second sample by negating the first sample.
#    - Optionally creating a histogram for the second sample.
#    - Printing the mean and median of the first sample.

# In[104]:


#6.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/manuelamc14/fast-food-Nutritional-Database/main/Tables/nutrition.csv"
df = pd.read_csv(url)


summary = df.describe()
print("Statistical Summary:")
print(summary)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/manuelamc14/fast-food-Nutritional-Database/main/Tables/nutrition.csv"
df = pd.read_csv(url)

plt.figure(figsize=(12, 6))


sns.boxplot(data=df.select_dtypes(include='number'))
plt.title('Box Plot of Nutritional Values')
plt.xticks(rotation=45)
plt.ylabel('Nutritional Value')
plt.xlabel('Nutritional Metrics')

plt.tight_layout()
plt.show()


# In[95]:


#7.
import plotly.express as px
import pandas as pd

gapminder = px.data.gapminder()

fig = px.scatter(gapminder, 
                 x='gdpPercap', 
                 y='lifeExp', 
                 size='pop', 
                 color='continent', 
                 animation_frame='year', 
                 range_x=[0, 10000], 
                 range_y=[20, 100],
                 title='Gapminder Animation: GDP vs Life Expectancy',
                 hover_name='country')
fig.show()


# In[105]:


#8.
import plotly.express as px
import pandas as pd

df = px.data.gapminder()
df['percent_change'] = (df['gdpPercap'].pct_change()).fillna(0)
df['rank'] = df.groupby('year')['gdpPercap'].rank(method='first', ascending=False)  
df['percent'] = df['pop'] / df['pop'].max() 

fig = px.scatter(df,
                 x='percent_change',  
                 y='rank',            
                 animation_frame='year',  
                 animation_group='country',  
                 size='percent',      
                 color='continent',   
                 hover_name='country', 
                 size_max=50,       
                 range_x=[-0.005, 0.005],  
                 title='Gapminder: Percent Change vs Rank by Continent Over Time') 
fig.show()


# 9. no
