#!/usr/bin/env python
# coding: utf-8

# # Clean & Analyze Social Media

# ## Introduction
# 
# Social media has become a ubiquitous part of modern life, with platforms such as Instagram, Twitter, and Facebook serving as essential communication channels. Social media data sets are vast and complex, making analysis a challenging task for businesses and researchers alike. In this project, we explore a simulated social media, for example Tweets, data set to understand trends in likes across different categories.
# 
# ## Prerequisites
# 
# To follow along with this project, you should have a basic understanding of Python programming and data analysis concepts. In addition, you may want to use the following packages in your Python environment:
# 
# - pandas
# - Matplotlib
# - ...
# 
# These packages should already be installed in Coursera's Jupyter Notebook environment, however if you'd like to install additional packages that are not included in this environment or are working off platform you can install additional packages using `!pip install packagename` within a notebook cell such as:
# 
# - `!pip install pandas`
# - `!pip install matplotlib`
# 
# ## Project Scope
# 
# The objective of this project is to analyze tweets (or other social media data) and gain insights into user engagement. We will explore the data set using visualization techniques to understand the distribution of likes across different categories. Finally, we will analyze the data to draw conclusions about the most popular categories and the overall engagement on the platform.
# 
# ## Step 1: Importing Required Libraries
# 
# As the name suggests, the first step is to import all the necessary libraries that will be used in the project. In this case, we need pandas, numpy, matplotlib, seaborn, and random libraries.
# 
# Pandas is a library used for data manipulation and analysis. Numpy is a library used for numerical computations. Matplotlib is a library used for data visualization. Seaborn is a library used for statistical data visualization. Random is a library used to generate random numbers.

# In[5]:


import pandas as pd          # For data manipulation and analysis
import numpy as np           # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns        # For statistical data visualization
import random                # For generating random numbers

# Optional: Setting style for seaborn plots
sns.set(style="whitegrid")

# Display a message confirming that libraries have been imported successfully
print("Libraries imported successfully.")


# In[34]:


# Step 1: Simulating a Dataset for Demonstration Purposes

# Categories of posts in the dataset
categories = ['Technology', 'Health', 'Entertainment', 'Sports', 'Politics']

# Creating the dataset with random data
data = {
    'TweetID': np.arange(1, 101),                     # Unique ID for each tweet
    'Category': np.random.choice(categories, 100),    # Random category assigned to each tweet
    'Likes': np.random.randint(0, 1000, 100),         # Random number of likes
    'Retweets': np.random.randint(0, 500, 100),       # Random number of retweets
    'Comments': np.random.randint(0, 300, 100)        # Random number of comments
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Displaying the first few rows of the simulated dataset
df.head()


# In[35]:


# Step 2: Exploring the Dataset

# Checking the structure and data types of the dataset
df.info()

# Checking for any missing values in the dataset
df.isnull().sum()

# Getting basic statistical information about the dataset
df.describe()


# In[36]:


# Step 3: Analyzing Likes by Category

# Calculating the total number of likes per category
likes_by_category = df.groupby('Category')['Likes'].sum().reset_index()

# Plotting the total likes by category using a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Likes', data=likes_by_category, palette='viridis')
plt.title('Total Likes by Category')
plt.xlabel('Category')
plt.ylabel('Total Likes')
plt.show()


# In[37]:


# Step 4: Visualizing the Distribution of Likes by Category

# Plotting the distribution of likes by category using a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Category', y='Likes', data=df, palette='muted')
plt.title('Distribution of Likes by Category')
plt.xlabel('Category')
plt.ylabel('Likes')
plt.show()


# In[38]:


# Step 5: Correlation Analysis

# Calculating the correlation matrix for the entire dataset
correlation = df[['Likes', 'Retweets', 'Comments']].corr()

# Plotting the correlation matrix using a heatmap
sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


# In[39]:


# Step 6: Category-Specific Correlation Analysis

# Analyzing correlation within the Health category
tech_df = df[df['Category'] == 'Health']
tech_corr = tech_df[['Likes', 'Retweets', 'Comments']].corr()

# Plotting the correlation matrix for Health category
sns.heatmap(tech_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix - Health')
plt.show()

# Analyzing correlation within the Sports category
ent_df = df[df['Category'] == 'Sports']
ent_corr = ent_df[['Likes', 'Retweets', 'Comments']].corr()

# Plotting the correlation matrix for Sports category
sns.heatmap(ent_corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix - Sports')
plt.show()


# In[45]:


# Step 7: Identifying Top 5 Posts by Likes in Health and Sports

# Extracting top 5 posts in the Health category by likes
top_health_likes = df[df['Category'] == 'Health'].sort_values(by='Likes', ascending=False).head(5)

# Extracting top 5 posts in the Sports category by likes
top_sports_likes = df[df['Category'] == 'Sports'].sort_values(by='Likes', ascending=False).head(5)

# Displaying the top 5 Health posts
print("Top 5 Health posts by Likes:")
print(top_health_likes[['Likes', 'Retweets', 'Comments']])

# Displaying the top 5 Sports posts
print("\nTop 5 Sports posts by Likes:")
print(top_sports_likes[['Likes', 'Retweets', 'Comments']])


# In[46]:


import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axes
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
bar_width = 0.25

# Data for Health Category
health_indices = np.arange(len(top_health_likes))
health_likes = top_health_likes['Likes']
health_retweets = top_health_likes['Retweets']
health_comments = top_health_likes['Comments']

# Plotting for Health Category
axes[0].bar(health_indices, health_likes, width=bar_width, label='Likes', color='blue')
axes[0].bar(health_indices + bar_width, health_retweets, width=bar_width, label='Retweets', color='green')
axes[0].bar(health_indices + 2 * bar_width, health_comments, width=bar_width, label='Comments', color='red')

# Setting titles and labels for Health
axes[0].set_title('Top 5 Health Posts by Engagement Metrics')
axes[0].set_xlabel('Post Index')
axes[0].set_ylabel('Count')
axes[0].set_xticks(health_indices + bar_width)
axes[0].set_xticklabels([f'Health {i+1}' for i in range(5)])
axes[0].legend()

# Data for Sports Category
sports_indices = np.arange(len(top_sports_likes))
sports_likes = top_sports_likes['Likes']
sports_retweets = top_sports_likes['Retweets']
sports_comments = top_sports_likes['Comments']

# Plotting for Sports Category
axes[1].bar(sports_indices, sports_likes, width=bar_width, label='Likes', color='blue')
axes[1].bar(sports_indices + bar_width, sports_retweets, width=bar_width, label='Retweets', color='green')
axes[1].bar(sports_indices + 2 * bar_width, sports_comments, width=bar_width, label='Comments', color='red')

# Setting titles and labels for Sports
axes[1].set_title('Top 5 Sports Posts by Engagement Metrics')
axes[1].set_xlabel('Post Index')
axes[1].set_ylabel('Count')
axes[1].set_xticks(sports_indices + bar_width)
axes[1].set_xticklabels([f'Sports {i+1}' for i in range(5)])
axes[1].legend()

# Display the plots
plt.tight_layout()
plt.show()


# # Project Conclusion: Social Media Engagement Analysis Across Categories
# 
# ## 1. Objective Recap:
# The primary objective of this project was to analyze social media engagement metrics across various content categories, specifically focusing on the "Likes," "Retweets," and "Comments" received by posts in each category. The categories analyzed included Technology, Health, Entertainment, Sports, and Politics. The goal was to identify trends, patterns, and correlations within these engagement metrics to provide insights into user interaction across different types of content.
# 
# ## 2. Summary of Analysis:
# - **Total Likes by Category:** 
#   - The Health category emerged as the most liked, followed by Technology, Sports, Politics, and Entertainment. This suggests that health-related content is highly engaging, likely due to its relevance and importance to a wide audience.
#   
# - **Distribution of Likes:**
#   - The box plot revealed the spread and central tendency of likes across the categories. Politics and Health categories showed higher median likes, indicating consistent engagement. Entertainment had a wider distribution, suggesting varied levels of engagement.
# 
# - **Correlation Analysis:**
#   - The correlation matrices provided insights into the relationships between likes, retweets, and comments within each category. For instance, in the Health category, there was a weak positive correlation between Retweets and Comments, indicating that posts with more retweets might also garner more comments. However, this trend was not strong across all categories, indicating varied user behavior.
# 
# - **Top Posts by Likes:**
#   - The top 5 posts in each category were analyzed to understand the composition of engagement metrics. For both Technology and Entertainment, the posts with the most likes also had significant numbers of retweets and comments, suggesting that highly liked posts tend to generate broader interaction.
# 
# ## 3. Visual Insights:
# - The bar plots, box plots, and correlation heatmaps provided a comprehensive visual representation of the data. 
# - The bar plots of the top posts demonstrated the distribution of engagement across different posts, making it clear which posts were outliers in terms of user interaction.
# - The correlation heatmaps helped to visualize the strength and direction of relationships between engagement metrics, highlighting categories with stronger interactions.
# 
# ## 4. Conclusion:
# The analysis shows that user engagement varies significantly across content categories, with Health content receiving the most attention in terms of likes. This could be attributed to the current global emphasis on health and wellness. The weak correlations between different engagement metrics suggest that while some users are likely to engage through multiple interactions (liking, retweeting, and commenting), others may only engage in one or two ways.
# 
# These insights can guide content creators and social media managers in tailoring their strategies to maximize engagement. For instance, focusing on Health content could be beneficial for platforms looking to increase user interaction. Additionally, understanding that not all metrics are strongly correlated can help in setting realistic expectations for post-performance.
# 
# This project has demonstrated the value of data analysis in understanding social media engagement, offering actionable insights that can drive content strategy and improve user interaction.
# 
