import pandas as pd # data science essentials
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


file ='googleplaystore.csv'
apps = pd.read_csv(file)

#viewing the head of the data 
with pd.option_context('display.max_rows', 50, 'display.max_columns', 50):
    print(apps.head())
    print(apps.describe())
    

#which columns are numeric and which columns are not
apps.info()
apps.shape

#displaying the all column names 
apps.columns
#exploring the columns and unique values
apps['App'].unique()
#Comment: App refers to app name

apps['Category'].unique()

apps['Type'].unique()
#Comment: we can categorize the types of the apps with 0, 1 and 2

#Number of unique values in the genres
apps['Genres'].nunique()
#since there are 120 categories, it is better to leave the column as a string

#there are 1378 unique categories for the Last Updated column.
apps['Last Updated'].nunique()
apps['Current Ver'].nunique()
apps['Android Ver'].unique()


#we only have 1 column that is numeric. But we can convert Size, Installs and Price to
#numeric

#Changing the column Size to numeric
apps['Size'] = apps['Size'].apply(lambda x: x.replace('M','000000'))
apps['Size'] = apps['Size'].apply(lambda x: x.replace('k','000'))
apps['Size'] = apps['Size'].apply(lambda x: x.replace('Varies with device','-4'))
apps['Size'] = apps['Size'].apply(lambda x: x.replace('1,000+','1000'))
apps['Size'] = apps['Size'].astype(float)


#Changing Installs Column to numeric
apps['Installs'].unique()
#replacing + sign 
apps['Installs'] = apps['Installs'].apply(lambda x: x.replace('+',''))
#replacing commas
apps['Installs'] = apps['Installs'].apply(lambda x: x.replace(',',''))

#replacing the value of Free to -1
apps['Installs'] = apps['Installs'].apply(lambda x: x.replace('Free','-1'))

#changing type to float
apps['Installs'] = apps['Installs'].astype(float)

#alternative way to convert values to numeric
#apps['Rating'] = goog_play['Rating'].apply(pd.to_numeric, errors='coerce')
#apps['Reviews'] = goog_play['Reviews'].apply(pd.to_numeric, errors='coerce')

#Changing Price column to numeric
apps['Price'].unique()
apps['Price'] = apps['Price'].apply(lambda x: x.replace('$', ''))
#Replacing Everyone with -2
apps['Price'] = apps['Price'].apply(lambda x: x.replace('Everyone', '-2'))
#Changing to numeric
apps['Price'] = apps['Price'].astype(float)

#Now we converted everything necessary to floats and integers. We can look at the 
#distributions
#Checking for missing values

#Flagging missing values
print(
      apps.isnull()
      .any()
      )

#We have missing values in the rating, type content rating, current vers and android version
#columns
apps.columns[apps.isnull().any()]

#Printing number of missing values for the columns that have at least one missing value
for col in apps.columns:
    if apps[col].isnull().any() :
        print(f"""{apps[col].name} : {apps[col].isnull().sum()}""")
        

# As a percentage of total observations (take note of the parenthesis)
print(
      round((apps.isnull().sum())
      /
      len(apps), 2)
)

apps.columns

#We can create some histograms 
#apps.columns
with pd.option_context('display.max_rows', 50, 'display.max_columns', 50):
    print(apps.describe(include=[np.object]))



#Dropping the missing values and looking at the distribution
apps_dropped = apps.dropna()

apps_dropped.columns
#Plot for Rating
plt.hist(apps_dropped['Rating'], bins='fd', color='green')
plt.title("Rating")
plt.xlabel("Value")
plt.ylabel("Frequency")

#Plot for Price
plt.hist(apps_dropped['Price'], color='blue')
plt.title("Price")
plt.xlabel("Value")
plt.ylabel("Frequency")


#Plot for Size
plt.hist(apps_dropped['Size'], bins='fd')
plt.title("Size")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.hist(apps_dropped['Installs'])
plt.title("Installs")
plt.xlabel("Value")
plt.ylabel("Frequency")

#Plotting with Seaborn
sns.distplot(apps_dropped['Installs'])

sns.distplot(apps_dropped['Size'])

sns.distplot(apps_dropped['Price'])

sns.distplot(apps_dropped['Rating'])


#Looking for the categorical variables distribution
#Type
type_plot = sns.countplot(x="Type", data=apps_dropped)


#Type for catefory
category_plot = sns.countplot(x="Category", data=apps_dropped)
sns.set_style("ticks", {"xtick.major.size":5, "ytick.major.size":7})
#sns.set_context("notebook", font_scale=0.5, rc={"lines.linewidth":0.3})
#improving the figure size
sns.set(rc={'figure.figsize':(11.7,8.27)})
#rotating the xtick labels
for item in category_plot.get_xticklabels():
    item.set_rotation(90)

plt.savefig("category.jpeg")

sns.countplot(x="Genres", data=apps_dropped)

#genres; ıt doesnt make sense to plot a countplot for genre because it has
#too many different genres, in fact 115 unique genres, when we run the code below
apps_dropped["Genres"].nunique()

#We can subset the Genres by setting some tresholds
#First we can look at the genres that have the most number of apps
genres = apps_dropped.Genres.value_counts().sort_values(ascending=False)


#Now creating a df, order the number of apps we can subset
apps_dropped_copy = apps_dropped
gdf = apps_dropped_copy.groupby("Genres").count().sort_values(by = "App", ascending=False)
#changing index to a column
gdf.reset_index(level=0, inplace=True)
gdf.columns

#looking at the distribution of number of apps per genre 
gda_p = sns.distplot(gdf["App"])
gda_p.set_title("Number of Apps per Genre")
gda_p.set_xlabel("Number of Apps")

#Genres that have more than 50 apps
gdf_hi = gdf[gdf["App"] > 50]

#now we can plot with a barplot: which genres have the most number of apps 
genre_plot = sns.barplot(x="Genres", y= "App", data=gdf_hi)
sns.set_style("ticks", {"xtick.major.size":5, "ytick.major.size":7})
#sns.set_context("notebook", font_scale=0.5, rc={"lines.linewidth":0.3})
#improving the figure size
sns.set(rc={'figure.figsize':(11.7,8.27)})
#rotating the xtick labels
for item in genre_plot.get_xticklabels():
    item.set_rotation(90)

#genres that has low number of apps
gdf_low = gdf[gdf["App"] < 5]

genre_plot = sns.barplot(x="Genres", y= "App", data=gdf_low)
sns.set_style("ticks", {"xtick.major.size":5, "ytick.major.size":7})
#sns.set_context("notebook", font_scale=0.5, rc={"lines.linewidth":0.3})
#improving the figure size
sns.set(rc={'figure.figsize':(11.7,8.27)})
#rotating the xtick labels
for item in genre_plot.get_xticklabels():
    item.set_rotation(90)


#Looking at the relationships
plot_1=sns.stripplot(x="Installs", y="Price", data=apps_dropped, hue="Type")
for item in plot_1.get_xticklabels():
    item.set_rotation(90)
