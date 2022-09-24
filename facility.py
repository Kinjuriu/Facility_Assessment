#Load the Health Facility Asssessment File

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# import scikit-learn as sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.cluster import KMeans


csv_file ='Health_Facility_Assessment.csv'
facility = pd.read_csv(csv_file) # into a dataframe


# PREVIEW THE DATA
print(facility.head()) #print the first five rows
facility.info() # Checking the info of the dataset
print(facility.shape) #How many rows and columns
print(facility.isnull().sum().sum()) #How many null values?
print(facility.isin(["---"]).sum().sum()) #How many '---' are there?


#DATA CLEANING
facility.replace('---', np.nan, regex=False, inplace=True) # replace the dashes with Nan
print(facility.isin(["---"]).sum().sum()) #check whether the dashes are gone

facility_df = facility.replace(r'^\s*$', np.nan, regex=True) # replace empty values with NAN 

print(facility_df.isna().sum().sum()) #check total number of NAN values
print(facility_df.head(6)) #print first six columns

#Calculate percentage of missing values in columns
percentage_columns = facility_df.isnull().mean() * 100
print(percentage_columns)


# Delete columns containing either 90% or more than 90% NaN Values
perc = 90.0
min_count =  int(((100-perc)/100)*facility_df.shape[0] + 1)
df = facility_df.dropna( axis=1, 
                thresh=min_count)
print(df.head())               

print(df.shape) #Lets check the shape of the dataframe now

#Check for duplicate entries
df.duplicated().sum()

#Columns that have a single observation or value are zero-variance predictors
counts = df.nunique() # get number of unique values for each column 

# record columns to delete
to_del = [i for i,v in enumerate(counts) if v == 1]

# drop zero-variance columns
df.drop(df.columns[to_del], axis=1, inplace=True)
print(df.shape)

# Print the first row to look at the dataset
print(df.iloc[0])

# drop first and second column by index
df.drop(df.columns[[0, 1]],axis = 1, inplace=True)
print(df.head())

#FILLING IN MISSING VALUES FOR NUMERICAL COLUMNS
print(df.describe()) #We will get a summary statistic for numerical columns only

#Finding the median of the column having NaN
median_value=df['form.score_total'].median()
median_value=df['form.score_max_total'].median()

#Replace NaNs in column with the mode of values in the same column
df['form.score_total'].fillna(value=median_value, inplace=True)
df['form.score_max_total'].fillna(value=median_value, inplace=True)

#Get numerical columns only from the dataframe
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_data = df.select_dtypes(include=numerics)

#plot both columns to see distribution so as to impute with mean/median
fig = plt.figure(figsize =(10, 7))
numerical_data['form.score_total'].hist() 
numerical_data['form.score_max_total'].hist() 
plt.show()

# Feature Transformation

# lets use min max scaler, start by defining it
scaler = MinMaxScaler()

# transform data
scaled_numerical_data = scaler.fit_transform(numerical_data)
print(scaled_numerical_data)

# Separate latitude and longitude into two columns
lat = list() # Create two lists for the loop results to be placed
lon = list()

# For each row in a varible,
for row in df['form.facility_gps']:
    # Try to,
    try:
        lat.append(row.split(',')[0])
        lon.append(row.split(',')[1])
    # But if you get an error
    except:
        # append a missing value to lat
        lat.append(np.NaN)
        # append a missing value to lon
        lon.append(np.NaN)

# Create two new columns from lat and lon
df['latitude'] = lat
df['longitude'] = lon

# drop form.facility_gps
df.drop("form.facility_gps",axis = 1, inplace=True)
df.head()


# DEALING WITH MISSING VALUES FOR CATEGORICAL DATA

# Fill in missing values with the most frequent value of that column
df=df.fillna(df.mode().iloc[0])

# FEATURE SCALING
#Categorical Data Types: 
categorical_df = df.select_dtypes(exclude=['float64','int64'])

df.drop(df.columns[1:5],axis = 1, inplace=True) #drop columns with town names
df.drop("form.sections_to_review",axis = 1, inplace=True) #dropping column form sections to review
column_types = df.dtypes #check the type of columns

#convert object columns to float
object_to_float_1 = df.iloc[:,5:25].astype(float) # first range [5, 25]

object_to_float_2 = df.iloc[:,27:42].astype(float) # second range  [27, 42]

object_to_float_3 = df.iloc[:,145:161].astype(float) # second range  [145, 161]

# add two df's together
float_cols = pd.concat([object_to_float_1, object_to_float_2, object_to_float_3], axis=1)
print(float_cols.dtypes)

#NORMALIZE THESE FLOAT NUMBERS 
# lets use min max scaler, start by defining it
scaler = MinMaxScaler()

# transform data
scaled_float_data = scaler.fit_transform(float_cols)
print(float_cols.head())

# Feature transform
dataframe_1=df.iloc[:,0:2] #get specific columns
dataframe_2=df['form.health_centre_information.setting']
get_dummy_data = pd.concat([dataframe_1, dataframe_2], axis=1) #add both df's since the columns were in different locations


# Creating dummy variables for the categorical columns
dummy_data = pd.get_dummies(
    get_dummy_data,
    columns=get_dummy_data.select_dtypes(include=["object", "category"]).columns.tolist(),
    drop_first=True, #drop_first=True is used to avoid redundant variables
)

#Label/ordinal encode data with relationships
dataframe_3 = df.iloc[:,43:144] #get specific columns from the main dataset

# define ordinal encoding
encoder = OrdinalEncoder()

# transform data
ordinal_encoded_data = encoder.fit_transform(dataframe_3)
print(ordinal_encoded_data) #comes out as a numpy array

# FEATURE TRANSFORMATION lon & lat
# NB // Lat/Long coordinates can often be used as-is with tree-based models like Random Forest or Gradient Boost 
# that do not require data to be normalized. 
# Other models such as neural network models would usually require those coordinates to be normalized: 

#For now I will use kmeans clustering
coordinates_data = df.iloc[:,164:166]
print(coordinates_data.head(5))

#Get number of clusters
K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = coordinates_data[['latitude']]
X_axis = coordinates_data[['longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]

# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()

#Elbow curve, the point after which the distortion/inertia starts decreasing in a linear fashion

kmeans = KMeans(n_clusters = 2, init ='k-means++')
kmeans.fit(coordinates_data) # Compute k-means clustering.
coordinates_data['cluster_label'] = kmeans.fit_predict(coordinates_data)

centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(coordinates_data[coordinates_data.columns[1:3]]) # Labels of each point
print(coordinates_data.head())
# drop lon and lat columns 
coordinates_data.drop(coordinates_data.columns[[0, 1]],axis = 1, inplace=True)
print(coordinates_data.head())

#combine the data that has been transformed except ordinal_encoded_data and scaled_numerical_data which are in numpy.array
facility_assesment_df = pd.concat([coordinates_data, dummy_data], axis=1)
facility_assesment_df.to_csv('facility_assesment_df.csv', index=False)
























