# MachineKnight_hackathon

## WEBAPP LINK :
https://joygeo007-machineknight-hackathon-main-2agjg1.streamlitapp.com/


## Objective and Dataset Info
The dataset consists of housing properties located in Bengaluru and Chennai. 
The objective is to create an ML model that can predict the rent of a house based on the given properties. 
The model has been trained using the train data and makes predictions for the test data. Train.csv has dimensions: 20500 rows X 25 columns, whereas, test.csv has dimensions: 4500 rows X 24 columns

## Data Cleaning 
1. No null values were found in the dataset
2. No duplicate values were found in the dataset

## Exploratory Data Analysis (EDA) 
1. Most of the East facing flats have rent in the range of 15000-20000 
2. Most of the North facing flats have rent less than 10000 
3. Most of the 2 BHKs have rent in the range 15000-20000
4. There was slight correlation between total number of floors and floor
5. There was slight correlation between bathroom,property_size and rent
6. Properties with either one of gym, swimming pool or lift facilities have higher chances of having the other 2 amenities.

## Data preprocessing and Feature Engineering
1. Dropped columns with high cardinality
2. We haven't taken locality into consideration as we had the variables latitude and longitude
3. Amenities column had already been processed , so we removed it.
4. Since it is regression problem, we encoded all the categorical variables:<br>
    4.1. Categorical variables with distinct hierarchical values were label-encoded<br>
    4.2. Rest of the categorical variables were one-hot encoded.
5. Separated target(rent) and predictor variables.
6. Scaled the train and test data using Standard Scaler

## Approach
1. After removing the columns previously mentioned, we performed Feature Engineering and EDA to gain initial insights from the given dataset
2. We used Sweetviz for EDA besides doing the same by ourselves
3. Next, we checked for the correlation between the columns
4. We started with the Linear Regression model initially, which was followed by Ridge and Lasso
5. After this, we used Gradient Boosting Regressor and Support Vector Regressor
6. This was followed by Random Forest Regressor and Decision Tree
7. For finding the best model, we calculated a particular model's root mean square error (RMSE) and R2 Score
8. We found out that Decision tree regressor and random forest regressor were giving the most optimal score
9. We also deduced that decision tree regressor was getting overfitted. 
10. We first tried to postrun the Decision Tree. Then we used RandomizedSearchCV, for a specific range of parameters, to find the best optimal score
11. We did the same thing with Random Forest Regressor. Then we got a specific range of parameters, which is giving us the best possible score for a model till now.
12. We used GridSearchCV on the range of the specific parameters and found out that the best suitable parameters that gave us the best possible model score



