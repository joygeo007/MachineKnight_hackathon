# MachineKnight_hackathon

## Data Cleaning
1. No null values were found in the dataset
2. No duplicate values were found in the dataset

## Feature Engineering and Exploratory Data Analysis (EDA) 
1. Most of the East facing flats have rent in the range of 15000-20000 
2. Most of the North facing flats have rent less than 10000 
3. Most of the 2 BHKs have rent in the range 15000-20000
4. We haven't taken locality into consideration as we had the variables latitude and longitude
5. We haven't added amenities for the model as it was not very relevant to the case
6. We removed rent as it was the target variable

## Approach
1. After removing the columns previously mentioned, we performed Feature Engineering and EDA to gain initial insights from the given dataset
2. We used Sweetviz for EDA besides doing the same by ourselves
3. Next, we checked for the correlation between the columns
4. We started with the linear regression model initially, which was followed by Ridge and Lasso
5. After this, we used Gradient Boosting Regressor and Support Vector Regressor
6. This was followed by Random Forest Regressor and Decision Tree
7. For finding the best model, we calculated a particular model's root mean square error (RMSE) and R2 Score
8. We found out that decision tree regressor and random forest regressor were giving the most optimal score
9. We also deduced that decision tree regressor was getting overfitted. 
10. We first tried to postrun the Decision Tree. Then we used RandomizedSearchCV, for a specific range of parameters, to find the best optimal score
11. We did the same thing with Random Forest Regressor. Then we got a specific range of parameters, which is giving us the best possible score for a model till now.
12. We used GridSearchCV on the range of the specific parameters and found out that the best suitable parameters that gave us the best possible model score
