# NYCDSA_MachineLearningProject



NYC Data Science Academy Machine Learning Project
House Sale Value Modeling

For our machine learning project, we were tasked with predicting the sale price of homes based on the Ames Housing dataset. The data set catalogues the major characteristics of 1460 houses that were sold in Ames, Iowa during an undisclosed time period.


# Exploratory Data Analysis

For anyone who has ever purchased a house, many of the major factors governing the sale price seem pretty obvious. Location, location, location, for one. In the Ames Housing data set, the neighborhood, house style, year built, and overall quality scores immediately stood out as factors that are likely to have a major effect on the sale price. What was less clear is how the other 75 features would factor in. Upon analysis, it was apparent that the median sale price is quite significantly different for each neighborhood, as one might expect.

There is also a strong linear relationship between sale price and features such total size (blue), and overall quality (red). For other features such as YearBuilt (green), it's clear that new houses are selling for substantially higher prices, though the relationship isn't strictly linear.

Data Cleaning

There were a number of issues that had to be dealt with before modelling the data. First, the distribution of sale prices is not normally distributed, making it likely that my models will be error prone for the houses at the top end of the price spectrum. To correct for this, I did a log transformation. This improved the fit significantly, though the lower end of the price spectrum was still potentially going to cause some issues.

Second, there is was a substantial amount of missing data across several columns. For example over 99% of the data are missing for Pool QC, likely due to most houses simply not having a pool. In other words, much of the missing data points are not missing at random. To account for this, I imputed NA values as "no pool". Similarly, I inferred that missing values related to basement, garage, or fireplace indicated a lack of those features as well. However, in the case of other features such as the veneer type (e.g. stone/brick), missing values were rare (only 0.5% of the data), potentially suggesting that they could have been missing at random, and that it would be more appropriate to impute a specific value.

To impute these values, I first separated the data into quantitative or categorical groups. I then sub-divided the categorical data into nominal and ordinal bins. At that point, I created a database of typical values for each feature by sub-grouped each by neighborhood and identifying the most common value (categorical columns), or median value (numerical columns) for each. I then replaced the missing cells with the corresponding common values.

Numerical Conversion

Following imputation, I converted the categorical data to numerical data. To ensure that numerical assignments correlate with higher sale prices, I sorted all categorical features according to the median sales prices for that feature value and assigned each the indexed value. For example, the neighborhoods were numbered 1-25 with the most expensive neighborhoods getting the higher numbers.

Finally, outliers were removed. Specifically, the SaleCondition feature was reduced down to only Normal sales, while homes with more thatn 4000 square feet of living space were removed.

Data modeling

Using the cleaned data set, I trained a random forest model, and used a grid search to fine tune the parameters. Specifically, I tested a range of estimators between 200 and 1000, auto versus square root for the maximum number of features per split, and a maximum depth up to 30. To help prevent overfitting, I kept the minimum number of samples per leaf at a minimum of 3-7. Despite several attempts at optimization, I found that the model only achieved an R-squared value of 0.86 – 0.88. However, when I graphed the errors out, it became clear that the model had a slight tendency to overestimate prices at the low end and under estimate the price of higher end homes. 

This indicated that it should be possible to correct predicted values according using a simple linear model. By plotting out the absolute error in dollars against the predicted value, I came up with a dollar amount to be added or subtracted from each of the initially predicted values. These corrected values were significantly more accurate than the initial predictions, increasing the R-squred value to 0.92.

Secondary Model Testing

Given that the random forest model wasn’t perfect, I also tested an XG Boosting model and a penalized linear regression model. Though the XG Boosting model had a similar R-squared value as the random forest model at 0.87, Lasso regression yielded a higher R-squred value (0.91) for the test data. Using cross validation, I was able to identify an optimal lambda value (~2.5) which allowed me to avoid overfitting.

In the end, the Lasso model's predictor importance function demonstrated that the strongest predictors are exactly what one would expect – overall house quality, size, and the neighborhood.
