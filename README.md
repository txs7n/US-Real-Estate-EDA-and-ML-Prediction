# US-Real-Estate-EDA-and-ML-Prediction

## Overview
This repository contains a comprehensive project focused on analyzing and predicting real estate prices. It showcases a blend of exploratory data analysis (EDA) and advanced predictive modeling to derive insights and forecast property values in a diverse US real estate market.

## Data
The [data](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) used in this project was made available on Kaggle. The data is a CSV file with 10 columns:

- status (housing status i.e. ready for sale or ready to build)
- bed (number of beds)
- bath (number of bathrooms)
- acre_lot (Property / Land size in acres)
- city (city name)
- state (state name)
- zip_code (postal code of the area)
- house_size (house area/size/living space in square feet)
- prev_sold_date (Previously sold date)
- price (housing price)

## Tools
- Python: For data processing and modeling.  
- Pandas & NumPy: For data manipulation.   
- Matplotlib & Seaborn: For data visualization.  
- CatBoost: For predictive modeling.  
- LightGBM: For predictive modeling  
- Scikit-learn: For model evaluation and metrics calculation.  

## Process
As with any data analysis/science process, this project began with importing the necessary libraries and reading the CSV file downloaded from the Kaggle source into a dataframe.  

I started with an initial data understanding to get a general overview of the data and noticed some nulls in the dataframe. I especially noticed a large number of nulls from the 'prev_sold' column (over 50% of the data was null). Since this much data was missing from this column, it only makes sense to remove it. But before that, I wanted to be sure of its correlation with the target variable **price**.  

Upon evaluation, I found the correlation between the **prev_sold** column and **price** to be approximately 0.01. This near-zero correlation suggests that there isn't a strong linear relationship between how long ago a house was sold and its current price.  

This means that the time elapsed since the last sale does not linearly predict the current price, hence would not affect the machine learning model in any significant way. This gave enough confirmation to remove the column. 

### Handling Nulls & Removing Outliers
There were still a significant number of nulls in the remaining features; 14%, 12%, 29%, and 32% of nulls for columns **bed**, **bath**, **acre_lot**, and **house_size** respectively. The first thing I did was to remove all nulls peculiar to all features as it'll be difficult to fill these values with corresponding features.  

After that, I decided to fill the nulls via imputation, which we normally do with the mean. But means are affected by outliers. It was pertinent to investigate outliers (if and how they exist) in the dataset before going ahead with this process.  


![feature_outliers](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/e51ebd75-413a-45cf-8d65-8a262e3ab18f)

  
There was a considerable amount of outliers in the dataset as seen in the boxplots above. It is common practice to use the interquartile range (IQR) technique to eliminate upper and lower-bound outliers but the nature of every dataset is unique.  

Additionally, the nature and scale of each feature in a dataset is also unique. Outlier removal, I have found, needs to be customized; must come from a place of deep understanding of the data and its features. I went into the outlier removal process of this dataset with this mindset.  

The first thing I did was to investigate the 99th, 95th, and 90th percentile (upper bound outliers) of the dataset. There were outrageous observations for some of these features. For example, there was an observation where there were 99 bedrooms and 198 bathrooms.  

At first glance, one might think it a luxurious, hollywood-esque property. But the house_size feature tells us another story; it is quite impossible to have these many bed and bathrooms in a property of only 14,000 sq. ft. given the [average bedroom size to be 132 square feet](https://cedreo.com/blog/average-bedroom-size/). There were other interesting outliers in this upper bound as well.  

The 95th percentile was a sweet spot to have my outlier upper bound cutoff because the minimum values in this threshold made common sense. For example, it is not absurd to have a 6-bed property (the minimum bedroom value in this threshold) nor is it ridiculous to get a property for 2.7 million in New York City.  

As for the lower bound outliers, I found that there weren't lower bound outliers for some features like the number of bedrooms, for example, whose minimum value in the dataset was 1. And a 1-bedroom apartment is not an outlier in any sense.  

However, for features like house_size, price, and acre_lot, there seemed to be some lower bound outlier inferred from common sense observation. For example, it is unreasonable to have a house listed as $0 for sale.  

For the lower bound outliers I settled for the 5th percentile because, again, the maximum values for the affected features in this threshold were reasonable.  

Once I had removed the outliers it was now time to handle the nulls. Even though we have removed considerable outliers, the dataset contained large variation in values. Which is expected and fairly normal in the real estate marketplace. Additionally, I corroborated the 'skewness' of the distribution of each feature.  

The best option to fill the nulls, therefore, was to use the median. 

## Exploratory Data Analysis
The exploratory data analysis of this real estate dataset mostly constituted the Bivariate analyses of the features with the target feature **price**.  


**Bivariate Analysis of Bedroom and Price**  

![median_home_price_by_bedroom](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/680f95a4-a773-4eb2-ad27-aa133c0bb52b)

For this plot, there is an overall increasing trend in median price as the number of bedrooms increases from 1 to 6. This is expected since more bedrooms generally correspond to larger properties which are likely to be more expensive.

However, the median housing price with respect to number of beds does not seem to follow a linear trajectory. This could be a result of other factors such as state, city, house size, etc., which we will analyze subsequently.  


**Bivariate Analysis of Bathrooms and Price**  

![median_home_price_by_bathroom](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/1d0b4875-ebd9-44d3-b343-0cb7a18bac35)

This graph, unlike the previous, shows a more linear upward trend between price and number of bathrooms. So irrespective of other factors, the median home price increases as the number of bathrooms increases.

  
**Bivariate Analysis of State and Price**
![median_home_price_by_state](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/5781c034-ed7b-4811-ac7e-de00bd325e6b)

From this plot, we can see that the most expensive place to live in is New York, while the least expensive place to live is Puerto Rico.  


**Bivariate Analysis of City and Price**  

![median_home_price_by_city](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/6ff5cb12-43ca-4b1a-bff4-39552424e600)

From this graph, the top 10 expensive cities to live are in Massachusetts, New York, and New Jersey.  


**Bivariate Analysis of House Size and Price**  
The house_size feature is continuous data and te best way to represent this is via scatterplot. However, the plot can easily become messy and incoherent when there are too many overlapping points. One way around this to to sample the data. 

![relationship_between_house_size_and_price1](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/cf315b53-e0e2-4396-83eb-cb633a7f5c22)

Running this plot using different random samples, we see a positive correlation between house size and price, which is expected in real estate markets. Larger houses tend to be priced higher than smaller ones. Despite the overall trend, there's considerable variability in prices for houses of similar sizes.  

This suggests that factors other than size, such as location, could be a factor. Additionally, this distribution does not form a tight line, indicating that the relationship between house size and price is not strictly linear and is influenced by other factors.  


​**Bivariate Analysis of Acre Lot and Price**  
I had the same dilemma with the acre_lot feature (i.e. continuous data on scatter plot which can get very messy). Hence, I employed the same method to analyze this feature with respect to price.

![relationship_between_acre_lot_and_price1](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/31501266-f347-4c6a-888b-dd3590b0b2c4)

This plot does not show a strong correlation between price and acre lot. This means that, on average, as the lot size increases, the price of the property does not necessarily increase and in most cases, it even decreases.  

This finding is somewhat counterintuitive, as one might generally expect larger lots to command higher prices. But this could be influenced by the specific dataset or market conditions. For instance, larger lots might be located in more rural areas where property values are generally lower.

To corroborate this assumption, I plotted the distribution of large-lot  and small-lot homes among states and found something interesting...

![distribution_of_10+_acre_properties_among_states](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/9fea4167-c98d-4ecc-a62f-090754b629c8)


![distribution_of_less_than_2_acre_properties_among_states](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/35c8edd5-b6a7-4fea-8293-23e3bd4eae21)


From these two pie charts, our assumption is corroborated; Most properties with large acre lots are mainly situated in rural areas (Vermont, Maine, New Hampshire). While properties with small acre lots (52.1%) are mostly located in urban regions (New York, New Jersey). This explains why we have a negative correlation between price and acre_lot.  


​**Bivariate Analysis of Zip Code and Price**
![top_15_zip_codes_by_median_price](https://github.com/txs7n/Retail-Business-Sales-Data-Analysis/assets/118135226/9006cb9a-33f3-46a4-94f4-0a350559440b)

As expected, the most expensive zip codes by median home prices are in Massachusetts, New York, and New Jersey.  


## Feature Engineering & Modeling  
This real estate dataset contained four categorical features - **status**, **city**, **state**, and **zip_code**. While the status feature only contained 2 unique categories, the other three contained numerous; 2,487, 18, and 3,140 unique categories for the city, state, and zip_code features respectively. Using one-hot encoding to encode these features for our model will lead huge increase in the dataset's dimensionality, which can cause overfitting and increased computational complexity.  

We could utilize other encoding techniques such as frequency and target encoding but each presents its unique limitations with respect to this dataset. Frequency encoding works best in cases where the frequency is somewhat related to the target variable. Target encoding depends heavily on the mean which can be affected by the established skewness of this data. The best bet was to choose a model that can handle categorical data as-is. **LightGBM** and **CatBoost** Models offer this capability.  

Starting with the LightGBM, we trained a base model and got the following evaluation metrics:
- Mean Absolute Error (MAE): 145698.63880546845  
- Mean Squared Error (MSE): 56436442815.639626  
- Root Mean Squared Error (RMSE): 237563.55531865495  
- R-squared (R²): 0.7517769514334849

These are very high error values so I proceeded to use the CatBoost Regressor model and got the following metrics:
- Mean Absolute Error (MAE): 148825.2731400052  
- Mean Squared Error (MSE): 60003835689.191536  
- Root Mean Squared Error: 244956.80372096534  
- R-squared (R²): 0.7360865731897639

These are equally high error values. The good thing about the CatBoost model is, it gives a visual representation of how the model performs with each iteration. I found that the model indicated convergence around RMSE = 160,000. So we either had to improve these base models via hyperparameter tuning or add more features via feature engineering. I proceeded with the latter.  

I added four features namely:  
- price_per_sqft: Price of the house by the house size 
- bed_to_bath_ratio: Number of bedrooms by number of bathrooms
- bed_to_acre_ratio: Number of bedrooms by acre lot
- bath_to_acre_ratio: Number of bathrooms by acre lot

I decided to use the LightGBM model once again with this new dataframe that has these additional features and found an improvement in the model's performance. LightGBM Regressor gave the following metrics:  
- Mean Absolute Error (MAE): 9266.099757717522
- Mean Squared Error (MSE): 313253540.8309629
- Root Mean Squared Error (RMSE): 17698.970050004686
- R-squared (R²): 0.9986222244882916

While the CatBoost Regressor model gave the following evaluation metrics:  
- Mean Absolute Error (MAE): 5784.1493203291275
- Mean Squared Error (MSE): 208237052.75840005
- Root Mean Squared Error: 14430.4210873557
- R-squared (R²): 0.9990841159810683


## Conclusion  
- The CatBoost model gives better evaluation metrics after adding a couple of features. Hence, we will be adopting this model for our dataset.  

- A Mean Absolute Error of 5784.15 suggests that on average, the model's predictions are off by about $6,000.  

- The RMSE value punishes the model even more because it gives relatively high weight to large errors. The Root Mean Square Error tells us that our model's predictions are off by $14,000.  

- Relative to the high prices present in this dataset (up to $2 million), an RMSE of 14,430 against a backdrop of these high property prices is relatively small. It indicates that the model's predictions are quite close to the actual values, especially for higher-priced properties.  

- An R-squared value of nearly 1 is extremely high and suggests that the model explains almost all the variance in the property prices. This is an indication of a very good fit.  


Normalizing the Root Mean Squared Error (RMSE) can be a useful way to put the error metric into context, especially when dealing with a wide range of values in your target variable, like in your real estate data. Normalized RMSE expresses the error as a percentage of the range of the target variable, making it easier to interpret regardless of the scale of the values.

I calculated a normalized RMSE of 0.54% indicating that the typical prediction error of the model is just over half a percent of the range of your property prices.  

Given that this range spans from $75,000 to $2,749,000, such a small percentage is a strong indicator of the model's accuracy.  


## Recommendation  
- To further improve the model and get the best fit, we can always use grid search cross-validation to tune the parameters of our model.
- As we saw from the modeling of our dataset, feature engineering greatly improved the performance. So we can also consider adding more features from external data if available (e.g. proximity to facilities, property conditions such as 'new', 'good', 'needs work, median household income of the area, etc.).
