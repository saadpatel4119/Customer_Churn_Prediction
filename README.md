## Customer Churn Prediction Project

#### Objective
Develop a machine learning model that predicts customer churn

#### Problem Statement
Given a dataset containing attribute of 1 lakh Customers where using the features available from dataset and define classification algorithms to identify whether the customer is churned or not, also to evaluate different machine learning models on the dataset.

#### Description Of Dataset
This dataset contains historical customer information of the customer, including customer 
attributes such as CustomerID, Name, Age, Gender, Location,	Subscription Length, Monthly Bill, Total Usage and whether they churned or not. The dataset is in CSV format.

#### Attribute Information
- CustomerID -> Customer's ID(numeric: from 1 to 100000)
- Name -> Customer's Name based on Customer ID(nominal: from Customer_1 to Customer_1000)
- Age -> Age of the customer in years(numeric: from 18 to 70)
- Gender -> Sex of Customers(Binary: Male and female)
- Location -> Location of the Customer(Nominal: Los Angeles, New York, Miami, Chicago, Houston)
- Subscription_Length_Months -> Subscription Duration in Months(numeric: from 1 to 24)
- Monthly_Bill -> Monthly Charges based on Usage(numeric: 30 to 100)
- Total_Usage_GB -> Total Usage in GB(numeric: 50 to 500)
- Churn -> Customer Churned or Not Churned(Binary: 1,0)

#### Approach
After feature engineering on the dataset, I have observed that there are no missing values in the dataset. After visualization of different attributes with respect to Churn and Not Churned, I came to a point that the dataset provided is highly balanced and the datapoints are equally distributed amony Churned and Not Churned. Coming to the Categorical features of the dataset, we had 3 categorical features. Out of three, I had dropped the attribute "Name" as the values in it were just an add on of "Customer_" to CustomerID. On the remaining two features, I approached with One-Hot Encoding Technique rather than preferring Frequency or Mean Encoding as there were very less unique values in each attribute.
Observing the statistics of the dataset, I found out the mean of all the attributes were not on a similar scale and hence performed Standardization(scaling features to have a mean of 0 and a standard deviation as 1) to bring them on a similar scale. In the model training part, the dataset was splitted into train and test and different machine learning algorithms were applied on the train dataset. After training the models, the performance each model was calculated using various model evaluation parameters and the best model was selected based on the Receiver Operating Characteristic(ROC)-Area Under the Curve(AUC) score.

#### Machine Learning Algorithms used
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- K-Neighbors Classifier
- Naive Bayes