
# Loan Prediction Using Machine Learning

An organization has a goal of forecasting which individuals are more likely to default on their consumer loan product. The company has collected information about previous customer behavior based on their observations. With this information, the organization intends to make predictions about the risk levels of new customers when they are acquired, thus enabling the organization to determine which customers pose a higher risk and which ones do not.


## Dataset Source

[Loan Prediction Based on Customer Behavior (Kaggle)](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)
## Content of Dataset

1. ID: id of the user
2. income: income of the user
3. age: age of the user
4. experience: professional experience of the user in years
5. profession: profession
6. married: whether married or single
7. house_ownership: owned or rented or neither
8. car_ownership: does the person own a car
9. current_job_years: years of experience in the current job
10. current_house_years: number of years in the current residence
11. city: city of residence
12. state: state of residence
13. risk_flag: defaulted on a loan (target variable)

* The risk_flag indicates whether there has been a default in the past or not.
* risk flag = 1 → defaulter: a person who fails to fulfill a duty, obligation, or undertaking, especially to pay a debt.
* risk flag = 0 → non-defaulter
## Tools

Jupyter notebook: 
* data exploring and cleansing
* data visualization
* looking for the best suitable machine learning model
## Results

Due to the limitations of GitHub, you can download the trained_model_w_pipe.joblib file from this [Google Drive link](https://drive.google.com/file/d/1PaB_XKl928h76PNOrLa7Y-6AXgXFSvQC/view?usp=sharing) (200 mb)

![Results of Machine Learning Model](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEibjbW1CQG86kv47rxR-7tvsB73qy0694sucg2kC8KHB1rTAC9BIz58k6oGaRqzDSIqoc16yQ5O4ZMJVdpF3kb9GYf5_4tHV0pBqPsT5qz0rfzwv8ETVlu-Sr9HJKyCsz0rDHDBxJlv_il5mSWdWc6Sh2SVBT4OxxBiqb9RGzjUE0IJDC-uCHgHigKG/s1600/results-of-machine-learning-model.png)

![Results of Machine Learning Model 2](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEgEoZaWXcKT6fSFgLnbQMlALzpvRGRLMhtpGyYWXqz5ac1cHiSqnB1WmLevUy8c9iaC1Z-9To5TWXubuEber1nAPGUGWUxwyYR_HZ_Z9LrfMBg2me6R5CE3sVHQt58pRETf24shO3_x74kZ4UWGBePNYLoomF7DpGcj-VKc_qnFlCF4QN1Np_2gE-ZE/s1600/results-of-machine-learning-model-2.png)
## Conclusions

* The datasets contain no missing values, duplicates, or outliers.

* 87.7% of people are considered non-defaulters, indicated by a risk flag of 0, while the remaining 12.3% are identified as defaulters, indicated by a risk flag of 1.

* The Random Forest Classifier model is deemed the most appropriate for the dataset, with an average cross-validation score of 89.30%. The Extra Trees Classifier model is also a viable option, with a score of 89.22%.
