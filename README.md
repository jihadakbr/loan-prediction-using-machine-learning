
# Loan Prediction Using Machine Learning

An organization has a goal of forecasting which individuals are more likely to default on their consumer loan product. The company has collected information about previous customer behavior based on their observations. With this information, the organization intends to make predictions about the risk levels of new customers when they are acquired, thus enabling the organization to determine which customers pose a higher risk and which ones do not.


## Dataset Source

[https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)
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
* create a GUI
## Results

![Results of Machine Learning Model](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjpWVYFf9_xjaxBFE-E6HQbieYpOfFy0JEhRaUG3LupRuk3xMKZMfJ830-EhVSlwTnIRyX3HkcGg_m4mm__GJxC-h__37qg2OlM6XFNObBywK6xRzBLzpmW4t0KjNi1jHjq91VoIMyHxHCKOsCYX-KrVj-wEwfmzkStuL_QTus0PEGS7j48PV46GYKo/s1600/results-of-machine-learning-model.png)

![GUI for Loan Prediction Using Machine Learning](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEifYOEIPGq1TORVGRqLIHXSnPtAEsTtWgcB61iyUZg22AbDEJc2Np5FDpi-pkAoUi8Ctre6x3gphyBsnlYetxbrLIUgVfZqIdN4f6u27TFSsLUV2hpOqeNHjbQvTzh8jykXfk12zP8MbNV11IsZGzYo-fOvodi3bGovwCnTg0RipiADYnUMyENw9KPy/s1600/gui-loan-prediction-using-machine-learning.png)

![GUI for Loan Prediction Using Machine Learning 2](https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEiJrvFmagHzvpc3vgLDIczMaon5aQli9BuDMvrWnlZCCNt88twx8AKg-DAU9__U9-Qc4y0M-RPaXBCN-MRVcxWUYYz42YKNEYS6lTbS7fIqR4cx-EgvA8wY0ovdVv9vUuPxfHKjlqYy1YY5v1DxQk2PLByNBGPnN3BCVAev5SCLAK99RerUuaiWiN3j/s1600/gui-loan-prediction-using-machine-learning-2.png)
## Required Files to Run the GUI

1. ET_model
2. scaler_ET
3. prof_c.csv
4. city_c.csv
5. state_c.csv

Run the "Loan Prediction Using Machine Learning.ipynb" first to get those files.
## Conclusions

* The datasets contain no missing values, duplicates, or outliers.

* 87.7% of people are considered non-defaulters, indicated by a risk flag of 0, while the remaining 12.3% are identified as defaulters, indicated by a risk flag of 1.

* The Extra Trees Classifier model is deemed the most appropriate for the dataset, with an average cross-validation score of 90.13%. The Random Forest Classifier model is also a viable option, with a score of 90.10%.

* A successful graphical user interface (GUI) has been created.