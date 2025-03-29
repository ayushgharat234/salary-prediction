import pandas as pd# Importing the pandas library

data = pd.read_csv("data/salary_data.csv")  # Loading the data from the salary_data.csv file

"""Selecting only 'Years of Experience', 'Salary' for analysis and motive as the project hypothesis of 
independence of the other features on the salary
"""

selected_features = ["Years of Experience", "Salary"]  # List of the selected features
extracted_data = data[selected_features]  # extracted dataframe

# Saving the data in data.csv file in the data folder
extracted_data.to_csv("data/data.csv", index=False)