import pandas
import os

print("hello")

csv_file_path = os.path.join("..", "sources", "database.csv")
dataframe = pandas.read_csv(csv_file_path)
print(dataframe)
