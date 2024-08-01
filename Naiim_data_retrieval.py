import pandas as pd
import requests
from datetime import datetime, timedelta

def check_link(url):
    try:
        response = requests.head(url)
        # Check if the status code is in the range of 2xx, indicating success
        if 200 <= response.status_code < 300:
            return True
        else:
            return False
    except requests.ConnectionError:
        return False


str1 = "https://www.naaim.org/wp-content/uploads/"
str3 = "/USE_Data-since-Inception_"

# Get today's date
today_date = datetime.now().date()

# Print today's date
print("Today's date:", today_date)

# Calculate and print the dates of the previous 10 days
for i in range(0, 10):
    previous_date = today_date - timedelta(days=i)
    print(f"Previous {i} days ago:", previous_date)
    str2 = str(previous_date)[:4] + "/" + str(previous_date)[5:7]
    str4 = str(previous_date)[:4] + "-" + str(previous_date)[5:7] + "-" + str(previous_date)[8:10] + ".xlsx"
    url = str1 + str2 + str3 + str4
    print(url)
    if(check_link(url)):
        # Replace 'your_excel_file.xls' with the actual file path or URL of your Excel file
        excel_file_path = url

        # Load the Excel file into a DataFrame
        excel_data = pd.read_excel(excel_file_path)

        # Replace 'your_csv_file.csv' with the desired CSV file path
        csv_file_path = 'naiim.csv'

        # Save the DataFrame to a CSV file
        excel_data.to_csv(csv_file_path, index=False)

        # Read the CSV file into a new DataFrame
        csv_data = pd.read_csv(csv_file_path)

        # Now you can work with the DataFrame as needed
        print(csv_data.head())
        break
