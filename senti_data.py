import requests
from bs4 import BeautifulSoup
import os

url = "https://www.naaim.org/programs/naaim-exposure-index/"

# Send a GET request to the webpage
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the HTML content of the webpage using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the download link by inspecting the webpage source code or using browser developer tools
    # For example, if the link is inside an anchor tag (a), you can use:
    download_link = soup.find('a', {'href': 'https://www.naaim.org/wp-content/uploads/2024/01/USE_Data-since-Inception_2023-01-03-1.xlsx'})

    # Get the actual download link
    if download_link:
        file_url = download_link.get('href')

        # Download the file
        file_name = os.path.basename(file_url)
        with requests.get(file_url, stream=True) as r:
            with open(file_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"File '{file_name}' downloaded successfully.")
    else:
        print("Download link not found on the webpage.")
else:
    print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
