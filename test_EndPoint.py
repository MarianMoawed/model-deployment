import requests

# Set the URL of the Flask endpoint
url = 'http://127.0.0.1:5000/predict'  # Replace with the appropriate URL

# Set the path to the image file you want to send
image_path = r'C:\Users\DELL\Downloads\gp_model\AppleCedarRust1.JPG'  # Replace with the actual image file path

# Create a POST request with the image file
files = {'file': open(image_path, 'rb')}
response = requests.post(url, files=files)

# Check the response status code
if response.status_code == 200:
    # Print the JSON response
    print(response.json())
else:
    print('Error:', response)