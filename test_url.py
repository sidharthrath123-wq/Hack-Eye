import requests

url = "http://127.0.0.1:8000/predict"

data = {
    "url": "https://www.google.com"
}

response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response:")
print(response.json())