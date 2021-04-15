import requests
new_measurement = {
"sepal_length": 1.2,
"sepal_width": 2.3,
"petal_length": 1.4,
"petal_width": 2.8
}
response = requests.post('http://127.0.0.1:8008/predict', json=new_measurement)
print(response.content)