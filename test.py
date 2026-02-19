import mlflow
print('Printing tracking URI scheme below')
print(mlflow.get_tracking_uri())
print('\n')

mlflow.set_tracking_uri('http://127.0.0.1:5000') # local ip at 5000 port

print('Printing NEW tracking URI scheme below')
print(mlflow.get_tracking_uri())
print('\n')