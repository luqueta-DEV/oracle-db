import cx_Oracle
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

username = 'luqueta-DEV'
password = '69'
host = 'localhost'
port = 1521
sid = 'orc1'

dsn = cx_Oracle.makedsn(host, port, sid)
connection = cx_Oracle.connect(username, password, dsn)
cursor = connection.cursor()

query = "SELECT * FROM tabela_exemplo"
df = pd.read_sql(query, con=connection)

X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential([
    layers.InputLayer(input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

loss, mae = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Mean Absolute Error: {mae}")

model.save("modelo_tensorflow.h5")

cursor.close()
connection.close()
