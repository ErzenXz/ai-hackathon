import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

data = pd.read_csv('october-data.csv', low_memory=False)
data.iloc[:, 8] = pd.to_numeric(data.iloc[:, 8], errors='coerce')

data.iloc[:, 8] = data.iloc[:, 8].fillna(data.iloc[:, 8].mean())

data = pd.get_dummies(data, columns=['manufacturer', 'device_type', 'clinic'], drop_first=True)

data['is_on_time'] = data['seconds_to_arrive'].apply(lambda x: 1 if x <= 120 else 0)

X = data.drop(columns=['session_date', 'arrive_date', 'seconds_to_arrive', 'is_on_time'])
y = data['is_on_time'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'model2.joblib')

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
