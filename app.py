import pandas as pd
import joblib

def predict_on_time_from_csv(file_path):
    model = joblib.load('model.joblib')
    
    new_data = pd.read_csv(file_path)
    
    original_columns = new_data[['manufacturer', 'device_type', 'clinic', 'billing_schedule']].copy()
    
    new_data = pd.get_dummies(new_data, columns=['manufacturer', 'device_type', 'clinic'], drop_first=True)
    
    for col in model.feature_names_in_:
        if col not in new_data:
            new_data[col] = 0

    new_data = new_data[model.feature_names_in_]
    predictions = model.predict(new_data)
    new_data['Prediction'] = ["On Time" if pred == 1 else "Not On Time" for pred in predictions]
    
    new_data = pd.concat([original_columns, new_data], axis=1)
    
    return new_data[['manufacturer', 'device_type', 'clinic', 'billing_schedule', 'Prediction']]
    
results = predict_on_time_from_csv('test.csv')
print(results)
