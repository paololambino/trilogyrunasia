import pandas as pd
import joblib

model = joblib.load("model.pkl")

# Example chip times in seconds
sample_input = pd.DataFrame([{
    "chipTime_16k": "00:52:36",   # 1 hr 20 min
    "chipTime_21k": "01:10:28",   # 1 hr 45 min
    "chipTime_32k": "01:54:35",   # 2 hr 40 min
    "gender": "Male",
    "age": "18-24",
    "city": "Manila",
    "chipTime_42k": ""
}])


from engineer_features import engineer_features
import datetime

sample_input = engineer_features(sample_input)

sample_input = sample_input.drop(columns=['chipTime_42k'])
sample_input = sample_input[['gender', 'chipTime_16k', 'city', 'chipTime_21k', 'chipTime_32k', 'age', 'pace_16k', 'pace_21k', 'pace_32k',
     'pace_change_16k_21k', 'pace_change_21k_32k']]

prediction = str(datetime.timedelta(seconds=int(model.predict(sample_input))))

sample_input['chipTime_42k'] = model.predict(sample_input)

print("Predicted 42k pace (sec/km):", prediction)
print (sample_input)
