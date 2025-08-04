from dataset import get_data
from engineer_features import engineer_features
import pandas as pd



def train_model(df):
    train_df = df[df['chipTime_42k'].notnull()]

    features = df.drop(columns=['id','chipTime_42k','year']).columns.tolist()
    target = 'chipTime_42k'

    # Train-test split for evaluation
    X = train_df[features]
    y = train_df[target]

    from sklearn.ensemble import RandomForestRegressor

    # Initialize model
    model = RandomForestRegressor(random_state=42)

    # Fit model
    model.fit(X, y)

    # Save model
    import joblib
    joblib.dump(model, "model.pkl")

    ## Save the columns used for training
    joblib.dump(features, "features.pkl")
    joblib.dump(target, "target.pkl")

    pred_df = df[(df['chipTime_42k'].isnull())&(df['year']=='2025')]
    pred_df['chipTime_42k'] = model.predict(pred_df[features])
    pred_df.to_csv("predictions.csv")

import pandas as pd
df = get_data()
df.to_csv('dataset.csv')
df = engineer_features(df)
train_model(df)
