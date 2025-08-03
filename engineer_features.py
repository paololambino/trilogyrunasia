import pandas as pd
def engineer_features(df):

    # fixing formatting of 16k time
    df['chipTime_16k'] = df['chipTime_16k'].apply(
        lambda x: f"00:{x}" if isinstance(x, str) and len(x.split(':')) == 2 else x)

    # formatting chip times to seconds
    chip_cols = ['chipTime_16k', 'chipTime_21k', 'chipTime_32k', 'chipTime_42k']
    df[chip_cols] = df[chip_cols].apply(pd.to_timedelta, errors='coerce')
    df[chip_cols] = df[chip_cols].apply(lambda col: col.dt.total_seconds())

    # calculating the paces
    df['pace_16k'] = df['chipTime_16k'] / 16
    df['pace_21k'] = df['chipTime_21k'] / 21.1
    df['pace_32k'] = df['chipTime_32k'] / 32

    # calculating pace changes between legs
    df['pace_change_16k_21k'] = df['pace_16k'] - df['pace_21k']
    df['pace_change_21k_32k'] = df['pace_21k'] - df['pace_32k']

    df = pd.get_dummies(df, columns=['city', 'gender', 'age'], prefix=['city', 'gender', 'age'], drop_first=False)

    return df

#from dataset import get_data

#df = get_data()
#engineer_features(df).to_csv('feature_data.csv')