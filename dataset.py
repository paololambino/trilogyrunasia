import pandas as pd
import requests

def get_data():
    # Base URL for fetching race results
    base_url = "https://results.raceroster.com/v2/api/result-events/{}/sub-events/{}/results?start={}&limit=1000&locale=en-US"

    # website: https://raceroster.com/search?q=trilogy%20run%20asia&t=past

    race_list = {
        "Manila": {
            "2024": {
                "16k": {"eventId": "73132", "raceId": "193987"},
                "21k": {"eventId": "73132", "raceId": "201025"},
                "32k": {"eventId": "73132", "raceId": "207624"}
            },
            "2025": {
                "16k": {"eventId": "85809", "raceId": "223772"},
                "21k": {"eventId": "87034", "raceId": "228307"},
                "32k": {"eventId": "87040", "raceId": "234242"}
            }
        },
        "Cagayan de Oro": {
            "2024": {
                "16k": {"eventId": "73282", "raceId": "198168"},
                "21k": {"eventId": "73282", "raceId": "212292"},
                "32k": {"eventId": "73282", "raceId": "217259"}
            },
            "2025": {
                "16k": {"eventId": "85811", "raceId": "223222"},
                "21k": {"eventId": "86868", "raceId": "227059"},
                "32k": {"eventId": "86869", "raceId": "235760"}
            }
        },
        "Iloilo-Bacolod": {
            "2024": {
                "16k": {"eventId": "73281", "raceId": "200756"},
                "21k": {"eventId": "73281", "raceId": "206891"},
                "32k": {"eventId": "73281", "raceId": "215761"}
            }
        },
        "Baguio": {
            "2024": {
                "16k": {"eventId": "73218", "raceId": "196797"},
                "21k": {"eventId": "73218", "raceId": "205293"},
                "32k": {"eventId": "73218", "raceId": "217815"}
            }
        },
        "Cebu": {
            "2024": {
                "16k": {"eventId": "73278", "raceId": "194448"},
                "21k": {"eventId": "73278", "raceId": "202195"},
                "32k": {"eventId": "73278", "raceId": "208879"}
            }
        },
        "Davao": {
            "2024": {
                "16k": {"eventId": "73284", "raceId": "203060"},
                "21k": {"eventId": "73284", "raceId": "209473"},
                "32k": {"eventId": "73284", "raceId": "216533"}
            }
        },
        "National Finals": {
            "2024": {
                "42k": {"eventId": "81614", "raceId": "219947"}
            }
        }
    }

    # Define the race order
    race_distances = ["16k", "21k", "32k", "42k"]
    race_dfs = {}

    for race in race_distances:
        df_race = pd.DataFrame()

        for city, years in race_list.items():
            for year, races in years.items():
                if race not in races:
                    continue

                event_id = races[race]["eventId"]
                race_id = races[race]["raceId"]

                start = 0
                race_data = []

                while True:
                    response = requests.get(base_url.format(event_id, race_id, start))
                    data = response.json().get('data', [])
                    if not data:
                        break
                    race_data.extend(data)
                    start += 1000

                if race_data:
                    race_df = pd.DataFrame.from_records(race_data)
                    if 'gender' in race_df.columns and 'genderSexId' not in race_df.columns:
                        race_df['genderSexId'] = race_df['gender']
                    race_df = race_df[['name', 'genderSexId', 'chipTime', 'division']]
                    race_df["city"] = city
                    race_df["year"] = year
                    race_df = race_df[race_df['chipTime'] != '']

                    race_df.rename(columns={
                        'chipTime': f'chipTime_{race}', }, inplace=True)

                    race_df.drop(columns=["race"], inplace=True, errors="ignore")

                    df_race = pd.concat([df_race, race_df], ignore_index=True)

        race_dfs[race] = df_race
    # ✅ Assign DataFrames to variables
    df_16k = race_dfs.get("16k", pd.DataFrame())
    df_21k = race_dfs.get("21k", pd.DataFrame())
    df_32k = race_dfs.get("32k", pd.DataFrame())
    df_42k = race_dfs.get("42k", pd.DataFrame())

    # ✅ Merge them using left joins (16k → 21k → 32k)
    df_final = df_16k.merge(df_21k.drop(columns='division'), on=['name', 'genderSexId', 'year', 'city'], how='left',
                            suffixes=('', '_21k'))
    df_final = df_final.merge(df_32k.drop(columns='division'), on=['name', 'genderSexId', 'year', 'city'], how='left',
                              suffixes=('', '_32k'))
    df_final = df_final.merge(df_42k.drop(columns='division'), on=['name', 'genderSexId', 'year'], how='left',
                              suffixes=('', '_42k'))

    # ✅ Drop duplicate columns caused by merging (if any)
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]
    df_final = df_final.drop(columns='city_42k')

    # ✅ Drop any remaining duplicate rows
    df_final.drop_duplicates(inplace=True)

    # Display Final Merged DataFrame
    df_final = df_final[~df_final['name'].str.contains('Verification')]
    df_final = df_final[(df_final['chipTime_16k'].notnull()) &
                        (df_final['chipTime_21k'].notnull()) &
                        (df_final['chipTime_32k'].notnull())]
    df_final = df_final.sort_values(by=['chipTime_32k', 'chipTime_21k', 'chipTime_16k']).groupby('name').head(1)  # grouping by the runner
    df_final = df_final.reset_index().rename(columns={"index": "id", 'genderSexId': 'gender'})  # renaming columns
    df_final = df_final[~df_final['division'].isin(['17 and Below Male','19 Below Male', '70+ Male'])]
    df_final = df_final[df_final['division'].notnull()]
    df_final['age'] = df_final['division'].str[:5]
    df_final['age'] = df_final['age'].replace({'20-24': '18-24'})


    df_final = df_final.drop(columns=['name','division'])  # dropping names of runners for privacy

    return df_final
#get_data().to_csv('dataset.csv')