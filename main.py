import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import altair as alt

from engineer_features import engineer_features

# Load model
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
target = joblib.load("target.pkl")

st.image("trilogyrunbanner.png", use_container_width=True)

st.title("Trilogy Run Asia Marathon Time Predictor")

col1, col2 = st.columns(2)
with col1:

    # Gender and age inputs
    gender = st.selectbox("Gender", ["Male", "Female"], index=None, placeholder="Select Gender")
    age_group = st.selectbox("Age Group", ['18-24',
                                        '25-29',
                                        '30-34',
                                        '35-39',
                                        '40-44',
                                        '45-49',
                                        '50-54',
                                        '55-59',
                                        '60-64',
                                        '65-69',
                                        '70-74',
                                        '75-79'], index=None, placeholder="Select Age Group")

    city = st.selectbox("City", ["Manila",
                                 'Cagayan de Oro',
                                 'Iloilo-Bacolod',
                                 'Baguio',
                                 'Cebu',
                                 'Davao'], index=None, placeholder="Select City")



# Helper to convert HH:MM:SS → total seconds
def get_chip_time(label):
    user_input = st.text_input(f"{label} Chip Time (HH:MM:SS)", value="00:00:00")
    return user_input

with col2:
    #st.markdown("Enter your chip times below (HH:MM:SS format):")

# Get chip times (in seconds)
    chip_16k = get_chip_time("16K - Leg 1")
    chip_21k = get_chip_time("21K - Leg 2")
    chip_32k = get_chip_time("32K - Leg 3")



# Only predict if all chip times are valid
if st.button("Predict Marathon Time"):
    invalid_fields = []
    if gender is None:
        invalid_fields.append("Gender")
    if age_group is None:
        invalid_fields.append("Age Group")
    if city is None:
        invalid_fields.append("City")
    if chip_16k == "00:00:00":
        invalid_fields.append("16K Time")
    if chip_21k == "00:00:00":
        invalid_fields.append("21K Time")
    if chip_32k == "00:00:00":
        invalid_fields.append("32K Time")

    if invalid_fields:
        st.error("Please complete the following fields: " + ", ".join(invalid_fields))

    else:
        input_df = pd.DataFrame([{
            "chipTime_16k": chip_16k,
            "chipTime_21k": chip_21k,
            "chipTime_32k": chip_32k,
            "gender": gender,
            "age": age_group,
            "city": city,
            "chipTime_42k": ""
        }])
        input_df = engineer_features(input_df)
        input_df = input_df.drop(columns=['chipTime_42k'])
        input_df = input_df.reindex(columns=features, fill_value=0)
        input_df = input_df[features]
        input_df['chipTime_42k'] = model.predict(input_df)

        prediction = input_df['chipTime_42k'][0]
        predicted_time = str(datetime.timedelta(seconds=int(prediction)))

        prediction_df = pd.read_csv("predictions.csv")
        prediction_df = pd.concat([prediction_df,input_df])
        prediction_df = prediction_df.reset_index(drop=True).reset_index().rename(columns={'index': 'runner_id'}).fillna(0)


        def time_bin(seconds):
            if pd.isnull(seconds):
                return "Unknown"

            total_minutes = seconds / 60  # convert to minutes

            # Define bins: from 150 mins (2:30) to 480 mins (8:00) in 10-minute steps
            bins = list(np.arange(150, 491, 10))  # end at 490 to include up to 8:00
            labels = []

            for i in range(len(bins) - 1):
                start_hr = bins[i] // 60
                start_min = bins[i] % 60
                end_hr = (bins[i + 1] - 1) // 60
                end_min = (bins[i + 1] - 1) % 60  # subtract 1 to make end inclusive of the last minute

                labels.append(f"{int(start_hr)}:{int(start_min):02d} - {int(end_hr)}:{int(end_min):02d}")

            binned = pd.cut([total_minutes], bins=bins, labels=labels, right=False)[0]
            return binned if pd.notnull(binned) else "8:10+"


        prediction_df['Time'] = prediction_df['chipTime_42k'].apply(time_bin)
        predicted_runner_id = prediction_df['runner_id'].iloc[-1]
        predicted_runner_time = prediction_df['Time'].iloc[-1]


        all_runners_chart = prediction_df.groupby('Time')['runner_id'].nunique().reset_index(name = 'Total')

        # Get city one-hot columns
        city_cols = [col for col in prediction_df.columns if col.startswith('city_')]
        gender_cols = [col for col in prediction_df.columns if col.startswith('gender_')]

        # Get the target city column where it's 1 for the given runner_id
        target_city_col = prediction_df.loc[prediction_df['runner_id'] == predicted_runner_id, city_cols].eq(1).idxmax(axis=1).values[0]
        target_gender_col = prediction_df.loc[prediction_df['runner_id'] == predicted_runner_id, gender_cols].eq(1).idxmax(axis=1).values[0]

        # Filter all rows where that same city column is 1
        city_runners_chart = prediction_df[prediction_df[target_city_col] == True].groupby('Time')['runner_id'].nunique().reset_index(name = 'Total')
        gender_runners_chart = prediction_df[prediction_df[target_gender_col] == True].groupby('Time')['runner_id'].nunique().reset_index(name = 'Total')


        color = alt.condition(
            alt.datum.Time == predicted_runner_time,
            alt.value("#F17036"),
            alt.value("#0992D8") #1F2353
        )
        cornerRadiusEnd = 15

        st.markdown(
            f"""
            <div style='background-color:#d4edda; padding:15px; border-radius:5px; color:#155724;'>
                <span style='font-size:22px; font-weight:normal;'>
                    🎉 Predicted Marathon Time: <span style='font-size:20px; font-weight:bold;'> {predicted_time:}</span>
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        prediction_df = prediction_df.sort_values(by='chipTime_42k')
        prediction_df["percentile"] = prediction_df["chipTime_42k"].rank(pct=True) * 100
        percentile = prediction_df.loc[prediction_df["runner_id"] == predicted_runner_id, "percentile"].values[0].round(2)
        total_runners = prediction_df['runner_id'].nunique()
        total_rank =  prediction_df[prediction_df['percentile']<= percentile]['runner_id'].nunique()
        percentile = 100 - percentile
        st.markdown(
            f"""
            You ran faster than  <span style='font-size:20px; font-weight:bold;'> {percentile:.2f}%</span> of all runners.
            """,
            unsafe_allow_html=True,
            help = f'You place {total_rank} out of {total_runners}.'
        )

        c = alt.Chart(all_runners_chart).mark_bar(cornerRadiusEnd = cornerRadiusEnd).encode(
            x=alt.X("Time:N", axis=alt.Axis(title="Predicted Finish Time of All Runners", labelAngle=-60)),
            y=alt.Y("Total:Q", axis=alt.Axis(title="Number of Runners")),
            color=color
        )
        st.altair_chart(c)

        gender_df = prediction_df[prediction_df[target_gender_col] == True]
        gender_df["percentile"] = gender_df["chipTime_42k"].rank(pct=True) * 100
        percentile = gender_df.loc[gender_df["runner_id"] == predicted_runner_id, "percentile"].values[0].round(2)
        total_runners = gender_df['runner_id'].nunique()
        total_rank =  gender_df[gender_df['percentile']<= percentile]['runner_id'].nunique()
        percentile = 100 - percentile

        st.markdown(
            f"""
            You ran faster than <span style='font-size:20px; font-weight:bold;'> {percentile:.2f}%</span> of all {gender} runners.
            """,
            unsafe_allow_html=True,
            help = f'You place {total_rank} out of {total_runners}.'
        )

        c2 = alt.Chart(gender_runners_chart).mark_bar(cornerRadiusEnd = cornerRadiusEnd).encode(
            x=alt.X("Time:N", axis=alt.Axis(title=f"Predicted Finish Time of  {gender} Runners", labelAngle=-60)),
            y=alt.Y("Total:Q", axis=alt.Axis(title="Number of Runners")),
            color=color
        )
        st.altair_chart(c2)

        city_df = prediction_df[prediction_df[target_city_col] == True]
        city_df["percentile"] = city_df["chipTime_42k"].rank(pct=True) * 100
        percentile = city_df.loc[city_df["runner_id"] == predicted_runner_id, "percentile"].values[0].round(2)
        total_runners = city_df['runner_id'].nunique()
        total_rank =  city_df[city_df['percentile']<= percentile]['runner_id'].nunique()
        percentile = 100 - percentile

        st.markdown(
            f"""
            You ran faster than <span style='font-size:20px; font-weight:bold;'> {percentile:.2f}%</span> of all {city} runners.
            """,
            unsafe_allow_html=True,
            help = f'You place {total_rank} out of {total_runners}.'
        )
        c3 = alt.Chart(city_runners_chart).mark_bar(cornerRadiusEnd = cornerRadiusEnd).encode(
            x=alt.X("Time:N", axis=alt.Axis(title=f"Predicted Finish Time of {city} Runners", labelAngle=-60)),
            y=alt.Y("Total:Q", axis=alt.Axis(title="Number of Runners")),
            color=color
        )
        if city in ['Manila', 'Cagayan de Oro']:
            st.altair_chart(c3)
        else:
            pass

