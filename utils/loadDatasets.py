import pandas as pd
import streamlit

def load_merged_dataset():
    races = pd.read_csv('Dataset/races.csv')
    drivers = pd.read_csv('Dataset/drivers.csv')
    constructors = pd.read_csv('Dataset/constructors.csv')
    results = pd.read_csv('Dataset/results.csv')
    qualifying = pd.read_csv('Dataset/qualifying.csv')
    lap_times = pd.read_csv('Dataset/lap_times.csv')
    pit_stops = pd.read_csv('Dataset/pit_stops.csv')
    driver_standings = pd.read_csv('Dataset/driver_standings.csv')
    circuits = pd.read_csv('Dataset/circuits.csv')

    df_race_circuits_all = pd.merge(races, circuits, on='circuitId', how='left', suffixes=['_race', '_circuits'])
    df_race_circuits_results = pd.merge(results, df_race_circuits_all, on='raceId', how='left', suffixes=['_results', '_raceCircuit'])
    df_race_circuits_results_drivers = pd.merge(df_race_circuits_results, drivers, how='left', on='driverId', suffixes=['', '_drivers'])
    df_race_circuits_results_drivers_constructor = pd.merge(df_race_circuits_results_drivers, constructors, how='left', on='constructorId', suffixes=['', '_constructors'])
    df_race_circuits_results_drivers_constructor.drop(
        columns=[
            'fp1_date',
            'fp1_time',
            'fp2_date',
            'fp2_time',
            'fp3_date',
            'fp3_time',
            # 'quali_date',
            # 'quali_time',
            'sprint_date',
            'sprint_time'
        ], inplace=True
    )
    df_race_circuits_results_drivers_constructor['driver'] = df_race_circuits_results_drivers_constructor['forename'] + " " + df_race_circuits_results_drivers_constructor['surname']
    df_race_circuits_results_drivers_constructor['date'] = pd.to_datetime(df_race_circuits_results_drivers_constructor['date'])

    return df_race_circuits_results_drivers_constructor

def load_recent_races_results(df_race_circuits_results_drivers_constructor):
    recent_race_ids = (
        df_race_circuits_results_drivers_constructor[['raceId', 'date']]
        .drop_duplicates()
        .sort_values('date', ascending=False)
        .head(10)
        ['raceId']
        .values
    )
    list_display_cols = ['driver', 'code', 'name', 'name_race', 'date', 'positionOrder', 'forename', 'surname', 'name', 'points', 'time_results', 'fastestLapTime', 'fastestLapSpeed']
    top_drivers_recent_races = (
    df_race_circuits_results_drivers_constructor[
        df_race_circuits_results_drivers_constructor['raceId'].isin(recent_race_ids)
    ]
    .sort_values(['date', 'raceId', 'positionOrder'], ascending=[False, True, True])
    .groupby('raceId')
    .head(5)
    [list_display_cols]
    )
    return top_drivers_recent_races

    