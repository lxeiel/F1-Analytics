import pandas as pd
import streamlit

def load_race_dataset():
	races = pd.read_csv('Dataset/races.csv')
	return races

def load_drivers_dataset():
	drivers = pd.read_csv('Dataset/drivers.csv')
	return drivers

def load_constructors_dataset():
	constructors = pd.read_csv('Dataset/constructors.csv')
	return constructors

def load_results_dataset():
	results = pd.read_csv('Dataset/results.csv')
	return results

def load_qualifying_dataset():
    qualifying = pd.read_csv('Dataset/qualifying.csv')
    return qualifying

def load_lap_times_dataset():
    lap_times = pd.read_csv('Dataset/lap_times.csv')
    return lap_times

def load_pit_stops_dataset():
    pit_stops = pd.read_csv('Dataset/pit_stops.csv')
    return pit_stops

def load_driver_standings_dataset():
    driver_standings = pd.read_csv('Dataset/driver_standings.csv')
    return driver_standings

def load_circuits_dataset():
    circuits = pd.read_csv('Dataset/circuits.csv')
    return circuits

def load_all_datasets():
    races = pd.read_csv('Dataset/races.csv')
    drivers = pd.read_csv('Dataset/drivers.csv')
    constructors = pd.read_csv('Dataset/constructors.csv')
    results = pd.read_csv('Dataset/results.csv')
    qualifying = pd.read_csv('Dataset/qualifying.csv')
    lap_times = pd.read_csv('Dataset/lap_times.csv')
    pit_stops = pd.read_csv('Dataset/pit_stops.csv')
    driver_standings = pd.read_csv('Dataset/driver_standings.csv')
    circuits = pd.read_csv('Dataset/circuits.csv')
    return races, drivers, constructors, results, qualifying, lap_times, pit_stops, driver_standings, circuits