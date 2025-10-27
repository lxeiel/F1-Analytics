# F1-Analytics
## Overview
Streamlit UI to visualise F1 data for drivers and races
## Set Up and Run
### Using Windows .bat file
Run Setup only on the first time
```powershell
setup
```
```powershell
run
```
### Manually Install Dependencies and Run
```powershell
python -m venv venv
pip install -r requirements.txt
```
```powershell
streamlit run app.py
```

### macOS quick start
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```


## Tabs
- Home: Key statistics and trends across seasons.
- Races: Pick a year and circuit, view race results and telemetry heatmaps.
- Drivers: (WIP) Driver-specific analytics.
- Constructors: Team analytics with:
	- Points by Season
	- Wins and Podiums
	- Average Finish and Top 10 Finishes
	- Standings Rank by Season
	- Race Finish Distribution (toggle counts/% per team)
	- Year-over-Year Points Change and head-to-head comparison
