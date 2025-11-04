import streamlit as st
from tabs.raceStats import raceStatsTab
from tabs.driverStats import driverStatsTab
# Import the optional driverComparison tab defensively so a failure there doesn't break the whole app
try:
    from tabs.driverComparison import driverComparisonTab
    _driver_comparison_error = None
except Exception as _e:  # pragma: no cover - runtime/import guard
    driverComparisonTab = None
    _driver_comparison_error = _e
from tabs.overallStats import homeTab
from tabs.constructorStats import constructorStatsTab
from tabs.weatherAnalysis import weatherAnalysisTab
from tabs.forecasting import forecastingTab
from tabs.tireAnalysis import tireAnalysisTab

st.set_page_config(
    page_title="F1 Dashboard",
    page_icon="🏎️",
    layout="wide"
)

st.title("🏁 F1 Dashboard")

available_tabs = [
    ("Home", homeTab),
    ("Races", raceStatsTab),
    ("Drivers", driverStatsTab),
]

# Add driver comparison tab if imported successfully; otherwise add a placeholder that shows the import error
if driverComparisonTab is not None:
    available_tabs.append(("Driver Comparison", driverComparisonTab))
else:
    def _driver_comp_placeholder():
        st.error(f"Driver Comparison tab failed to load: {_driver_comparison_error}")
    available_tabs.append(("Driver Comparison (unavailable)", _driver_comp_placeholder))

available_tabs.extend([
    ("Constructors", constructorStatsTab),
    ("Forecasting", forecastingTab),
    ("Weather Analysis", weatherAnalysisTab),
    ("Tire Analysis", tireAnalysisTab),
])

tabs = st.tabs([label for label, _ in available_tabs])

for i, (_, fn) in enumerate(available_tabs):
    with tabs[i]:
        try:
            fn()
        except Exception as e:  # show errors inside the tab rather than crashing the whole app
            st.exception(e)

