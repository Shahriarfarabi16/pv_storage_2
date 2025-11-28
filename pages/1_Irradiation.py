import pandas as pd
import requests
import streamlit as st
import altair as alt  # Added for custom charts

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Location & Irradiation",
    page_icon="🌍",
    layout="wide",  # Use wide layout for horizontal arrangement
)

st.title("🌍 Location & Irradiation")
st.markdown(
    """
    Define the building's location to calculate the **Global Tilted Irradiation (GTI)**.
    This value is crucial for all subsequent PV energy production calculations.
    """
)


# --------------------------------------------------------------------------------
# PVGIS Data Fetching Logic (Cached for performance)
# --------------------------------------------------------------------------------

# Helper function to parse time strings from the PVGIS API response
def _parse_pvgis_time(s: pd.Series) -> pd.DatetimeIndex:
    """Robustly parse different timestamp formats from PVGIS."""
    for fmt in ["%Y%m%d:%H%M", None]:  # Try specific format, then fall back to auto-detection
        try:
            ts = pd.to_datetime(s, format=fmt, utc=True, errors="coerce")
            if not ts.isna().all():
                return ts
        except (TypeError, ValueError):
            continue
    # Final fallback for unusual space-separated formats
    s_cleaned = s.astype(str).str.replace(" ", "T", regex=False)
    return pd.to_datetime(s_cleaned, utc=True, errors="coerce")


@st.cache_data(show_spinner="Fetching solar data from PVGIS...")
def fetch_pvgis_gti(lat: float, lon: float) -> dict:
    """
    Fetches hourly solar irradiation data from PVGIS for optimal angles
    and calculates the multi-year mean GTI.
    """
    params = {
        "lat": lat,
        "lon": lon,
        "raddabase": "PVGIS-ERA5",  # Reliable reanalysis data
        "usehorizon": 1,
        "components": 1,
        "optimalangles": 1,  # Let PVGIS calculate the best tilt/azimuth
        "outputformat": "json",
    }
    try:
        resp = requests.get(
            "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc",
            params=params,
            timeout=30
        )
        resp.raise_for_status()  # Raise an exception for bad status codes
        payload = resp.json()
    except requests.exceptions.RequestException as e:
        raise ConnectionError(f"Failed to connect to PVGIS API: {e}")

    hourly_data = (payload.get("outputs", {}) or {}).get("hourly", [])
    if not hourly_data:
        raise ValueError("PVGIS returned no hourly data. Please check the location coordinates.")

    df = pd.DataFrame(hourly_data)
    time_col = next((col for col in ["time", "time(UTC)"] if col in df.columns), df.columns[0])
    df.index = _parse_pvgis_time(df[time_col])
    df = df.sort_index()

    # Calculate Plane-of-Array (POA) irradiance in W/m^2
    poa_components = ["Gb(i)", "Gd(i)", "Gr(i)"]
    if "G(i)" in df.columns:
        poa_irrad = pd.to_numeric(df["G(i)"], errors="coerce").fillna(0.0)
    else:
        poa_irrad = sum(pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0) for c in poa_components)

    # Calculate time step in hours to correctly integrate energy
    dt_h = df.index.to_series().diff().dt.total_seconds().div(3600)
    dt_h.iloc[0] = dt_h.median()  # Fill the first NaN value

    # Integrate to get annual energy (kWh/m^2)
    kwh_per_m2_annual = (poa_irrad * dt_h).groupby(df.index.year).sum() / 1000.0

    return {
        "gti_mean": float(kwh_per_m2_annual.mean()),
        "gti_annual_series": kwh_per_m2_annual,
        "hourly_series": poa_irrad,  # ADDED: Return full hourly data
        "meta": {"api_url": resp.url},
    }


# --------------------------------------------------------------------------------
# User Interface
# --------------------------------------------------------------------------------

# Set default location to your university in Munich for a good starting point
if 'user_location' not in st.session_state:
    st.session_state.user_location = (48.14, 11.57)  # Default: Munich, Germany

# Create two main columns for the input and basic metrics
col1, col2 = st.columns([0.4, 0.6], gap="large")

with col1:
    st.subheader("1. Building Location")

    lat = st.number_input(
        "Latitude (°N)",
        value=st.session_state.user_location[0],
        step=0.0001,
        format="%.4f",
        key="lat_input"
    )
    lon = st.number_input(
        "Longitude (°E)",
        value=st.session_state.user_location[1],
        step=0.0001,
        format="%.4f",
        key="lon_input"
    )

    if st.button("Calculate GTI", type="primary", use_container_width=True):
        try:
            result = fetch_pvgis_gti(lat, lon)

            # --- Store results in session state for other pages ---
            st.session_state.gti_kwh_m2yr = result["gti_mean"]
            st.session_state.user_location = (lat, lon)
            st.session_state.gti_source = "PVGIS-ERA5"
            st.session_state.gti_annual_series = result["gti_annual_series"]
            st.session_state.hourly_series = result["hourly_series"]  # Store hourly data for plotting

            # Clear any old simulation results if location changes
            st.session_state.pop('simulation_results', None)

        except (ConnectionError, ValueError) as e:
            st.error(f"Error: {e}", icon="❌")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}", icon="🔥")

    st.map(pd.DataFrame({'lat': [lat], 'lon': [lon]}), zoom=5, use_container_width=True)

with col2:
    st.subheader("2. Calculation Results")

    if "gti_kwh_m2yr" not in st.session_state:
        st.info("Click 'Calculate GTI' to fetch solar data and see the results here.")
    else:
        gti_value = st.session_state.gti_kwh_m2yr

        st.metric(
            label="☀️ Global Tilted Irradiation (GTI)",
            value=f"{gti_value:,.0f} kWh/m² per year",
            help="This is the total solar energy per square meter that a PV panel with optimal tilt and orientation would receive in an average year."
        )

        st.success(
            f"The GTI for location ({lat:.2f}, {lon:.2f}) has been successfully calculated and stored for the next steps."
        )

        # REMOVED: Year-by-Year bar chart from here

# --------------------------------------------------------------------------------
# Full Width Visualization Section
# --------------------------------------------------------------------------------
if "hourly_series" in st.session_state:
    st.divider()
    st.header("3. Hourly Irradiance Profile")
    st.markdown(
        "Visualize the hourly solar irradiance for a specific year to understand the seasonal and daily variations.")

    # Convert the index to datetime if it isn't already (safety check)
    full_series = st.session_state.hourly_series

    # Get available years from the index
    available_years = sorted(full_series.index.year.unique())

    # --- Hourly Chart ---
    if available_years:
        selected_year_hourly = st.selectbox("Select Year (Hourly)", available_years, index=len(available_years) - 1,
                                            key="year_select_hourly")

        # Filter data for selected year
        year_data_hourly = full_series[full_series.index.year == selected_year_hourly]

        # Prepare data for Altair
        chart_data_hourly = pd.DataFrame({
            "Time": year_data_hourly.index,
            "Irradiance": year_data_hourly.values
        })

        # Create Altair chart
        c_hourly = alt.Chart(chart_data_hourly).mark_bar().encode(
            x='Time',
            y=alt.Y('Irradiance', title='Irradiance (W/m²)')
        ).interactive()  # Added interactive() for zoom/pan
        st.altair_chart(c_hourly, use_container_width=True)

        st.caption(f"Hourly Global Tilted Irradiance (GTI) for {selected_year_hourly}.")
    else:
        st.warning("No yearly data available to display.")

    # --- Daily Chart ---
    st.divider()
    st.header("4. Daily Irradiance Profile")
    st.markdown(
        "Visualize the daily aggregated solar irradiation. This helps in understanding the energy potential on a day-to-day basis.")

    if available_years:
        selected_year_daily = st.selectbox("Select Year (Daily)", available_years, index=len(available_years) - 1,
                                           key="year_select_daily")

        # Filter data for selected year
        year_data_daily = full_series[full_series.index.year == selected_year_daily]

        # Resample to Daily Sum (Wh/m² -> kWh/m²)
        daily_sum = year_data_daily.resample('D').sum() / 1000.0

        # Prepare data for Altair
        chart_data_daily = pd.DataFrame({
            "Date": daily_sum.index,
            "Energy": daily_sum.values
        })

        # Create Altair chart
        c_daily = alt.Chart(chart_data_daily).mark_bar().encode(
            x='Date',
            y=alt.Y('Energy', title='Daily Irradiation (kWh/m²)')
        ).interactive()  # Added interactive() for zoom/pan
        st.altair_chart(c_daily, use_container_width=True)

        st.caption(f"Daily Total Irradiation for {selected_year_daily}.")

    # --- Monthly Chart ---
    st.divider()
    st.header("5. Monthly Irradiance Profile")
    st.markdown("Visualize the monthly aggregated solar irradiation to identify seasonal trends.")

    if available_years:
        selected_year_monthly = st.selectbox("Select Year (Monthly)", available_years, index=len(available_years) - 1,
                                             key="year_select_monthly")

        # Filter data for selected year
        year_data_monthly = full_series[full_series.index.year == selected_year_monthly]

        # Resample to Monthly Sum (Wh/m² -> kWh/m²)
        monthly_sum = year_data_monthly.resample('M').sum() / 1000.0

        # Prepare data for Altair
        chart_data_monthly = pd.DataFrame({
            "Date": monthly_sum.index,
            "Energy": monthly_sum.values
        })

        # Create Altair chart
        # We use 'month(Date):O' to treat month as an Ordinal variable, which makes the bars thick and discrete.
        # axis=alt.Axis(format='%B') ensures full month names (January, February...)
        c_monthly = alt.Chart(chart_data_monthly).mark_bar().encode(
            x=alt.X('Date', timeUnit='month', type='ordinal', axis=alt.Axis(format='%B', title='Month', labelAngle=0)),
            y=alt.Y('Energy', title='Monthly Irradiation (kWh/m²)')
        )
        st.altair_chart(c_monthly, use_container_width=True)

        st.caption(f"Monthly Total Irradiation for {selected_year_monthly}.")