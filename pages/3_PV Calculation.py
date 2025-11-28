import math
import numpy as np
import pandas as pd
import requests
import streamlit as st
import altair as alt  # <-- IMPORTED ALTAIR

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(page_title="PV Simulation", page_icon="🔆", layout="wide")  # Renamed title
st.title("🔆 PV Simulation (With & Without Battery)")  # Renamed title
st.markdown(
    "Simulate the performance of a PV system with a battery, and "
    "see how it improves your energy balance and electricity bill."
)

# --------------------------------------------------------------------------------
# Initialization & Defaults
# --------------------------------------------------------------------------------
# Initialize session state with defaults if data is missing from previous pages

# 1. Location & Irradiation Defaults
if "user_location" not in st.session_state:
    st.session_state.user_location = (48.14, 11.57)  # Default: Munich

if "gti_kwh_m2yr" not in st.session_state:
    st.session_state.gti_kwh_m2yr = 1350.0  # Default value as requested

# 2. Demand Defaults
if "annual_demand" not in st.session_state:
    st.session_state.annual_demand = 4500.0  # Default value as requested

# 3. Demand Profile Default (Required for simulation)
if "generated_8760_profile_df" not in st.session_state:
    # Generate a synthetic flat profile if no detailed profile exists from Page 2
    # This ensures the simulation can run immediately with the default annual demand.
    hours = 8760
    avg_hourly_kwh = st.session_state.annual_demand / hours
    st.session_state.generated_8760_profile_df = pd.DataFrame({
        "Energy Demand (kWh)": [avg_hourly_kwh] * hours
    })


# --------------------------------------------------------------------------------
# Styling Helper Functions
# --------------------------------------------------------------------------------
def highlight_min_cell(s):
    """Highlights the minimum value in a DataFrame with a thick border."""
    # s is the DataFrame. Find the minimum value.
    min_val = s.min(skipna=True).min(skipna=True)
    # Create a boolean mask where True indicates the minimum value(s)
    is_min = (s == min_val)
    # Create a DataFrame of styles, applying the border where is_min is True
    styles = is_min.applymap(lambda x: 'border: 2.5px solid #333;' if x else '')
    return styles


def highlight_max_cell(s):
    """Highlights the maximum value in a DataFrame with a thick border."""
    # s is the DataFrame. Find the maximum value.
    max_val = s.max(skipna=True).max(skipna=True)
    # Create a boolean mask where True indicates the maximum value(s)
    is_max = (s == max_val)
    # Create a DataFrame of styles, applying the border where is_max is True
    styles = is_max.applymap(lambda x: 'border: 2.5px solid #333;' if x else '')
    return styles


# --- NEW: Function to center-align all cells ---
def center_align_cells(s):
    """Returns a DataFrame of styles to center-align text."""
    return pd.DataFrame('text-align: center', index=s.index, columns=s.columns)


# --- NEW: Style for table headers ---
header_style = [
    {'selector': 'th.col_heading', 'props': [('background-color', '#D2B48C'), ('color', 'black')]},  # Light Brown (Tan)
    {'selector': 'th.row_heading', 'props': [('background-color', '#D2B48C'), ('color', 'black')]},
    {'selector': 'th.blank', 'props': [('background-color', '#D2B48C'), ('color', 'black')]}  # Top-left corner
]

# --------------------------------------------------------------------------------
# Helper Functions (PVGIS fetch + Simulation)
# --------------------------------------------------------------------------------
PVGIS_ENDPOINT = "https://re.jrc.ec.europa.eu/api/v5_3/seriescalc"


def _parse_pvgis_time(s: pd.Series) -> pd.DatetimeIndex:
    ts = pd.to_datetime(s, format="%Y%m%d:%H%M", utc=True, errors="coerce")
    if not ts.isna().all(): return ts
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if not ts.isna().all(): return ts
    return pd.to_datetime(s.astype(str).str.replace(" ", "T"), utc=True, errors="coerce")


@st.cache_data(show_spinner="Fetching hourly PV production data...")
def fetch_hourly_pvgis_era5_year(lat: float, lon: float, year: int) -> pd.DataFrame:
    params = {
        "lat": lat, "lon": lon, "raddabase": "PVGIS-ERA5", "usehorizon": 1,
        "components": 1, "optimalangles": 1, "outputformat": "json", "startyear": year, "endyear": year
    }
    resp = requests.get(PVGIS_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    hourly = (payload.get("outputs", {}) or {}).get("hourly", [])
    df = pd.DataFrame(hourly)
    ts = _parse_pvgis_time(df[df.columns[0]].astype(str))
    df.index = ts

    # Ensure exactly 8760 hours
    if len(df) > 8760:
        df = df.iloc[:8760]

    df = df.loc[~df.index.duplicated(keep='first')]

    for c in ["Gb(i)", "Gd(i)", "Gr(i)"]:
        df[c] = pd.to_numeric(df.get(c, 0.0), errors="coerce").fillna(0.0)
    df["G_poa"] = df["Gb(i)"] + df["Gd(i)"] + df["Gr(i)"]
    return df


def _dt_hours(index: pd.DatetimeIndex) -> np.ndarray:
    diffs_h = index.to_series().diff().dt.total_seconds() / 3600.0
    fallback = float(diffs_h[diffs_h > 0].median()) if (diffs_h > 0).any() else 1.0
    return diffs_h.fillna(fallback).where(diffs_h > 0, fallback).values


# --- "No Battery" Simulation Function ---
def simulate_no_batt(df_pv_8760, pv_kwp, pr, load_kwh_8760):
    """
    Runs the simulation using the 8760h PV profile and 8760h load profile.
    """
    dt_h = _dt_hours(df_pv_8760.index)

    # 1. Calculate PV Generation (kWh)
    pv_kwh = pv_kwp * pr * (df_pv_8760["G_poa"].values / 1000.0) * dt_h

    # 2. Load (kWh) is now passed in directly
    load_kwh = np.array(load_kwh_8760)

    # 3. Do the Balance
    imp = np.maximum(0.0, load_kwh - pv_kwh)
    exp = np.maximum(0.0, pv_kwh - load_kwh)

    out = pd.DataFrame({
        "pv_kWh": pv_kwh, "load_kWh": load_kwh,
        "import_kWh": imp, "export_kWh": exp,
    }, index=df_pv_8760.index)

    totals = {
        "pv_production_kWh": out["pv_kWh"].sum(),
        "grid_import_kWh": out["import_kWh"].sum(),
        "export_kWh": out["export_kWh"].sum(),
        "annual_load_kWh": out["load_kWh"].sum(),
    }
    return out, totals


# --- NEW: Battery Simulation Function (SIMPLIFIED) ---
def simulate_with_batt(
        df_pv_8760, pv_kwp, pr, load_kwh_8760,
        batt_kwh_total
):
    """
    Runs the hourly 8760 simulation with a battery system.
    Follows the logic:
    1. PV covers load first.
    2. Surplus PV charges the battery.
    3. Remaining surplus is exported.
    4. Unmet load is covered by battery.
    5. Remaining unmet load is imported.
    ** SIMPLIFIED: Assumes 100% efficiency and no charge/discharge kW limits. **
    """
    dt_h = _dt_hours(df_pv_8760.index)
    if not (dt_h > 0).all():
        dt_h = np.ones_like(dt_h)

    # 1. Calculate PV Generation (kWh)
    pv_kwh = pv_kwp * pr * (df_pv_8760["G_poa"].values / 1000.0) * dt_h
    load_kwh = np.array(load_kwh_8760)

    # Simulation array initialization
    n_hours = 8760
    batt_soc_kwh = np.zeros(n_hours + 1)  # State of Charge (SoC)
    batt_soc_kwh[0] = batt_kwh_total * 0.70  # NEW: Set initial SoC to 70%
    batt_in_kwh = np.zeros(n_hours)  # Energy into battery
    batt_out_kwh = np.zeros(n_hours)  # Energy out of battery
    import_kwh = np.zeros(n_hours)  # Grid import
    export_kwh = np.zeros(n_hours)  # Grid export

    # --- Run the hourly loop ---
    for i in range(n_hours):
        soc_start_kwh = batt_soc_kwh[i]
        pv_gen = pv_kwh[i]
        load = load_kwh[i]

        # 1. PV covers load directly
        pv_to_load = min(pv_gen, load)
        surplus_pv = pv_gen - pv_to_load
        unmet_load = load - pv_to_load

        # 2. Handle Unmet Load (Discharge battery) - Simplified
        batt_to_load = min(
            unmet_load,
            soc_start_kwh
        )
        grid_import = unmet_load - batt_to_load
        import_kwh[i] = grid_import

        # 3. Handle Surplus PV (Charge battery) - Simplified
        batt_room_kwh = batt_kwh_total - (soc_start_kwh - batt_to_load)
        batt_charge_net = min(
            surplus_pv,
            batt_room_kwh
        )
        grid_export = surplus_pv - batt_charge_net  # Export what's left
        export_kwh[i] = grid_export

        # 4. Store intermediate values
        batt_in_kwh[i] = batt_charge_net
        batt_out_kwh[i] = batt_to_load

        # 5. Update SOC for next hour
        batt_soc_kwh[i + 1] = soc_start_kwh - batt_to_load + batt_charge_net

    # --- Compile results ---
    out = pd.DataFrame({
        "pv_kWh": pv_kwh, "load_kWh": load_kwh,
        "import_kWh": import_kwh, "export_kWh": export_kwh,
        "batt_charge_kWh": batt_in_kwh, "batt_discharge_kWh": batt_out_kwh,
        "batt_soc_kwh": batt_soc_kwh[:-1]
    }, index=df_pv_8760.index)

    totals = {
        "pv_production_kWh": out["pv_kWh"].sum(),
        "grid_import_kWh": out["import_kWh"].sum(),
        "export_kWh": out["export_kWh"].sum(),
        "annual_load_kWh": out["load_kWh"].sum(),
    }
    return out, totals


# --- UPDATED: annual_bill function ---
def annual_bill(totals, retail, fit, allow_negative):
    """Calculates the annual bill, optionally flooring at zero."""
    bill = totals["grid_import_kWh"] * retail - totals["export_kWh"] * fit
    if not allow_negative:
        bill = max(0.0, bill)
    return bill


# --- NEW: Helper functions for SCR and SSR ---
def scr(totals):
    """Calculates Self-Consumption Rate (SCR)."""
    pv_gen = totals["pv_production_kWh"]
    if pv_gen <= 1e-9:
        return 0.0  # No production, so 0% consumption
    pv_used_on_site = pv_gen - totals["export_kWh"]
    return pv_used_on_site / pv_gen


def ssr(totals):
    """Calculates Self-Sufficiency Rate (SSR)."""
    load = totals["annual_load_kWh"]
    if load <= 1e-9:
        return 1.0  # No load, so 100% sufficient
    # This is the same calculation as the old Degree of Autonomy
    return 1.0 - totals["grid_import_kWh"] / load


# --- NEW: Helper function for Annuity Factor ---
def calculate_anf(interest_rate_percent, project_lifetime):
    """Calculates the Annuity Factor (ANF)."""
    if interest_rate_percent == 0:
        return 0  # Indicates no financing

    i = interest_rate_percent / 100.0
    n = project_lifetime

    # Check for edge case where (1+i)^n is very large or 1
    try:
        numerator = ((1 + i) ** n) * i
        denominator = ((1 + i) ** n) - 1
        if denominator == 0:
            return 0  # Should not happen if i > 0, but good to check
        return numerator / denominator
    except OverflowError:
        return i  # Approximation for very large n


# --- NEW: Helper function for monthly stacked bar chart ---
def create_monthly_coverage_chart(df_sim_hourly: pd.DataFrame, title: str) -> alt.Chart:
    """
    Creates a monthly stacked bar chart showing the composition of energy demand
    (how much was met by PV vs. Grid).
    """
    # 1. Resample to monthly totals
    df_monthly = df_sim_hourly.resample('MS').sum()  # 'MS' = Month Start

    # 2. Calculate "PV Energy Used" (Self-Consumption)
    # This is the PV energy that was *not* exported.
    df_monthly['pv_used_kWh'] = df_monthly['pv_kWh'] - df_monthly['export_kWh']

    # 3. Select only the columns we need for the stacked bar
    # The total height (pv_used_kWh + import_kWh) will equal the total load.
    df_plot = df_monthly[['import_kWh', 'pv_used_kWh']]

    # 4. Reset index, explicitly naming the new column 'Month'
    df_plot_reset = df_plot.reset_index(names='Month')

    # 5. "Melt" the DataFrame from wide to long format for Altair
    df_plot_long = df_plot_reset.melt(
        id_vars='Month',  # Use the new explicit column name
        value_vars=['import_kWh', 'pv_used_kWh'],
        var_name='Source',
        value_name='Energy (kWh)'
    )

    # 6. Create custom labels and colors
    # (Removed unused source_domain and source_range)

    # Map internal names to pretty names for the legend
    df_plot_long['Source'] = df_plot_long['Source'].map({
        'import_kWh': 'Grid Import',
        'pv_used_kWh': 'PV Energy Used'
    })

    # 7. Build the chart
    chart = alt.Chart(df_plot_long).mark_bar().encode(
        # Set Month on the X-axis, formatted as 'Jan', 'Feb', etc.
        x=alt.X('Month:T',
                axis=alt.Axis(title='Month', format='%b'),
                sort=alt.SortField("Month", order="ascending")  # Explicitly sort Jan -> Dec
                ),

        # Set Energy on the Y-axis
        y=alt.Y('Energy (kWh):Q', title='Monthly Energy Demand (kWh)'),

        # Define the colors for the stacks
        color=alt.Color(
            'Source:N',
            legend=alt.Legend(title="Energy Source"),
            scale=alt.Scale(
                domain=['Grid Import', 'PV Energy Used'],
                range=['#ADD8E6', '#FFD700']  # Light Blue, Gold
            )
        ),

        # Add tooltips
        tooltip=[
            alt.Tooltip('Month:T', format='%B %Y'),
            'Source:N',
            alt.Tooltip('Energy (kWh):Q', format=',.0f')
        ]
    ).properties(
        title=title
    ).interactive()

    return chart


# --- NEW: Helper function for weekly daily stacked bar chart ---
def create_weekly_daily_coverage_chart(df_sim_hourly: pd.DataFrame, week_number: int, title: str) -> alt.Chart:
    """
    Creates a daily stacked bar chart for a specific week showing energy demand coverage.
    Week 1 is days 0-6, Week 2 is days 7-13, etc.
    """
    # 1. Determine start and end indices
    # Each week is 7 days * 24 hours
    hours_per_week = 7 * 24
    start_loc = (week_number - 1) * hours_per_week
    end_loc = start_loc + hours_per_week

    # Handle end of year bounds
    if start_loc >= len(df_sim_hourly):
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data"))

    if end_loc > len(df_sim_hourly):
        end_loc = len(df_sim_hourly)

    # 2. Slice the dataframe
    df_week = df_sim_hourly.iloc[start_loc:end_loc].copy()

    # 3. Resample to daily totals
    df_daily = df_week.resample('D').sum()

    # 4. Calculate PV Used
    df_daily['pv_used_kWh'] = df_daily['pv_kWh'] - df_daily['export_kWh']

    # 5. Prepare for Altair
    df_plot = df_daily[['import_kWh', 'pv_used_kWh']].reset_index(names='Date')

    # Melt
    df_plot_long = df_plot.melt(
        id_vars='Date',
        value_vars=['import_kWh', 'pv_used_kWh'],
        var_name='Source',
        value_name='Energy (kWh)'
    )

    df_plot_long['Source'] = df_plot_long['Source'].map({
        'import_kWh': 'Grid Import',
        'pv_used_kWh': 'PV Energy Used'
    })

    # 6. Chart
    chart = alt.Chart(df_plot_long).mark_bar().encode(
        # Updated axis format to date only
        x=alt.X('Date:T', axis=alt.Axis(format='%d %b', title='Date')),
        y=alt.Y('Energy (kWh):Q', title='Daily Energy (kWh)'),
        color=alt.Color(
            'Source:N',
            legend=alt.Legend(title="Energy Source"),
            scale=alt.Scale(
                domain=['Grid Import', 'PV Energy Used'],
                range=['#ADD8E6', '#FFD700']  # Light Blue, Gold
            )
        ),
        tooltip=[
            alt.Tooltip('Date:T', format='%A, %d %B'),
            'Source:N',
            alt.Tooltip('Energy (kWh):Q', format=',.1f')
        ]
    ).properties(
        title=f"{title} (Week {week_number})"
    ).interactive()

    return chart


# --- 9. NEW: Helper function for PV Energy Use chart ---
def create_weekly_daily_pv_use_chart(df_sim_hourly: pd.DataFrame, week_number: int, title: str) -> alt.Chart:
    """
    Creates a daily stacked bar chart for a specific week showing PV energy use (on-site vs export).
    """
    hours_per_week = 7 * 24
    start_loc = (week_number - 1) * hours_per_week
    end_loc = start_loc + hours_per_week

    if start_loc >= len(df_sim_hourly):
        return alt.Chart(pd.DataFrame()).mark_text().encode(text=alt.value("No Data"))

    if end_loc > len(df_sim_hourly):
        end_loc = len(df_sim_hourly)

    df_week = df_sim_hourly.iloc[start_loc:end_loc].copy()
    df_daily = df_week.resample('D').sum()
    df_daily['pv_used_kWh'] = df_daily['pv_kWh'] - df_daily['export_kWh']

    df_plot = df_daily[['pv_used_kWh', 'export_kWh']].reset_index(names='Date')

    df_plot_long = df_plot.melt(
        id_vars='Date',
        value_vars=['pv_used_kWh', 'export_kWh'],
        var_name='Use',
        value_name='Energy (kWh)'
    )

    df_plot_long['Use'] = df_plot_long['Use'].map({
        'pv_used_kWh': 'PV Used On-Site',
        'export_kWh': 'Grid Feed-In (Export)'
    })

    sort_order = ['PV Used On-Site', 'Grid Feed-In (Export)']

    chart = alt.Chart(df_plot_long).mark_bar().encode(
        # Updated axis format to date only
        x=alt.X('Date:T', axis=alt.Axis(format='%d %b', title='Date')),
        y=alt.Y('Energy (kWh):Q', title='Daily Energy (kWh)'),
        color=alt.Color(
            'Use:N',
            legend=alt.Legend(title="PV Energy Use"),
            scale=alt.Scale(
                domain=sort_order,
                range=['#FFD700', '#90EE90']
            )
        ),
        order=alt.Order("Use", sort="ascending"),
        tooltip=[
            alt.Tooltip('Date:T', format='%A, %d %B'),
            'Use:N',
            alt.Tooltip('Energy (kWh):Q', format=',.1f')
        ]
    ).properties(
        title=f"{title} (Week {week_number})"
    ).interactive()

    return chart


def create_monthly_pv_use_chart(df_sim_hourly: pd.DataFrame, title: str) -> alt.Chart:
    """
    Creates a monthly stacked bar chart showing the use of PV energy
    (how much was used on-site vs. exported).
    """
    # 1. Resample to monthly totals
    df_monthly = df_sim_hourly.resample('MS').sum()  # 'MS' = Month Start

    # 2. Calculate "PV Energy Used" (Self-Consumption)
    df_monthly['pv_used_kWh'] = df_monthly['pv_kWh'] - df_monthly['export_kWh']

    # 3. Select only the columns we need for the stacked bar
    # The total height (pv_used_kWh + export_kWh) will equal the total PV generation.
    df_plot = df_monthly[['pv_used_kWh', 'export_kWh']]

    # 4. Reset index, explicitly naming the new column 'Month'
    df_plot_reset = df_plot.reset_index(names='Month')

    # 5. "Melt" the DataFrame from wide to long format for Altair
    df_plot_long = df_plot_reset.melt(
        id_vars='Month',
        value_vars=['pv_used_kWh', 'export_kWh'],
        var_name='Use',
        value_name='Energy (kWh)'
    )

    # 6. Map internal names to pretty names for the legend
    df_plot_long['Use'] = df_plot_long['Use'].map({
        'pv_used_kWh': 'PV Used On-Site',
        'export_kWh': 'Grid Feed-In (Export)'
    })

    # 7. Define the stacking order (PV Used On-Site at the bottom)
    sort_order = ['PV Used On-Site', 'Grid Feed-In (Export)']

    # 8. Build the chart
    chart = alt.Chart(df_plot_long).mark_bar().encode(
        # Set Month on the X-axis
        x=alt.X('Month:T',
                axis=alt.Axis(title='Month', format='%b'),
                sort=alt.SortField("Month", order="ascending")  # Explicitly sort Jan -> Dec
                ),

        # Set Energy on the Y-axis
        y=alt.Y('Energy (kWh):Q', title='Monthly PV Generation (kWh)'),

        # Define the colors for the stacks
        color=alt.Color(
            'Use:N',
            legend=alt.Legend(title="PV Energy Use"),
            scale=alt.Scale(
                domain=sort_order,  # Use the sort_order list for domain
                range=['#FFD700', '#90EE90']  # Yellow, Light Green
            )
        ),

        # --- FIXED: Set stacking order ---
        order=alt.Order("Use", sort="ascending"),

        # Add tooltips
        tooltip=[
            alt.Tooltip('Month:T', format='%B %Y'),
            'Use:N',
            alt.Tooltip('Energy (kWh):Q', format=',.0f')
        ]
    ).properties(
        title=title
    ).interactive()

    return chart


# --- 10. NEW: Helper function for LCOE Sensitivity Table ---
# --- REMOVED @st.cache_data ---
def calculate_lcoe_sensitivity(
        _df_hourly, _pr, _load_kwh_8760,  # Simulation inputs
        pv_axis_labels, batt_axis_labels,  # The axes for the table
        _pv_capex, _batt_capex, _opex_rate, _lifetime,  # Economic inputs
        _interest_rate, _retail, _fit,
        _allow_negative_bill  # NEW: Bill toggle
):
    """
    Calculates a 2D DataFrame of LCOE values for sensitivity analysis.
    This version runs the full simulation for all 64 combinations
    to get the annual bill for the new LCOE formulas.
    """
    lcoe_data = []

    # Calculate ANF once
    anf = calculate_anf(_interest_rate, _lifetime)

    for loop_pv_kwp in pv_axis_labels:
        row_lcoe = []
        for loop_batt_kwh in batt_axis_labels:

            # 1. Run the full simulation for THIS specific combination
            _df_sim, totals = simulate_with_batt(
                _df_hourly, loop_pv_kwp, _pr, _load_kwh_8760, loop_batt_kwh
            )

            # 2. Get results needed for LCOE
            # --- UPDATED: Pass _allow_negative_bill ---
            annual_bill_scenario = annual_bill(totals, _retail, _fit, _allow_negative_bill)
            annual_pv_production = totals['pv_production_kWh']

            # 3. Calculate CAPEX and Annual OPEX for this combo
            capex = (loop_pv_kwp * _pv_capex) + (loop_batt_kwh * _batt_capex)
            annual_opex = capex * _opex_rate

            # 4. Calculate LCOE based on financing
            lcoe = 0.0
            if _interest_rate > 0:
                # Use Financed Formula (Formula 2)
                if annual_pv_production > 1e-9:
                    numerator = (anf * capex) + annual_opex + annual_bill_scenario
                    denominator = annual_pv_production
                    lcoe = numerator / denominator
            else:
                # Use Un-Financed Formula (Formula 3)
                lifetime_gen = annual_pv_production * _lifetime
                if lifetime_gen > 1e-9:
                    lifetime_opex = annual_opex * _lifetime
                    lifetime_bill = annual_bill_scenario * _lifetime
                    numerator = capex + lifetime_opex + lifetime_bill
                    denominator = lifetime_gen
                    lcoe = numerator / denominator

            row_lcoe.append(lcoe if lcoe > 0 else np.nan)

        lcoe_data.append(row_lcoe)

    # 5. Create the final DataFrame
    df_lcoe = pd.DataFrame(
        lcoe_data,
        index=pd.Index(pv_axis_labels, name="PV (kWp) - Battery (kWh)"),
        columns=pd.Index(batt_axis_labels, name="Battery (kWh)")
    )

    return df_lcoe


# --- 11. NEW: Helper function for TCO Sensitivity Table ---
# --- REMOVED @st.cache_data ---
def calculate_tco_sensitivity(
        pv_axis_labels, batt_axis_labels,  # The axes for the table
        _pv_capex, _batt_capex, _opex_rate, _lifetime  # Economic inputs
):
    """
    Calculates a 2D DataFrame of TCO (Total Cost of Ownership) values.
    """
    tco_data = []

    for loop_pv_kwp in pv_axis_labels:
        row_tco = []
        for loop_batt_kwh in batt_axis_labels:
            # Calculate economics for this specific PV/Battery combination
            capex = (loop_pv_kwp * _pv_capex) + (loop_batt_kwh * _batt_capex)
            tco = capex * (1 + (_opex_rate * _lifetime))
            row_tco.append(tco)

        tco_data.append(row_tco)

    # 3. Create the final DataFrame
    df_tco = pd.DataFrame(
        tco_data,
        index=pd.Index(pv_axis_labels, name="PV (kWp) - Battery (kWh)"),
        columns=pd.Index(batt_axis_labels, name="Battery (kWh)")
    )

    return df_tco


# --- 12. NEW: Helper function for Savings Sensitivity Table ---
# --- REMOVED @st.cache_data ---
def calculate_savings_sensitivity(
        _df_hourly, _pr, _load_kwh_8760,  # Simulation inputs
        pv_axis_labels, batt_axis_labels,  # The axes for the table
        _retail, _fit, _annual_demand, _lifetime,  # Economic inputs
        _allow_negative_bill  # NEW: Bill toggle
):
    """
    Calculates a 2D DataFrame of Total Lifetime Savings values.
    This is the most intensive calculation, as it must run the full
    battery simulation for every PV/Battery combination (64 times).
    """
    savings_data = []

    # Calculate the baseline annual cost (constant)
    _baseline_annual_bill = _annual_demand * _retail

    for loop_pv_kwp in pv_axis_labels:
        row_savings = []
        for loop_batt_kwh in batt_axis_labels:
            # 1. Run the full simulation for THIS specific combination
            _df_sim, totals = simulate_with_batt(
                _df_hourly, loop_pv_kwp, _pr, _load_kwh_8760, loop_batt_kwh
            )

            # 2. Calculate the new annual bill for this combination
            # --- UPDATED: Pass _allow_negative_bill ---
            new_annual_bill = annual_bill(totals, _retail, _fit, _allow_negative_bill)

            # 3. Calculate lifetime savings
            annual_savings = _baseline_annual_bill - new_annual_bill
            lifetime_savings = annual_savings * _lifetime

            row_savings.append(lifetime_savings)

        savings_data.append(row_savings)

    # 3. Create the final DataFrame
    df_savings = pd.DataFrame(
        savings_data,
        index=pd.Index(pv_axis_labels, name="PV (kWp) - Battery (kWh)"),
        columns=pd.Index(batt_axis_labels, name="Battery (kWh)")
    )

    return df_savings


# --- 13. NEW: Helper function for Net Savings Sensitivity Table ---
# --- REMOVED @st.cache_data ---
def calculate_net_savings_sensitivity(
        _df_hourly, _pr, _load_kwh_8760,  # Simulation inputs
        pv_axis_labels, batt_axis_labels,  # The axes for the table
        _retail, _fit, _annual_demand, _lifetime,  # Economic inputs
        _pv_capex, _batt_capex, _opex_rate,
        _allow_negative_bill  # NEW: Bill toggle
):
    """
    Calculates a 2D DataFrame of Net Lifetime Savings (Savings - TCO) values.
    This combines the logic of the previous two sensitivity functions.
    """
    net_savings_data = []

    # Calculate the baseline annual cost (constant)
    _baseline_annual_bill = _annual_demand * _retail

    for loop_pv_kwp in pv_axis_labels:
        row_net_savings = []
        for loop_batt_kwh in batt_axis_labels:
            # --- 1. Calculate Lifetime Savings ---
            _df_sim, totals = simulate_with_batt(
                _df_hourly, loop_pv_kwp, _pr, _load_kwh_8760, loop_batt_kwh
            )
            # --- UPDATED: Pass _allow_negative_bill ---
            new_annual_bill = annual_bill(totals, _retail, _fit, _allow_negative_bill)
            annual_savings = _baseline_annual_bill - new_annual_bill
            lifetime_savings = annual_savings * _lifetime

            # --- 2. Calculate Total Cost of Ownership (TCO) ---
            capex = (loop_pv_kwp * _pv_capex) + (loop_batt_kwh * _batt_capex)
            tco = capex * (1 + (_opex_rate * _lifetime))

            # --- 3. Calculate Net Savings ---
            net_savings = lifetime_savings - tco

            row_net_savings.append(net_savings)

        net_savings_data.append(row_net_savings)

    # 4. Create the final DataFrame
    df_net_savings = pd.DataFrame(
        net_savings_data,
        index=pd.Index(pv_axis_labels, name="PV (kWp) - Battery (kWh)"),
        columns=pd.Index(batt_axis_labels, name="Battery (kWh)")
    )

    return df_net_savings


# --------------------------------------------------------------------------------
# User Interface
# --------------------------------------------------------------------------------

# --- 1. Load data from previous pages and set fixed parameters ---
gti_val = st.session_state["gti_kwh_m2yr"]
annual_demand = st.session_state["annual_demand"]
lat, lon = st.session_state["user_location"]
pr = 0.75  # Performance Ratio is now a fixed default
load_profile_8760_kwh = st.session_state["generated_8760_profile_df"]['Energy Demand (kWh)'].values

# --- 2. Display the core inputs being used for the simulation ---
st.subheader("Core Simulation Inputs")
st.markdown(
    "These values are set in the previous pages. They form the basis for this simulation."
)

col1, col2, col3 = st.columns(3)
col1.metric("☀️ Global Tilted Irradiation", f"{gti_val:,.0f} kWh/m²/yr")
col2.metric("💡 Annual Energy Demand", f"{annual_demand:,.0f} kWh/yr")
col3.metric("⚙️ PV Performance Ratio", f"{pr:.0%}")

st.markdown("---")

# --- 4. NEW: Economic Parameters --- (MOVED UP)
st.subheader("Economic Parameters")

# Create 5 columns
e_col1, e_col2, e_col3, e_col4, e_col5 = st.columns(5)
with e_col1:
    pv_capex_per_kwp = st.number_input(
        "PV Price (€/kWp)", min_value=0,
        value=st.session_state.get("pv_capex_per_kwp", 1200),
        step=50, key="pv_capex_per_kwp"
    )
with e_col2:
    batt_capex_per_kwh = st.number_input(
        "Battery Price (€/kWh)", min_value=0,
        value=st.session_state.get("batt_capex_per_kwh", 800),
        step=50, key="batt_capex_per_kwh"
    )
with e_col3:
    opex_percent = st.number_input(
        "Annual OPEX (% of CAPEX)", min_value=0.0, max_value=100.0,
        value=st.session_state.get("opex_percent", 2.0),
        step=0.1, format="%.1f", key="opex_percent",
        help="Annual operational cost as a percentage of the initial investment (CAPEX)."
    )
with e_col4:
    project_lifetime = st.number_input(
        "Project Lifetime (Years)", min_value=1, max_value=50,
        value=st.session_state.get("project_lifetime", 20),
        step=1, key="project_lifetime"
    )
# --- NEW: Interest Rate Input ---
with e_col5:
    interest_rate = st.number_input(
        "Interest Rate (%)", min_value=0.0, max_value=100.0,
        value=st.session_state.get("interest_rate", 2.0),
        step=0.1, format="%.1f", key="interest_rate",
        help="Interest rate for financing. Set to 0 for an un-financed project."
    )

# --- 3. Interactive simulation controls --- (MOVED DOWN)
st.subheader("Simulation Controls")

# --- NEW: Create a placeholder for the dynamic metrics ---
col1, col2 = st.columns([0.6, 0.4])  # 60% for inputs, 40% for metrics

with col1:
    st.markdown(
        "These values are **linked** across the app. "
        "Changing them here will also change them on other pages."
    )
    c1, c2 = st.columns(2)
    with c1:
        retail = st.number_input(
            "Retail Price (€/kWh)", 0.0, 2.0,
            value=st.session_state.get("retail", 0.35),
            step=0.01,
            key="retail"
        )
    with c2:
        fit = st.number_input(
            "Feed-in Tariff (€/kWh)", 0.0, 2.0,
            value=st.session_state.get("fit", 0.10),
            step=0.01,
            key="fit"
        )

    # --- NEW: Add toggle for negative bill ---
    allow_negative_bill = st.toggle(
        "Allow negative annual bill (profit)?",
        value=st.session_state.get("allow_negative_bill", True),
        key="allow_negative_bill",
        help="If 'On', the annual bill can be negative (a profit). If 'Off', the minimum annual bill is €0."
    )

    # PV Plant Size
    # Calculate a sensible 'needed' PV size to guide the. user
    production_ratio = gti_val * pr
    needed_kwp = annual_demand / production_ratio if production_ratio > 0 else 0
    suggested_kwp = float(math.ceil(needed_kwp))
    default_pv_kwp = st.session_state.get("pv_kwp", suggested_kwp)

    pv_kwp = st.number_input(
        "Enter the size of the PV system (in kWp):",
        min_value=0.0,
        value=default_pv_kwp,
        step=0.1,
        format="%.1f",
        help=f"Suggested size to cover annual demand: {suggested_kwp} kWp",
        key="pv_kwp"
    )

    # --- MOVED BATTERY INPUTS HERE ---
    st.markdown("Define the specifications for your battery system.")
    c1_batt, c2_batt = st.columns(2)
    with c1_batt:
        batt_kwh = st.number_input(
            "Battery Usable Capacity (kWh)",
            min_value=0.0,
            value=st.session_state.get("batt_kwh", 5.0),
            step=0.5,
            format="%.1f",
            key="batt_kwh"
        )
    with c2_batt:
        batt_count = st.number_input(
            "Number of Batteries",
            min_value=0,  # Allow 0 to "disable"
            value=st.session_state.get("batt_count", 1),
            step=1,
            key="batt_count"
        )

    # Display total system size
    total_batt_kwh = batt_kwh * batt_count
    # st.info(f"**Total System:** {total_batt_kwh:.1f} kWh Usable Capacity") # <-- REMOVED AS REQUESTED
    # --- END OF MOVED SECTION ---

with col2:
    # This placeholder will be filled later, after calculations
    key_metrics_placeholder = st.empty()

# --- 5. Main Simulation Logic ---
year = 2023  # Fixed weather year

try:
    df_hourly = fetch_hourly_pvgis_era5_year(lat, lon, year)

    if len(df_hourly) != len(load_profile_8760_kwh):
        st.error(
            f"Data length mismatch. PV data has {len(df_hourly)} hours, Load data has {len(load_profile_8760_kwh)} hours.")
        st.stop()

    # --- 6. "No Battery" Simulation (Runs automatically) ---
    df_sim_no_batt, totals_no_batt = simulate_no_batt(df_hourly, pv_kwp, pr, load_profile_8760_kwh)
    # --- UPDATED: Pass allow_negative_bill ---
    bill_eur_no_batt = annual_bill(totals_no_batt, retail, fit, allow_negative_bill)
    scr_val_no_batt = scr(totals_no_batt)  # NEW
    ssr_val_no_batt = ssr(totals_no_batt)  # NEW (replaces doa_val_no_batt)

    # --- NEW: Economic Results (No Battery) ---
    # (ALL CALCULATIONS REMAIN)
    opex_rate = opex_percent / 100.0
    baseline_annual_bill = annual_demand * retail  # Baseline cost without any PV

    annual_savings_no_batt = baseline_annual_bill - bill_eur_no_batt
    total_savings_lifetime_no_batt = annual_savings_no_batt * project_lifetime

    capex_no_batt = pv_kwp * pv_capex_per_kwp
    annual_opex_no_batt = capex_no_batt * opex_rate
    tco_lifetime_no_batt = capex_no_batt * (1 + (opex_rate * project_lifetime))

    # --- LCOE Calculation (No Battery) - UPDATED ---
    lcoe_no_batt = 0.0
    anf = calculate_anf(interest_rate, project_lifetime)
    annual_pv_gen_no_batt = totals_no_batt['pv_production_kWh']

    if interest_rate > 0:
        # Financed LCOE (Formula 2)
        if annual_pv_gen_no_batt > 1e-9:
            numerator = (anf * capex_no_batt) + annual_opex_no_batt + bill_eur_no_batt
            lcoe_no_batt = numerator / annual_pv_gen_no_batt
    else:
        # Un-Financed LCOE (Formula 3)
        lifetime_gen_no_batt = annual_pv_gen_no_batt * project_lifetime
        if lifetime_gen_no_batt > 1e-9:
            lifetime_opex = annual_opex_no_batt * project_lifetime
            lifetime_bill = bill_eur_no_batt * project_lifetime
            numerator = capex_no_batt + lifetime_opex + lifetime_bill
            lcoe_no_batt = numerator / lifetime_gen_no_batt

    # --- NEW: Calculate Net Savings (No Battery) ---
    net_savings_lifetime_no_batt = total_savings_lifetime_no_batt - tco_lifetime_no_batt

    # --- 7. "With Battery" Simulation (NEW SIMPLIFIED SECTION) ---
    # --- REMOVED SUBHEADER AND INPUTS, as they are now in Section 3 ---
    st.markdown("---")  # Added a markdown line for separation

    # (Button removed, simulation now runs automatically)
    with st.spinner("Running 8760-hour battery simulation..."):
        # (Call the new simplified simulation function)
        # This will run with total_batt_kwh=0 if count is 0,
        # which will produce the same result as the 'no_batt' simulation.
        df_sim_batt, totals_batt = simulate_with_batt(
            df_hourly,
            pv_kwp,
            pr,
            load_profile_8760_kwh,
            total_batt_kwh
        )

        # (Store results in session state - good practice)
        st.session_state.simulation_results = df_sim_batt
        st.session_state.simulation_totals = totals_batt

        # (Calculate new metrics)
        # --- UPDATED: Pass allow_negative_bill ---
        bill_eur_batt = annual_bill(totals_batt, retail, fit, allow_negative_bill)
        scr_val_batt = scr(totals_batt)  # NEW
        ssr_val_batt = ssr(totals_batt)  # NEW (replaces doa_val_no_batt)

        # --- NEW: Economic Results (With Battery) ---
        annual_savings_batt = baseline_annual_bill - bill_eur_batt
        total_savings_lifetime_batt = annual_savings_batt * project_lifetime

        capex_batt = (pv_kwp * pv_capex_per_kwp) + (total_batt_kwh * batt_capex_per_kwh)
        annual_opex_batt = capex_batt * opex_rate
        tco_lifetime_batt = capex_batt * (1 + (opex_rate * project_lifetime))

        # --- LCOE Calculation (With Battery) - UPDATED ---
        lcoe_batt = 0.0
        # 'anf' is already calculated from the 'No Battery' section
        annual_pv_gen_batt = totals_batt['pv_production_kWh']

        if interest_rate > 0:
            # Financed LCOE (Formula 2)
            if annual_pv_gen_batt > 1e-9:
                numerator = (anf * capex_batt) + annual_opex_batt + bill_eur_batt
                lcoe_batt = numerator / annual_pv_gen_batt
        else:
            # Un-Financed LCOE (Formula 3)
            lifetime_gen_batt = annual_pv_gen_batt * project_lifetime
            if lifetime_gen_batt > 1e-9:
                lifetime_opex = annual_opex_batt * project_lifetime
                lifetime_bill = bill_eur_batt * project_lifetime
                numerator = capex_batt + lifetime_opex + lifetime_bill
                lcoe_batt = numerator / lifetime_gen_batt

        # --- NEW: Calculate Net Savings (With Battery) ---
        net_savings_lifetime_batt = total_savings_lifetime_batt - tco_lifetime_batt

        # --- CHANGED: Fill the MAIN placeholder (from Section 3) ---
        with key_metrics_placeholder.container():
            st.markdown("##### Simulation Results")  # Simplified title
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                # UPDATED LCOE DISPLAY: Custom HTML for color/bold + hover tooltip
                st.markdown(
                    f"""
                    <div style="background-color: #e8f5e9; padding: 15px; border: 2px solid #2e7d32; border-radius: 10px; margin-bottom: 15px;">
                        <p style="margin: 0; font-size: 18px; font-weight: bold; color: #1b5e20;">
                            LCOE (All Costs Included) 
                            <span title="Overall electricity cost per kWh" style="cursor: help; color: #1b5e20; font-size: 16px; vertical-align: top;">&#9432;</span>
                        </p>
                        <p style="margin: 0; font-size: 32px; font-weight: 900; color: #1b5e20;">
                            €{lcoe_batt:.2f} / kWh
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.metric("Net Lifetime Savings", f"€{net_savings_lifetime_batt:,.0f}")
                st.metric("Grid Import", f"{totals_batt['grid_import_kWh']:.0f} kWh",
                          help="Energy bought from the grid.")
                st.metric("Self-Consumption", f"{scr_val_batt:.1%}",
                          help="The percentage of your PV production that was used on-site (either by the load or to charge the battery).")
            with m_col2:
                st.metric("Annual Bill", f"€{bill_eur_batt:,.2f}",
                          help="Your final electricity bill after accounting for imports and exports.")
                st.metric("Grid Export", f"{totals_batt['export_kWh']:.0f} kWh",
                          help="Surplus PV energy sold to the grid.")
                st.metric("Self-Sufficiency", f"{ssr_val_batt:.1%}",
                          help="The percentage of your electricity demand covered by your own system (from PV or battery).")

        st.markdown("##### Economic Results ")
        ec1b, ec2b, ec4b = st.columns(3)  # Changed to 3
        ec1b.metric("Total Investment (CAPEX)", f"€{capex_batt:,.0f}",
                    help="Initial cost for the PV system AND the battery system.")
        ec2b.metric("Total Cost of Ownership (TCO)", f"€{tco_lifetime_batt:,.0f}",
                    help=f"{project_lifetime}-year TCO, including CAPEX and annual OPEX.")
        # ec3b.metric("LCOE", f"€{lcoe_batt:.2f} / kWh",  <-- REMOVED
        #             help="Levelized Cost of Electricity: The average cost per kWh supplied over the project lifetime.")
        ec4b.metric("Savings on Electricity Bill", f"€{total_savings_lifetime_batt:,.0f}",  # <-- FIXED
                    help=f"Savings over {project_lifetime} years compared to buying all energy from the grid.")

        # (Add expander for battery data)
        with st.expander("View Aggregated 'With Battery' Simulation Data"):
            tab1b, tab2b = st.tabs(["Daily Data", "Hourly Data"])
            with tab1b:
                st.markdown("Daily totals for the year (in kWh)")
                daily_batt = df_sim_batt.resample("D").sum(numeric_only=True)
                st.dataframe(daily_batt)
            with tab2b:
                st.markdown("Full 8760-hour simulation data (in kWh)")
                st.dataframe(df_sim_batt)

    # --- 8. NEW: Monthly Coverage Charts (Bottom of page) ---
    st.markdown("---")
    st.subheader("Monthly Energy Demand Coverage")

    # Layout for Monthly and Weekly charts
    cov_col1, cov_col2 = st.columns([1, 1])

    with cov_col1:
        st.markdown("##### Annual Overview (Monthly)")
        # Create and display the "With Battery" chart
        chart_with_batt = create_monthly_coverage_chart(
            df_sim_batt,
            "Monthly Coverage"
        )
        st.altair_chart(chart_with_batt, use_container_width=True)

    with cov_col2:
        st.markdown("##### Weekly Detail (Daily)")
        # Changed slider to selectbox as requested
        selected_week = st.selectbox("Select Week", range(1, 53), index=24, key="coverage_week_select")

        # New helper function call
        chart_weekly_daily = create_weekly_daily_coverage_chart(
            df_sim_batt,
            selected_week,
            "Daily Coverage"
        )
        st.altair_chart(chart_weekly_daily, use_container_width=True)

    # --- 9. NEW: PV Energy Use Charts (Bottom of page) ---
    st.markdown("---")
    st.subheader("Use of PV Energy")

    use_col1, use_col2 = st.columns([1, 1])

    with use_col1:
        st.markdown("##### Annual Overview (Monthly)")
        chart_pv_use_batt = create_monthly_pv_use_chart(
            df_sim_batt,
            "Monthly PV Use"
        )
        st.altair_chart(chart_pv_use_batt, use_container_width=True)

    with use_col2:
        st.markdown("##### Weekly Detail (Daily)")
        # Changed slider to selectbox as requested
        selected_week_use = st.selectbox("Select Week", range(1, 53), index=24, key="pv_use_week_select")

        chart_weekly_daily_use = create_weekly_daily_pv_use_chart(
            df_sim_batt,
            selected_week_use,
            "Daily PV Use"
        )
        st.altair_chart(chart_weekly_daily_use, use_container_width=True)

    # --- 10. NEW: Sensitivity Analysis ---
    st.markdown("---")
    st.subheader("LCOE Sensitivity Analysis (€/kWh)")
    st.markdown("How LCOE changes with different PV and Battery system sizes.")

    # 1. Define the 8 PV size labels
    # Center value (pos 4) is the current pv_kwp
    center_pv_kwp = round(pv_kwp, 1)  # Use the rounded value from the input
    pv_axis_labels = [
        max(1.0, center_pv_kwp - 3),
        max(1.0, center_pv_kwp - 2),
        max(1.0, center_pv_kwp - 1),
        center_pv_kwp,
        center_pv_kwp + 1,
        center_pv_kwp + 2,
        center_pv_kwp + 3,
        center_pv_kwp + 4
    ]
    # Ensure no duplicates and all are positive
    pv_axis_labels = sorted(list(set([max(1.0, round(p, 1)) for p in pv_axis_labels])))
    # If list is still not 8, pad it out (edge case)
    while len(pv_axis_labels) < 8:
        pv_axis_labels.append(round(pv_axis_labels[-1] + 1, 1))

    # 2. Define the 8 Battery size labels
    # Center value (pos 4) is the current total_batt_kwh
    center_batt_kwh = round(total_batt_kwh, 1)
    batt_axis_labels = [
        0.0,  # Position 1 is always 0
        max(0.0, center_batt_kwh - 2),
        max(0.0, center_batt_kwh - 1),
        center_batt_kwh,
        center_batt_kwh + 1,
        center_batt_kwh + 2,
        center_batt_kwh + 3,
        center_batt_kwh + 4
    ]
    # Ensure no duplicates and all are positive
    batt_axis_labels = sorted(list(set([max(0.0, round(b, 1)) for b in batt_axis_labels])))
    # If list is not 8, pad it out
    while len(batt_axis_labels) < 8:
        batt_axis_labels.append(round(batt_axis_labels[-1] + 1, 1))

    # 3. Run the sensitivity calculation
    # We pass the fetched df_hourly and economic params
    with st.spinner("Running LCOE Sensitivity Analysis..."):
        df_lcoe = calculate_lcoe_sensitivity(
            df_hourly, pr, load_profile_8760_kwh,
            pv_axis_labels, batt_axis_labels,
            pv_capex_per_kwp, batt_capex_per_kwh, opex_rate, project_lifetime,
            interest_rate, retail, fit,
            allow_negative_bill  # <-- Pass new param
        )

        # 4. Style and display the DataFrame
        st.dataframe(
            df_lcoe.style
            .apply(highlight_min_cell, axis=None)  # <-- ADDED
            .apply(center_align_cells, axis=None)  # <-- ADDED
            .set_table_styles(header_style)  # <-- ADDED
            .background_gradient(cmap='Oranges', axis=None)
            .format("{:.2f}")
            .set_caption("LCOE (€/kWh) vs. System Size")
        )

    # --- 11. NEW: Net Lifetime Savings Sensitivity Analysis (MOVED) ---
    st.markdown("---")
    st.subheader("Net Lifetime Savings - Sensitivity Analysis (€)")
    st.markdown("How net profit (Savings - TCO) changes with different PV and Battery system sizes.")

    # 1. Run the Net Savings sensitivity calculation
    with st.spinner("Running Net Savings Sensitivity Analysis..."):
        df_net_savings = calculate_net_savings_sensitivity(
            df_hourly, pr, load_profile_8760_kwh,
            pv_axis_labels, batt_axis_labels,
            retail, fit, annual_demand, project_lifetime,
            pv_capex_per_kwp, batt_capex_per_kwh, opex_rate,
            allow_negative_bill  # <-- Pass new param
        )

        # 2. Style and display the DataFrame
        st.dataframe(
            df_net_savings.style
            .apply(highlight_max_cell, axis=None)  # <-- ADDED
            .apply(center_align_cells, axis=None)  # <-- ADDED
            .set_table_styles(header_style)  # <-- ADDED
            .background_gradient(cmap='YlOrRd', axis=None)  # <-- CHANGED
            .format("€{:,.0f}")  # Format as currency
            .set_caption("Net Lifetime Savings (Savings - TCO) (€) vs. System Size")
        )

    # --- 12. NEW: TCO Sensitivity Analysis (MOVED) ---
    st.markdown("---")
    st.subheader("Total Cost of Ownership - Sensitivity Analysis (€)")
    st.markdown("How TCO changes with different PV and Battery system sizes.")

    # 1. Run the TCO sensitivity calculation (uses the same axes)
    with st.spinner("Running TCO Sensitivity Analysis..."):
        df_tco = calculate_tco_sensitivity(
            pv_axis_labels, batt_axis_labels,
            pv_capex_per_kwp, batt_capex_per_kwh, opex_rate, project_lifetime
        )

        # 2. Style and display the DataFrame
        st.dataframe(
            df_tco.style
            .apply(highlight_min_cell, axis=None)  # <-- ADDED
            .apply(center_align_cells, axis=None)  # <-- ADDED
            .set_table_styles(header_style)  # <-- ADDED
            .background_gradient(cmap='Blues', axis=None)  # Use Blues colormap
            .format("€{:,.0f}")  # Format as currency
            .set_caption("TCO (€) vs. System Size")
        )

    # --- 13. NEW: Total Lifetime Savings Sensitivity Analysis (MOVED) ---
    st.markdown("---")
    st.subheader("Total Lifetime Savings - Sensitivity Analysis (€)")
    st.markdown("How lifetime savings change with different PV and Battery system sizes.")

    # 1. Run the Savings sensitivity calculation
    with st.spinner("Running Total Savings Sensitivity Analysis..."):
        df_savings = calculate_savings_sensitivity(
            df_hourly, pr, load_profile_8760_kwh,
            pv_axis_labels, batt_axis_labels,
            retail, fit, annual_demand, project_lifetime,
            allow_negative_bill  # <-- Pass new param
        )

        # 2. Style and display the DataFrame
        st.dataframe(
            df_savings.style
            .apply(highlight_max_cell, axis=None)  # <-- ADDED
            .apply(center_align_cells, axis=None)  # <-- ADDED
            .set_table_styles(header_style)  # <-- ADDED
            .background_gradient(cmap='Greens', axis=None)  # Use Greens colormap
            .format("€{:,.0f}")  # Format as currency
            .set_caption("Total Lifetime Savings (€) vs. System Size")
        )

except Exception as e:
    st.error(f"Simulation failed: {e}")