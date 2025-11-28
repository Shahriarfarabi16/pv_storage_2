import numpy as np
import pandas as pd
import streamlit as st
import re
import altair as alt
import io  # Added for Excel export functionality

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Electricity Demand Profile", page_icon="💡", layout="wide")
st.title("💡 Electricity Demand Profile")
st.markdown(
    "Choose a method to define the daily electricity demand for your household. "
    "This profile will be used in the subsequent simulation pages."
)

# --------------------------------------------------------------------------------
# Data & Constants
# --------------------------------------------------------------------------------
APPLIANCES = {
    "Kitchen": [
        {"name": "Refrigerator", "power_w": 45, "icon": "🧊"},
        {"name": "Microwave", "power_w": 1050, "icon": "⚡"},
        {"name": "Toaster", "power_w": 1000, "icon": "🍞"},
        {"name": "Coffee Machine", "power_w": 1200, "icon": "☕"},
        {"name": "Dishwasher", "power_w": 2150, "icon": "🍽️"},
    ],
    "Laundry & Cleaning": [
        {"name": "Washing Machine", "power_w": 2150, "icon": "🧺"},
        {"name": "Vacuum Cleaner", "power_w": 900, "icon": "🧹"},
    ],
    "Entertainment & Office": [
        {"name": "TV (LED/OLED)", "power_w": 80, "icon": "📺"},
        {"name": "Game Console", "power_w": 200, "icon": "🎮"},
        {"name": "Wi-Fi Router", "power_w": 10, "icon": "📶"},
        {"name": "Light", "power_w": 20, "icon": "💡"},
    ]
}

# --- SLP Calculation Tables ---
APARTMENT_DEMAND = {
    # (electric, non-electric)
    1: (1600, 1200), 2: (2500, 1900), 3: (3500, 2400),
    4: (4000, 2600), 5: (5000, 3100),
}
HOUSE_DEMAND = {
    # (electric, non-electric)
    1: (2100, 1800), 2: (3200, 2700), 3: (4100, 3500),
    4: (4700, 3800), 5: (6000, 4500),
}

# --------------------------------------------------------------------------------
# Initialize Session State
# --------------------------------------------------------------------------------
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "custom_appliance_name" not in st.session_state:
    st.session_state.custom_appliance_name = ""
if "custom_appliance_power" not in st.session_state:
    st.session_state.custom_appliance_power = 1000
if "schedule_id_counter" not in st.session_state:
    st.session_state.schedule_id_counter = 0
# To store the generated 8760 profile for download
if "generated_8760_profile_df" not in st.session_state:
    st.session_state.generated_8760_profile_df = None
# --- Add keys for Upload method to preserve state ---
if "uploader_existing_profile" not in st.session_state:
    st.session_state.uploader_existing_profile = None

# --- Add keys for SLP method ---
if "slp_input_method" not in st.session_state:
    st.session_state.slp_input_method = "Calculate Automatically"
if "slp_household_size" not in st.session_state:
    st.session_state.slp_household_size = 1
if "slp_building_type" not in st.session_state:
    st.session_state.slp_building_type = "Apartment"
if "slp_hot_water" not in st.session_state:
    st.session_state.slp_hot_water = "Without Electric"
if "slp_manual_demand" not in st.session_state:
    st.session_state.slp_manual_demand = 3500


# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------
# --- SLP Loading (cached) ---
SLP_FILE_PATH = "data/slp_household.csv"

@st.cache_data
def load_slp_data(file_path):
    df = pd.read_csv(file_path)
    # Find the correct column name (it might vary slightly)
    if "Dynamized Evergy Value (kWh)" in df.columns:
        col_name = "Dynamized Evergy Value (kWh)"
    else:
        value_col = [col for col in df.columns if "Value" in col or "kWh" in col]
        if not value_col:
            raise ValueError("Could not find the energy value column in the SLP CSV.")
        col_name = value_col[0] # Take the first match

    profile_8760_raw = df[col_name].to_numpy(dtype=float)

    if len(profile_8760_raw) != 8760:
        raise ValueError(f"Standard profile file is expected to have 8760 rows, but it has {len(profile_8760_raw)}.")

    return profile_8760_raw, df # Return raw data and the dataframe

# --- Profile Builder Functions ---
def add_appliance_to_schedule(appliance):
    appliance_instance = appliance.copy()
    appliance_instance["id"] = st.session_state.schedule_id_counter
    st.session_state.schedule_id_counter += 1
    appliance_instance["usage_text"] = "08:00-09:00"
    st.session_state.schedule.append(appliance_instance)

def remove_appliance_from_schedule(appliance_id):
    st.session_state.schedule = [
        item for item in st.session_state.schedule if item.get("id") != appliance_id
    ]
    st.session_state.generated_8760_profile_df = None # Clear generated profile if schedule changes
    st.rerun()

def reset_schedule():
    """Clears the entire appliance schedule and resets the builder."""
    st.session_state.schedule = []
    st.session_state.schedule_id_counter = 0
    st.session_state.custom_appliance_name = ""
    st.session_state.custom_appliance_power = 1000
    if "hourly_profile_df" in st.session_state:
        del st.session_state["hourly_profile_df"] # Clear downloaded profile on reset
    st.session_state.generated_8760_profile_df = None

def parse_time_ranges_to_minutes(time_str: str):
    """Parses a comma-separated string of time ranges into minute-of-the-day tuples."""
    ranges = []
    parts = time_str.split(',')
    for part in parts:
        part = part.strip()
        if not part: continue

        match = re.match(r"^(\d{2}):(\d{2})\s*-\s*(\d{2}):(\d{2})$", part)
        if not match: return f"Invalid format: '{part}'. Use HH:MM-HH:MM."

        start_h, start_m, end_h, end_m = map(int, match.groups())

        if not (0 <= start_h < 24 and 0 <= start_m < 60):
            return f"Invalid start time in '{part}'. Hours must be 0-23, minutes 0-59."

        start_minute = start_h * 60 + start_m

        if end_h == 24 and end_m == 0:
            end_minute = 24 * 60
        elif not (0 <= end_h < 24 and 0 <= end_m < 60):
            return f"Invalid end time in '{part}'. Hours must be 0-23, minutes 0-59 (or 24:00)."
        else:
            end_minute = end_h * 60 + end_m

        if start_minute >= end_minute:
            return f"End time must be after start time in '{part}'."

        ranges.append((start_minute, end_minute))
    return ranges

def calculate_demand_profile_from_schedule(schedule):
    """Calculates a 1-minute resolution power profile."""
    profile_kw = np.zeros(24 * 60 + 1) # 1441 points to include 24:00
    for item in schedule:
        if "parsed_usage" in item and isinstance(item["parsed_usage"], list):
            power_kw = item["power_w"] / 1000.0
            for start_minute, end_minute in item["parsed_usage"]:
                profile_kw[start_minute:end_minute] += power_kw
    return process_minute_profile(profile_kw)

def process_minute_profile(profile_kw: np.ndarray):
    """Takes a 1441-point kW profile and calculates metrics and series for plotting."""
    total_kwh = profile_kw[:-1].sum() / 60.0

    time_index = pd.date_range("2024-01-01 00:00", "2024-01-02 00:00", freq="1T")
    profile_series = pd.Series(profile_kw, index=time_index, name="Demand (kW)")

    return profile_series, total_kwh

# --- Excel Functions ---
def to_excel(df):
    """Converts a DataFrame to an Excel file in-memory."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Hourly_Profile', index=False) # Do not write the dataframe index
    processed_data = output.getvalue()
    return processed_data

def create_sample_hourly_file():
    """Creates an in-memory Excel file for the 24-hour template."""
    new_index = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)]
    df = pd.DataFrame(
        {'Time Range': new_index,
         'Energy Demand (kWh)': [0.0] * 24}
    )
    return to_excel(df[['Time Range', 'Energy Demand (kWh)']])

def create_sample_annual_file():
    """Creates an in-memory Excel file for the 8760-hour template."""
    days = np.repeat(range(1, 366), 24) # Day 1 (24 times), Day 2 (24 times), ...
    hours = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)] * 365

    df = pd.DataFrame({
        'Day': days,
        'Hour Range': hours,
        'Energy Demand (kWh)': [0.0] * 8760
    })

    return to_excel(df[['Day', 'Hour Range', 'Energy Demand (kWh)']])

# --- UPDATED: Function implements the new logic described by user ---
def generate_8760_profile_from_24h_shape(
    shape_24h_normalized: np.ndarray, # User's daily profile shape (sum=1)
    target_annual_demand: float,
    slp_raw_8760: np.ndarray # Raw SLP data (sum=1M)
):
    """Generates an 8760h profile by scaling a 24h shape using SLP's daily totals."""

    # 1. Get SLP Daily Totals
    slp_daily_totals = slp_raw_8760.reshape(365, 24).sum(axis=1)

    # 2. Scale SLP Daily Totals
    slp_sum = slp_daily_totals.sum()
    if slp_sum == 0:
        daily_scaling_factor = 0
    else:
        daily_scaling_factor = target_annual_demand / slp_sum
    scaled_daily_kwh_targets = slp_daily_totals * daily_scaling_factor

    # 3. Apply Building Shape and Synthesize 8760h Profile
    final_profile_8760 = np.zeros(8760)
    for day in range(365):
        start_idx = day * 24
        end_idx = start_idx + 24
        # Apply the normalized building shape to the target energy for this specific day
        daily_profile = shape_24h_normalized * scaled_daily_kwh_targets[day]
        final_profile_8760[start_idx:end_idx] = daily_profile

    # 4. Apply correction to ensure exact sum
    current_sum = final_profile_8760.sum()
    difference = target_annual_demand - current_sum
    if abs(difference) > 1e-9:
        correction_per_hour = difference / 8760.0
        final_profile_8760 += correction_per_hour

    # Recalculate exact annual demand after correction
    final_annual_demand = final_profile_8760.sum()

    return final_profile_8760, final_annual_demand

# --- Updated Save Function ---
def save_profile_to_session(daily_kwh, annual_demand, shape_24_values_kw, hourly_df_for_download, generated_8760_df=None):
    """Saves the final profile data to session state for other pages."""
    st.session_state["daily_kwh"] = daily_kwh
    st.session_state["annual_demand"] = annual_demand # This should now be exact

    # Shape_24 should represent AVERAGE HOURLY POWER (kW), normalized to sum=1
    shape_sum = shape_24_values_kw.sum()
    if shape_sum > 1e-9: # Avoid division by zero
        st.session_state["shape_24"] = shape_24_values_kw / shape_sum
    else: # If sum is zero, use a flat profile
        st.session_state["shape_24"] = np.ones(24) / 24.0

    st.session_state["hourly_profile_df"] = hourly_df_for_download
    st.session_state["generated_8760_profile_df"] = generated_8760_df # Store the 8760 DF if generated

    st.success("Profile has been saved and is ready for the next steps!")

# --------------------------------------------------------------------------------
# Main UI - Method Selection
# --------------------------------------------------------------------------------
st.subheader("1. Choose Your Method")
# Updated order: SLP first, then Upload, then Generator
profile_method = st.radio(
    "Select how you want to create the demand profile:",
    ["Use a Standard Load Profile", "Upload an Existing Profile", "Use the Demand Profile Generator"],
    horizontal=True, label_visibility="collapsed"
)
st.markdown("---")

# --- Load SLP Data (needed for options 1 & 3, and indirectly option 2 logic) ---
slp_raw = None
slp_df_raw = None
# if profile_method != "Use a Standard Load Profile": # Load only if needed
try:
    slp_raw, slp_df_raw = load_slp_data(SLP_FILE_PATH)
except FileNotFoundError:
     st.error(f"Error: Could not find the standard load profile file (`{SLP_FILE_PATH}`). This is needed to add seasonal variation. Please ensure the file exists in the `data` folder.")
     st.stop()
except Exception as e:
     st.error(f"An error occurred loading the standard profile: {e}")
     st.stop()


# --- OPTION 1: Standard Load Profile (SLP) ---
if profile_method == "Use a Standard Load Profile":
    st.subheader("2. Use a Standard Load Profile (SLP)")
    st.markdown("This method uses a standard 8760-hour household profile (normalized to 1,000,000 kWh) and scales it to your annual consumption.")

    # --- Download button for raw SLP file ---
    try:
        raw_slp_excel = to_excel(slp_df_raw)
        st.download_button(
            label="Download Raw 8760h Standard Load Profile (.xlsx)",
            data=raw_slp_excel,
            file_name="raw_slp_household_8760h.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Failed to generate SLP download: {e}")

    st.markdown("---")

    # --- User input method ---
    input_method = st.radio(
        "How do you want to set your annual demand?",
        ["Calculate Automatically", "Enter Manually"],
        horizontal=True,
        key="slp_input_method"
    )

    annual_demand_kwh_target = 0 # This is the target value

    if input_method == "Calculate Automatically":
        st.markdown("##### Household Characteristics")
        col1, col2, col3 = st.columns(3)
        with col1:
            size = st.selectbox("Household Size", options=[1, 2, 3, 4, 5], format_func=lambda x: f"{x} Person(s)", key="slp_household_size")
        with col2:
            b_type = st.radio("Building Type", ["Apartment", "Single-Family House"], key="slp_building_type")
        with col3:
            hot_water = st.radio("Hot Water", ["With Electric", "Without Electric"], key="slp_hot_water")

        lookup_table = APARTMENT_DEMAND if b_type == "Apartment" else HOUSE_DEMAND
        demand_tuple = lookup_table[size]
        annual_demand_kwh_target = demand_tuple[0] if hot_water == "With Electric" else demand_tuple[1]
        st.metric("Calculated Annual Energy Demand (Target)", f"{annual_demand_kwh_target:,.0f} kWh/year")

    else: # "Enter Manually"
        # UPDATED: This pattern explicitly links the widget's value to the session state
        # ensuring it persists across page navigation.
        annual_demand_kwh_target = st.number_input(
            "Enter your Total Annual Energy Demand (kWh/year)",
            min_value=100,
            value=st.session_state.slp_manual_demand,  # Explicitly read value from state
            step=100,
            key="slp_manual_demand" # This key links it to st.session_state
        )


    if annual_demand_kwh_target > 0:
        st.markdown("---")
        st.subheader("3. Your User-Specific 8760h Profile")

        # --- Calculate scaled profile ---
        # Assume SLP raw sums to 1M, calculate scaling factor
        scaling_factor = annual_demand_kwh_target / 1_000_000.0
        profile_kwh = slp_raw * scaling_factor

        # Apply correction
        current_sum = profile_kwh.sum()
        difference = annual_demand_kwh_target - current_sum
        if abs(difference) > 1e-9:
            correction_per_hour = difference / 8760.0
            profile_kwh += correction_per_hour

        final_annual_demand = profile_kwh.sum() # Should be exactly target now
        final_daily_kwh = final_annual_demand / 365.0

        # --- Prepare 8760h download ---
        days = np.repeat(range(1, 366), 24)
        hours = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)] * 365
        df_user_specific = pd.DataFrame({'Day': days, 'Hour Range': hours, 'Energy Demand (kWh)': profile_kwh})
        try:
            user_slp_excel = to_excel(df_user_specific)
            st.download_button(
                label="Download Your User-Specific 8760h Profile (.xlsx)",
                data=user_slp_excel,
                file_name="user_specific_slp_8760h.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Failed to generate user-specific SLP download: {e}")

        # --- Process for saving (shape_24 and avg 24h df) ---
        index_8760 = pd.date_range(start="2023-01-01 00:00", periods=8760, freq='H')
        series_8760 = pd.Series(profile_kwh, index=index_8760)
        # Shape is average HOURLY POWER (kW), which is == average hourly energy for hourly data
        final_shape_24_values_kw = series_8760.groupby(series_8760.index.hour).mean().values

        # Create 24h average dataframe for download (ENERGY kWh)
        # Calculate average hourly ENERGY based on the shape and daily total
        final_shape_kw_sum = final_shape_24_values_kw.sum()
        if final_shape_kw_sum > 1e-9:
            avg_hourly_kwh = (final_shape_24_values_kw / final_shape_kw_sum) * final_daily_kwh
        else:
            avg_hourly_kwh = np.zeros(24)
        new_index = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)]
        profile_series_for_avg_dl = pd.Series(avg_hourly_kwh, index=new_index, name="Average Energy Demand (kWh)")
        hourly_df_for_download = profile_series_for_avg_dl.to_frame()
        hourly_df_for_download.index.name = "Time Range"

        st.metric("Total Annual Energy Demand", f"{final_annual_demand:,.0f} kWh", help="This value will be used for the simulation.")

        if st.button("Confirm and Save Standard Profile"):
            # Save the 8760h profile itself for download later if needed
            save_profile_to_session(final_daily_kwh, final_annual_demand, final_shape_24_values_kw, hourly_df_for_download, df_user_specific)


# --- OPTION 2: Upload an Existing Profile ---
elif profile_method == "Upload an Existing Profile":
    st.subheader("2. Upload Your Custom Profile")
    st.markdown("Download a template, fill it with your hourly **energy (kWh)** values, and upload the completed file.")

    # 1. Download templates
    c1, c2 = st.columns(2)
    with c1:
        sample_excel_24 = create_sample_hourly_file()
        st.download_button(
            label="Download 24-Hour Template (.xlsx)",
            data=sample_excel_24,
            file_name="hourly_demand_template_24h.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with c2:
        sample_excel_8760 = create_sample_annual_file() # New function call
        st.download_button(
            label="Download 8760-Hour Template (.xlsx)", # New button
            data=sample_excel_8760,
            file_name="hourly_demand_template_8760h.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    # 2. Upload file
    uploaded_file = st.file_uploader(
        "Upload your completed CSV or Excel file (must have 24 or 8760 data rows).",
        type=['csv', 'xlsx'],
        key="uploader_existing_profile" # ADDED KEY
    )

    if uploaded_file is not None:
        try:
            # Read the header from the first row
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file, header=0)
            else:
                df = pd.read_excel(uploaded_file, header=0)

            # Check for the required column
            if 'Energy Demand (kWh)' not in df.columns:
                st.error("Error: Uploaded file must contain an 'Energy Demand (kWh)' column.")
                st.stop()

            num_rows = len(df)
            generated_8760_df_upload = None # For 8760h download
            final_shape_24_values_kw = None # Initialize

            if num_rows == 24: # --- Handle 24-hour profile ---
                st.subheader("3. Processing Uploaded 24-Hour Profile") # Changed title
                profile_kwh_24 = df['Energy Demand (kWh)'].to_numpy(dtype=float)

                target_annual_demand = profile_kwh_24.sum() * 365.0

                # Get the user's NORMALIZED 24h shape
                shape_24_user_norm = profile_kwh_24 # kWh values
                shape_sum = shape_24_user_norm.sum()
                if shape_sum > 1e-9:
                    shape_24_user_norm = shape_24_user_norm / shape_sum
                else:
                    shape_24_user_norm = np.ones(24) / 24.0 # Fallback

                # Generate the 8760h profile using the NEW logic
                final_profile_8760, final_annual_demand = generate_8760_profile_from_24h_shape(
                    shape_24_user_norm, target_annual_demand, slp_raw
                )

                # --- Derive outputs for saving ---
                final_daily_kwh = final_annual_demand / 365.0

                index_8760 = pd.date_range(start="2023-01-01 00:00", periods=8760, freq='H')
                series_8760_final_for_shape = pd.Series(final_profile_8760, index=index_8760)
                # Group by hour and calculate MEAN (gives average power kW for that hour slot)
                final_shape_24_values_kw = series_8760_final_for_shape.groupby(series_8760_final_for_shape.index.hour).mean().values

                # Create the 24h average dataframe for download button (showing ENERGY kWh)
                final_shape_kw_sum = final_shape_24_values_kw.sum()
                if final_shape_kw_sum > 1e-9:
                     avg_hourly_kwh = (final_shape_24_values_kw / final_shape_kw_sum) * final_daily_kwh
                else:
                     avg_hourly_kwh = np.zeros(24)
                new_index = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)]
                profile_series_for_avg_dl = pd.Series(avg_hourly_kwh, index=new_index, name="Average Energy Demand (kWh)")
                hourly_df_for_download = profile_series_for_avg_dl.to_frame()
                hourly_df_for_download.index.name = "Time Range"

                # Create the 8760h dataframe for the new download button
                days = np.repeat(range(1, 366), 24)
                hours = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)] * 365
                generated_8760_df_upload = pd.DataFrame({
                    'Day': days, 'Hour Range': hours, 'Energy Demand (kWh)': final_profile_8760
                })

            elif num_rows == 8760: # --- Handle 8760-hour profile ---
                st.subheader("3. Uploaded 8760-Hour Profile (Averaged)")
                profile_kwh_8760 = df['Energy Demand (kWh)'].to_numpy(dtype=float) # Direct use

                # Calculate session state values
                final_annual_demand = profile_kwh_8760.sum()
                final_daily_kwh = final_annual_demand / 365.0

                # Create series with time index to group
                index_8760 = pd.date_range(start="2023-01-01 00:00", periods=8760, freq='H')
                series_8760_final = pd.Series(profile_kwh_8760, index=index_8760)

                # Calculate average 24-hour shape (POWER kW == ENERGY kWh for hourly data)
                final_shape_24_values_kw = series_8760_final.groupby(series_8760_final.index.hour).mean().values

                # Create dataframe for plotting and download (showing average kWh)
                # Use the calculated shape values directly as they represent avg kWh here
                new_index = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)]
                profile_series_for_chart = pd.Series(final_shape_24_values_kw, index=new_index, name="Average Energy Demand (kWh)")

                hourly_df_for_download = profile_series_for_chart.to_frame() # Use avg kWh for download
                hourly_df_for_download.index.name = "Time Range"

                # Prepare the uploaded 8760h data for download format
                days = np.repeat(range(1, 366), 24)
                hours = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)] * 365
                generated_8760_df_upload = pd.DataFrame({
                    'Day': days, 'Hour Range': hours, 'Energy Demand (kWh)': profile_kwh_8760
                })

            else:
                st.error(f"Error: File must contain 24 or 8760 data rows (plus a header). Your file has {num_rows} data rows.")

            # --- Display results section (outside the if/elif for num_rows) ---
            if final_shape_24_values_kw is not None: # Check if processing was successful
                # Display results consistently
                st.metric("Avg. Daily Energy Demand", f"{final_daily_kwh:.2f} kWh")
                st.metric("Total Annual Energy Demand", f"{final_annual_demand:,.0f} kWh")

                if st.button("Confirm and Save Uploaded Profile"):
                    save_profile_to_session(final_daily_kwh, final_annual_demand, final_shape_24_values_kw, hourly_df_for_download, generated_8760_df_upload)

                # Add download buttons if profile is saved
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    if "hourly_profile_df" in st.session_state and st.session_state["hourly_profile_df"] is not None:
                        df_to_download_24 = st.session_state["hourly_profile_df"]
                        try:
                            excel_data_24 = to_excel(df_to_download_24)
                            st.download_button(
                                label="Download Avg 24h Profile (.xlsx)",
                                data=excel_data_24,
                                file_name="uploaded_average_daily_profile.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key="download_avg_excel_btn_upload"
                            )
                        except Exception as e:
                            st.error(f"Failed to generate 24h Excel file: {e}")
                with col_dl2:
                    if "generated_8760_profile_df" in st.session_state and st.session_state["generated_8760_profile_df"] is not None:
                         df_to_download_8760 = st.session_state["generated_8760_profile_df"]
                         try:
                             excel_data_8760 = to_excel(df_to_download_8760)
                             st.download_button(
                                 label="Download Full 8760h Profile (.xlsx)",
                                 data=excel_data_8760,
                                 file_name="uploaded_user_specific_8760h.xlsx",
                                 mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                 use_container_width=True,
                                 key="download_8760_excel_btn_upload"
                             )
                         except Exception as e:
                             st.error(f"Failed to generate 8760h Excel file: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# --- OPTION 3: Custom Profile Builder ---
elif profile_method == "Use the Demand Profile Generator":
    st.subheader("2. Build Your Custom Profile")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("##### Add Appliances")
        for category, appliances in APPLIANCES.items():
            with st.expander(f"**{category}**", expanded=(category=="Kitchen")):
                for app in appliances:
                    if st.button(f"{app['icon']} {app['name']} ({app['power_w']} W)", key=f"add_{app['name']}"):
                        add_appliance_to_schedule(app)

        st.markdown("---")
        st.markdown("##### Add a Custom Appliance")
        custom_name = st.text_input("Appliance Name", key="custom_appliance_name")
        custom_power = st.number_input("Appliance Power (W)", min_value=0, step=50, key="custom_appliance_power")
        if st.button("Add Custom Appliance to Schedule"):
            if custom_name and custom_power > 0:
                add_appliance_to_schedule({"name": custom_name, "power_w": custom_power, "icon": "⚙️"})
            else:
                st.warning("Please provide a name and a positive power value.")

    with col2:
        st.markdown("##### Configure Usage Times")
        if st.session_state.schedule:
            st.button("Clear All Appliances", on_click=reset_schedule, type="primary", use_container_width=True)
        else:
            st.info("No appliances added yet. Click on an appliance from the left to begin.")

        is_schedule_valid = True
        num_items = len(st.session_state.schedule)

        for i, item in enumerate(st.session_state.schedule):
            # Only expand the last item in the list
            is_expanded = (i == num_items - 1)

            with st.expander(f"{item['icon']} {item['name']} ({item['power_w']} W)", expanded=is_expanded):
                c1, c2 = st.columns([0.9, 0.1])
                with c1:
                    usage_text = st.text_input(
                        "Usage Times (e.g., `08:00-09:15, 19:30-20:00`)",
                        value=item.get("usage_text", "08:00-09:00"),
                        key=f"text_{item['id']}",
                        label_visibility="collapsed"
                    )
                with c2:
                    st.button("🗑️", key=f"remove_{item['id']}", on_click=remove_appliance_from_schedule, args=(item['id'],), help="Remove this appliance")

                st.session_state.schedule[i]["usage_text"] = usage_text
                parsed_ranges = parse_time_ranges_to_minutes(usage_text)

                if isinstance(parsed_ranges, str):
                    st.error(parsed_ranges, icon="⚠️")
                    is_schedule_valid = False
                    st.session_state.schedule[i]["parsed_usage"] = []
                else:
                    st.session_state.schedule[i]["parsed_usage"] = parsed_ranges

    if st.session_state.schedule and is_schedule_valid:
        st.markdown("---")
        st.subheader("3. Resulting Demand Profile")
        profile_minute_series, daily_total_kwh_gen = calculate_demand_profile_from_schedule(st.session_state.schedule) # Renamed variable

        res_col1, res_col2 = st.columns([3, 1])
        with res_col1:
            chart_data = profile_minute_series.reset_index().rename(columns={'index': 'Time'})

            label_expression = "datum.label === '00:00' && day(datum.value) > 1 ? '24:00' : datum.label"

            chart = alt.Chart(chart_data).mark_line(
                interpolate='step-after'
            ).encode(
                x=alt.X('Time:T', axis=alt.Axis(
                    title='Time of Day',
                    format='%H:%M',
                    labelExpr=label_expression
                )),
                y=alt.Y('Demand (kW):Q', title='Power (kW)'),
                tooltip=[
                    alt.Tooltip('Time:T', format='%H:%M', title='Time'),
                    alt.Tooltip('Demand (kW):Q', format='.2f', title='Power')
                ]
            ).properties(
                title='Daily Demand Profile (Generator Input)'
            )

            st.altair_chart(chart, use_container_width=True)
            st.caption("Minute-by-minute power demand (kW) based on your appliance schedule.")
        with res_col2:
            st.metric("Total Daily Energy Demand (Generator)", f"{daily_total_kwh_gen:.2f} kWh")

            if st.button("Confirm and Save Profile"):
                target_annual_demand = daily_total_kwh_gen * 365.0

                # Derive the user's 24h shape (average power kW)
                hourly_avg_power_kw_user = profile_minute_series.iloc[:-1].resample('H').mean()

                # Get the NORMALIZED 24h shape (sums to 1)
                shape_24_user_norm = hourly_avg_power_kw_user.values
                shape_sum = shape_24_user_norm.sum()
                if shape_sum > 1e-9:
                    shape_24_user_norm = shape_24_user_norm / shape_sum
                else:
                    shape_24_user_norm = np.ones(24) / 24.0 # Fallback flat shape

                # Generate the 8760h profile using the NEW logic
                final_profile_8760, final_annual_demand = generate_8760_profile_from_24h_shape(
                    shape_24_user_norm, target_annual_demand, slp_raw
                )

                # --- Derive outputs for saving ---
                final_daily_kwh = final_annual_demand / 365.0

                # Shape_24 is the average hourly pattern (POWER kW) of the final 8760 profile
                index_8760 = pd.date_range(start="2023-01-01 00:00", periods=8760, freq='H')
                series_8760_final_for_shape = pd.Series(final_profile_8760, index=index_8760)
                final_shape_24_values_kw = series_8760_final_for_shape.groupby(series_8760_final_for_shape.index.hour).mean().values

                # Create the 24h average dataframe for the main download button (showing ENERGY kWh)
                # Calculate average hourly ENERGY based on the final_shape_24_values_kw and final_daily_kwh
                # Ensure the shape kw values are normalized before scaling by daily energy
                final_shape_kw_sum = final_shape_24_values_kw.sum()
                if final_shape_kw_sum > 1e-9:
                    avg_hourly_kwh = (final_shape_24_values_kw / final_shape_kw_sum) * final_daily_kwh
                else:
                    avg_hourly_kwh = np.zeros(24)

                new_index = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)]
                profile_series_for_avg_dl = pd.Series(avg_hourly_kwh, index=new_index, name="Average Energy Demand (kWh)")
                hourly_df_for_download = profile_series_for_avg_dl.to_frame()
                hourly_df_for_download.index.name = "Time Range"

                # Create the 8760h dataframe for the new download button
                days = np.repeat(range(1, 366), 24)
                hours = [f"{h:02d}:00 - {h+1:02d}:00" for h in range(24)] * 365
                generated_8760_df = pd.DataFrame({
                    'Day': days, 'Hour Range': hours, 'Energy Demand (kWh)': final_profile_8760
                })

                save_profile_to_session(
                    final_daily_kwh, final_annual_demand, final_shape_24_values_kw, hourly_df_for_download, generated_8760_df
                )

            # Add download buttons if profiles have been saved
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                 if "hourly_profile_df" in st.session_state and st.session_state["hourly_profile_df"] is not None:
                    df_to_download_24 = st.session_state["hourly_profile_df"]
                    try:
                        excel_data_24 = to_excel(df_to_download_24)
                        st.download_button(
                            label="Download Avg 24h Profile (.xlsx)",
                            data=excel_data_24,
                            file_name="generated_average_daily_profile.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_avg_excel_btn"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate 24h Excel file: {e}")
            with col_dl2:
                if "generated_8760_profile_df" in st.session_state and st.session_state["generated_8760_profile_df"] is not None:
                    df_to_download_8760 = st.session_state["generated_8760_profile_df"]
                    try:
                        excel_data_8760 = to_excel(df_to_download_8760)
                        st.download_button(
                            label="Download Full 8760h Profile (.xlsx)",
                            data=excel_data_8760,
                            file_name="generated_user_specific_8760h.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                            key="download_8760_excel_btn"
                        )
                    except Exception as e:
                        st.error(f"Failed to generate 8760h Excel file: {e}")