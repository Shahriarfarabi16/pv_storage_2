import streamlit as st

# --------------------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------------------
st.set_page_config(page_title="Welcome", page_icon="👋", layout="wide")

st.title("☀️ Welcome to the PV-Storage Evaluation Tool!")

# --- Main Instructions Block ---
st.markdown(
    """
    <div style="
        font-size: 1.2rem; 
        line-height: 1.8; 
        padding: 25px; 
        border: 2px solid #D2B48C; 
        border-radius: 12px; 
        background-color: #fdfdfd;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    ">

    <h3 style="font-weight: 600;">Follow these simple steps to get started with your analysis:</h3>

    <ol style="margin-left: 20px; padding-left: 10px;">
        <li style="margin-bottom: 12px;">
            First, go to the <strong>Irradiation</strong> page and type your location data and find the value of Global Tilted Irradiation.
        </li>
        <li style="margin-bottom: 12px;">
            Then, go to the <strong>Electricity Demand Profile</strong> page and choose your preferred method to provide the electricity demand profile of your house. Don't forget to click the <strong>Confirm and Save Profile</strong> button before leaving.
        </li>
        <li style="margin-bottom: 12px;">
            Go to the <strong>PV Calculation</strong> page to see the simulation result and play around with the parameters.
        </li>
    </ol>

    <h2 style="font-weight: 700; text-align: center; margin-top: 30px; color: #333;">
        Enjoy the simulation!
    </h2>

    </div>
    """,
    unsafe_allow_html=True
)

# --- Add some spacing at the bottom ---
st.markdown("<br><br>", unsafe_allow_html=True)

# --- Developer Profile Section ---
st.markdown("---")
st.subheader("About the Developer")

# --- NEW STYLED CONTAINER FOR PROFILE ---
st.markdown(
    """
    <div style="
        font-size: 1.1rem; 
        line-height: 1.7; 
        padding: 20px 25px; 
        border: 1px solid #ddd; 
        border-radius: 12px; 
        background-color: #fafafa;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    ">

    <!-- NAME IS NOW ALIGNED TO THE LEFT -->
    <h2 style="text-align: left; font-weight: 600; color: #333;">Md Shahriar Farabi</h2>

    <strong>Educational Qualification:</strong>
    <ul>
        <li>M.Sc. in Renewable Energy Systems (on-going)
            <br>
            <em style="font-size: 0.95em; color: #555;">Technische Hochschule Ingolstadt, Germany</em>
        </li>
        <li>B.Sc. in Mechanical Engineering
            <br>
            <em style="font-size: 0.95em; color: #555;">Ahsanullah University of Science and Technology, Dhaka, Bangladesh</em>
        </li>
    </ul>

    <strong>Contact:</strong>
    <ul style="list-style-type: none; padding-left: 0;">
        <li>📞 +49 155 10115562 (What's app)</li>
        <li>📍 Ingolstadt, Germany</li>
        <li>📧 <a href="mailto:shahriarfarabiaust@gmail.com">shahriarfarabiaust@gmail.com</a></li>
        <li>🔗 <a href="https://www.linkedin.com/in/md-shahriar-farabi/" target="_blank">LinkedIn Profile</a></li>
    </ul>

    </div>
    """,
    unsafe_allow_html=True
)