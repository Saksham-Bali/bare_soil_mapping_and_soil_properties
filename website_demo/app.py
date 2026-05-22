import streamlit as st
import pydeck as pdk
import json
import os
import base64
from datetime import date

# --- Page Configuration ---
st.set_page_config(page_title="Bare Soil Mask Viewer", layout="wide")

# --- Sample Data for Dropdowns ---
LOCATION_DATA = {
    "Haryana": ["Ambala", "Karnal", "Rohtak", "Hisar", "Gurugram", "Faridabad"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Bikaner", "Kota"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Karnataka": ["Bengaluru", "Mysuru", "Mangaluru", "Hubballi", "Belagavi"],
    "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Bathinda"]
}

def load_bounds():
    if not os.path.exists("bounds.json"):
        return None
    with open("bounds.json", "r") as f:
        return json.load(f)

@st.cache_data
def get_image_data_uri(filepath, file_mtime):
    """Convert local image to base64 data URI for PyDeck BitmapLayer"""
    with open(filepath, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded_string}"

def render_landing_page():
    # Show centered layout
    st.markdown("<h1 style='text-align: center;'>🌾 Interactive Bare Soil Mask Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Visualize high-resolution Sentinel-2 bare soil masks.</h4>", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.write("")
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("### Select Region to Analyze")
        # Cascading Dropdowns
        states = list(LOCATION_DATA.keys())
        default_state_idx = states.index("Haryana") if "Haryana" in states else 0
        selected_state = st.selectbox("Select State", states, index=default_state_idx)

        districts = LOCATION_DATA[selected_state]
        default_idx = districts.index("Ambala") if "Ambala" in districts else 0
        selected_district = st.selectbox("Select District", districts, index=default_idx)

        # Date Range
        date_range = st.date_input(
            "Select Date Range",
            value=(date(2024, 1, 1), date(2024, 1, 31)),
            min_value=date(2020, 1, 1),
            max_value=date.today()
        )
        
        st.write("")
        if st.button("Generate Map", type="primary"):
            # Save selections to state
            st.session_state.selected_state = selected_state
            st.session_state.selected_district = selected_district
            # Move to dashboard state
            st.session_state.app_state = "map"
            st.rerun()

def render_dashboard():
    # --- Sidebar ---
    st.sidebar.header("Data Selection")
    
    # Pre-fill sidebar with previously selected state
    states = list(LOCATION_DATA.keys())
    default_state_idx = states.index(st.session_state.get('selected_state', 'Haryana')) if st.session_state.get('selected_state', 'Haryana') in states else 0
    selected_state = st.sidebar.selectbox("Select State", states, index=default_state_idx)

    # Sync state so if they change it in sidebar, the district dropdown updates correctly
    districts = LOCATION_DATA[selected_state]
    default_dist_val = st.session_state.get('selected_district', 'Ambala')
    
    # If the user changed the state, the old district might not be in the new state's districts
    if default_dist_val in districts:
        default_dist_idx = districts.index(default_dist_val)
    else:
        default_dist_idx = 0
        
    selected_district = st.sidebar.selectbox("Select District", districts, index=default_dist_idx)
    
    # Save back to state to keep it persistent
    st.session_state.selected_state = selected_state
    st.session_state.selected_district = selected_district

    # Date Range
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(date(2024, 1, 1), date(2024, 1, 31)),
        min_value=date(2020, 1, 1),
        max_value=date.today()
    )

    if st.sidebar.button("⬅ Back to Selection"):
        st.session_state.app_state = "selection"
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Map Controls")

    # Layer Toggles and Opacity Sliders
    st.sidebar.subheader("Overlay Settings")
    show_rgb = st.sidebar.checkbox("Show True Color Imagery", value=True)
    if show_rgb:
        rgb_opacity = st.sidebar.slider("True Color Opacity", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    else:
        rgb_opacity = 0
        
    show_mask = st.sidebar.checkbox("Show Bare Soil Mask", value=True)
    if show_mask:
        mask_opacity = st.sidebar.slider("Bare Soil Opacity", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
    else:
        mask_opacity = 0

    st.sidebar.markdown("---")
    st.sidebar.header("Export Data")
    st.sidebar.caption("Export a 10-band GeoTIFF containing only bare soil pixels.")
    if st.sidebar.button("Export Bare Soil GeoTIFF", type="primary"):
        with st.spinner("Generating multi-band GeoTIFF... this may take up to a minute."):
            try:
                import export_tiff
                export_tiff.generate_bare_soil_tiff()
                st.sidebar.success("The file has been exported")
            except Exception as e:
                st.sidebar.error(f"Failed to export: {e}")

    # --- Data Loading ---
    bounds = load_bounds()

    if bounds is None:
        st.warning("⚠️ Pre-processed bounds.json not found. Please run the data processing script first.")
        return

    # --- Main Screen Visualization ---
    st.title(f"{selected_district}, {selected_state}")
    
    if selected_district != "Jalandhar" or selected_state != "Punjab":
        st.info("Note: Currently displaying the pre-processed sample dataset for Jalandhar, March 2024.")

    # Calculate center
    center_lat = (bounds["min_lat"] + bounds["max_lat"]) / 2
    center_lon = (bounds["min_lon"] + bounds["max_lon"]) / 2

    # PyDeck Layers
    layers = []
    
    if os.path.exists("rgb_overlay.png"):
        rgb_mtime = os.path.getmtime("rgb_overlay.png")
        rgb_layer = pdk.Layer(
            "BitmapLayer",
            id="rgb-layer",
            image=pdk.types.String(get_image_data_uri("rgb_overlay.png", rgb_mtime)),
            bounds=[bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"]],
            opacity=rgb_opacity,
            visible=show_rgb,
            pickable=False
        )
        layers.append(rgb_layer)

    if os.path.exists("mask_overlay.png"):
        mask_mtime = os.path.getmtime("mask_overlay.png")
        mask_layer = pdk.Layer(
            "BitmapLayer",
            id="mask-layer",
            image=pdk.types.String(get_image_data_uri("mask_overlay.png", mask_mtime)),
            bounds=[bounds["min_lon"], bounds["min_lat"], bounds["max_lon"], bounds["max_lat"]],
            opacity=mask_opacity,
            visible=show_mask,
            pickable=False
        )
        layers.append(mask_layer)

    # Initial View State
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=9.5,
        pitch=0
    )

    # Render PyDeck Chart
    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_style='light', # Clean base map styling
            tooltip=False
        ),
        width="stretch",
        height=760
    )

def main():
    # Initialize application state
    if "app_state" not in st.session_state:
        st.session_state.app_state = "selection"

    # Route to correct layout
    if st.session_state.app_state == "selection":
        render_landing_page()
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
