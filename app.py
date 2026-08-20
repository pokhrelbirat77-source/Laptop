import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved machine learning pipeline
@st.cache_resource
def load_model():
    model_path = "best_laptop_model.pkl"
    if os.path.exists(model_path):
        model_obj = joblib.load(model_path)
        
        # Patch SimpleImputer for scikit-learn version compatibility (fixes '_fill_dtype' errors)
        def patch_imputer(obj):
            from sklearn.impute import SimpleImputer
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.compose import TransformedTargetRegressor

            if isinstance(obj, SimpleImputer):
                if not hasattr(obj, '_fill_dtype'):
                    obj._fill_dtype = obj.statistics_.dtype
            elif isinstance(obj, Pipeline):
                for name, step in obj.steps:
                    patch_imputer(step)
            elif isinstance(obj, ColumnTransformer):
                for name, transformer, columns in obj.transformers_:
                    patch_imputer(transformer)
            elif isinstance(obj, TransformedTargetRegressor):
                patch_imputer(obj.regressor_)

        patch_imputer(model_obj)
        return model_obj
    else:
        st.error(f"Model file '{model_path}' not found! Please run the training notebook first to generate the model.")
        return None

model = load_model()

# Header Section
st.title("💻 Laptop Price Predictor & Spec Evaluator")
st.write(
    "Estimate the retail price of a laptop based on its hardware specifications. "
    "This application uses an optimized, cross-validated XGBoost regressor trained with log-transformed targets."
)

st.divider()

if model is not None:
    # Sidebar for Laptop Configurator
    st.sidebar.header("🔧 Laptop Configuration")
    
    # Brand
    brands = ['HP', 'Lenovo', 'Asus', 'Dell', 'Acer', 'Apple', 'MSI', 'Samsung', 'Gigabyte', 'Infinix', 'Other']
    brand = st.sidebar.selectbox("Laptop Brand", brands, index=0)
    
    # RAM Specs
    st.sidebar.subheader("🧠 Memory & RAM")
    ram_gb = st.sidebar.slider("RAM Capacity (GB)", min_value=2, max_value=64, value=8, step=2)
    ram_types = ['DDR4', 'DDR5', 'LPDDR5', 'LPDDR4X', 'LPDDR4', 'DDR3', 'LPDDR5X']
    ram_type = st.sidebar.selectbox("RAM Type", ram_types, index=0)
    
    # Storage Specs
    st.sidebar.subheader("💾 Storage")
    rom_gb = st.sidebar.selectbox("Storage Capacity (GB)", [64, 128, 256, 512, 1024, 2048], index=3)
    rom_type = st.sidebar.radio("Storage Type", ['SSD', 'HDD'], index=0)
    
    # Display Specs
    st.sidebar.subheader("🖥️ Display Details")
    display_size = st.sidebar.slider("Screen Size (Inches)", min_value=10.0, max_value=18.0, value=15.6, step=0.1)
    
    resolutions = {
        "Full HD (1920 x 1080)": (1920, 1080),
        "HD (1366 x 768)": (1366, 768),
        "WUXGA (1920 x 1200)": (1920, 1200),
        "Quad HD / 2K (2560 x 1440)": (2560, 1440),
        "QHD+ (2560 x 1600)": (2560, 1600),
        "4K Ultra HD (3840 x 2160)": (3840, 2160),
        "Custom Resolution": None
    }
    selected_res = st.sidebar.selectbox("Screen Resolution Preset", list(resolutions.keys()), index=0)
    
    if selected_res == "Custom Resolution":
        res_width = st.sidebar.number_input("Resolution Width (pixels)", min_value=800, max_value=4000, value=1920, step=1)
        res_height = st.sidebar.number_input("Resolution Height (pixels)", min_value=600, max_value=3000, value=1080, step=1)
    else:
        res_width, res_height = resolutions[selected_res]
        
    # Processor & CPU
    st.sidebar.subheader("⚙️ Processor (CPU)")
    cpu_brand = st.sidebar.selectbox("Processor Brand", ['Intel', 'AMD', 'Apple', 'MediaTek', 'Other'], index=0)
    cpu_cores = st.sidebar.slider("CPU Cores", min_value=2, max_value=24, value=6, step=2)
    cpu_threads = st.sidebar.slider("CPU Threads", min_value=2, max_value=32, value=12, step=2)
    
    # GPU
    st.sidebar.subheader("🎮 Graphics (GPU)")
    gpu_brand = st.sidebar.selectbox("GPU Brand", ['Intel', 'AMD', 'NVIDIA', 'Apple', 'Other'], index=0)
    gpu_vram = st.sidebar.slider("Dedicated VRAM (GB)", min_value=0, max_value=16, value=0, step=2)
    
    # OS & Warranty
    st.sidebar.subheader("🛡️ Software & Warranty")
    os_options = ['Windows 11', 'Mac OS', 'Windows 10', 'Chrome OS', 'DOS', 'Ubuntu', 'Android', 'Other']
    os_choice = st.sidebar.selectbox("Operating System", os_options, index=0)
    warranty = st.sidebar.selectbox("Warranty Length (Years)", [0, 1, 2, 3], index=1)
    
    # Specification Rating
    st.sidebar.subheader("📊 Performance Index")
    spec_rating = st.sidebar.slider("Specification Rating (out of 100)", min_value=30.0, max_value=100.0, value=70.0, step=1.0)
    
    # MAIN PAGE PREDICTION VIEW
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Selected Specifications Summary")
        spec_df = pd.DataFrame({
            "Specification": [
                "Brand", "RAM Size", "RAM Type", "Storage", "Storage Type", 
                "Screen Size", "Resolution", "CPU Brand", "CPU Cores/Threads",
                "GPU Brand", "Dedicated VRAM", "Operating System", "Warranty", "Performance Rating"
            ],
            "Value": [
                brand, f"{ram_gb} GB", ram_type, f"{rom_gb} GB", rom_type,
                f"{display_size} inches", f"{res_width} x {res_height}", cpu_brand, f"{cpu_cores} Cores / {cpu_threads} Threads",
                gpu_brand, f"{gpu_vram} GB", os_choice, f"{warranty} Year(s)", f"{spec_rating} / 100"
            ]
        })
        st.dataframe(spec_df, use_container_width=True, hide_index=True)
        
    with col2:
        st.subheader("Predicted Market Value")
        
        # Build features DataFrame matching preprocessing columns exactly
        input_data = pd.DataFrame([{
            'brand': brand,
            'spec_rating': float(spec_rating),
            'display_size': float(display_size),
            'resolution_width': float(res_width),
            'resolution_height': float(res_height),
            'OS': f"{os_choice} OS" if os_choice in ['Windows 11', 'Windows 10'] else os_choice,
            'warranty': int(warranty),
            'Ram_GB': int(ram_gb),
            'ROM_GB': int(rom_gb),
            'Ram_type': ram_type,
            'ROM_type': rom_type,
            'CPU_cores': float(cpu_cores),
            'CPU_threads': float(cpu_threads),
            'CPU_brand': cpu_brand,
            'GPU_brand': gpu_brand,
            'GPU_vram_GB': int(gpu_vram)
        }])
        
        # Predict price
        try:
            # Predict using pipeline (automatically scales, encodes, predicts, and back-transforms log)
            predicted_log_price = model.predict(input_data)
            price_val = predicted_log_price[0]
            
            # Format and display
            st.metric(
                label="Estimated Price (INR)",
                value=f"₹{price_val:,.2f}"
            )
            
            # Add some context cards
            st.success("🎉 Prediction generated successfully!")
            
            st.info(
                "💡 **How to interpret:** This price is estimated based on the historical dataset of retail laptop prices. "
                "Premium brands (like Apple) or high performance metrics (like dedicated NVIDIA GPU and high CPU cores) "
                "exert the highest positive coefficients on the predicted valuation."
            )
        except Exception as e:
            st.error(f"Prediction failed during feature pipeline: {e}")
            st.write("Ensure your input features match the model schema.")

else:
    st.info("Please generate the model file `best_laptop_model.pkl` to activate this configurator dashboard.")
