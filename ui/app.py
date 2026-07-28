import streamlit as st
import requests
import os
import pandas as pd
from PIL import Image
from io import BytesIO

# Use the Docker network name for the API when running in compose, otherwise localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Garbage Classification ML Pipeline", layout="wide")

st.title("♻️ Garbage Classification ML Pipeline")

tabs = st.tabs(["Prediction", "Retrain & Data Upload", "Visualizations", "System Status"])

with tabs[0]:
    st.header("Make a Prediction")
    uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    if response.status_code == 200:
                        res = response.json()
                        st.success(f"Prediction: **{res['predicted_class']}** (Confidence: {res['confidence']:.2%})")
                        st.json(res["probabilities"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")

with tabs[1]:
    st.header("Upload Data & Retrain")
    st.write("Upload a ZIP file containing class folders (e.g., `cardboard`, `glass`) with images to merge into the dataset.")
    bulk_file = st.file_uploader("Upload Bulk Data (ZIP)", type=["zip"])
    if bulk_file is not None:
        if st.button("Upload & Merge"):
            with st.spinner("Uploading and merging..."):
                try:
                    files = {"file": (bulk_file.name, bulk_file.getvalue(), "application/zip")}
                    response = requests.post(f"{API_URL}/upload-bulk", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")
                    
    st.divider()
    st.subheader("Trigger Retraining")
    st.write("This will rebuild the train/test splits from all available raw data and trigger a MobileNetV2 fine-tuning.")
    if st.button("Retrain Model"):
        with st.spinner("Sending retrain trigger..."):
            try:
                response = requests.post(f"{API_URL}/retrain")
                if response.status_code == 200:
                    st.success("Retraining triggered successfully! Check 'System Status' tab for progress.")
                elif response.status_code == 409:
                    st.warning("Retraining is already in progress.")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect: {e}")

with tabs[2]:
    st.header("Data Visualizations")
    st.write("These visualizations represent features and interpretations from the dataset.")
    vis_dir = "/app/data/visualizations" # Docker path
    if not os.path.exists(vis_dir):
        # Fallback for local testing
        vis_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "visualizations")
        
    if os.path.exists(vis_dir):
        cols = st.columns(2)
        images = [f for f in os.listdir(vis_dir) if f.endswith(".png") or f.endswith(".jpg")]
        for i, img_name in enumerate(images):
            with cols[i % 2]:
                st.image(os.path.join(vis_dir, img_name), caption=img_name, use_container_width=True)
        if not images:
            st.info("No visualizations found in the directory.")
    else:
        st.info("Visualization directory not found. Run the Jupyter Notebook to generate EDA plots.")

with tabs[3]:
    st.header("System Status & Metrics")
    if st.button("Refresh Status", type="primary"):
        try:
            uptime_res = requests.get(f"{API_URL}/uptime")
            metrics_res = requests.get(f"{API_URL}/metrics")
            
            if uptime_res.status_code == 200:
                uptime = uptime_res.json()["uptime_seconds"]
                st.metric("Uptime", f"{int(uptime)} seconds")
                
            if metrics_res.status_code == 200:
                m = metrics_res.json()
                st.write(f"**Is Retraining:** {'Yes ⏳' if m['is_retraining'] else 'No ✅'}")
                st.write(f"**Images Merged:** {m['merged_images_count']}")
                st.write(f"**Was Deployed After Last Retrain:** {m['was_deployed']}")
                
                if m["last_retrain_metrics"]:
                    st.subheader("Last Retrain Metrics")
                    st.json(m["last_retrain_metrics"])
        except Exception as e:
            st.error(f"Failed to fetch status: {e}")
