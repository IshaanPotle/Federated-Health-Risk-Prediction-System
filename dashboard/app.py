"""
Streamlit dashboard for federated health risk prediction system (stub).
"""
import streamlit as st
import pandas as pd
import glob
import os
import altair as alt
import ast
import numpy as np

# --- UI/UX ENHANCEMENTS ---
st.set_page_config(page_title="Federated Health Risk Prediction Dashboard", layout="wide")

# Custom CSS for consistent #111 black background everywhere and high-contrast text
st.markdown("""
<style>
    html, body, .stApp, .block-container, .main, .css-18e3th9, .css-1d391kg, .css-1v3fvcr, .css-1l02zno, .css-1n76uvr, .css-1p05t8e, .css-1q8dd3e, .css-1offfwp, .css-1cpxqw2, .css-1v0mbdj {
        background: #111 !important;
        color: #fff !important;
    }
    .st-bb {background: #111 !important; border-radius: 8px; box-shadow: 0 2px 8px #222; padding: 1.5rem;}
    h1, h2, h3, h4, .stMarkdown, .stCaption, .stText, .st-bb, label, .css-1cpxqw2, .css-1v0mbdj, .css-1d391kg, .css-1offfwp, .css-1v3fvcr, .css-1l02zno, .css-1n76uvr, .css-1p05t8e, .css-1q8dd3e { color: #fff !important; }
    .sidebar .sidebar-content, .css-1d391kg { background: #111 !important; }
    .stSelectbox, .stRadio, .stButton, .stTextInput, .stNumberInput, .stSlider, .stCheckbox, .stFileUploader, .stDateInput, .stTimeInput, .stColorPicker, .stMarkdown, .stCaption, .stText, .st-bb { color: #fff !important; background: #111 !important; }
    .stSelectbox > div, .stRadio > div { background: #111 !important; color: #fff !important; }
    .stApp { background: #111 !important; }
    header, .css-18ni7ap, .css-1dp5vir, .css-1avcm0n, .css-1v3fvcr, .css-1l02zno, .css-1n76uvr, .css-1p05t8e, .css-1q8dd3e, .css-1offfwp, .css-1cpxqw2, .css-1v0mbdj { background: #111 !important; }
</style>
""", unsafe_allow_html=True)

# Sidebar with logo
with st.sidebar:
    st.image("https://img.icons8.com/ios-filled/100/ffffff/hospital-room.png", width=80)
    st.header("Controls")

LOG_DIR = "data/logs"
log_files = sorted(glob.glob(os.path.join(LOG_DIR, "client*_log.csv")))
shap_files = sorted(glob.glob(os.path.join(LOG_DIR, "client*_shap.csv")))
client_names = [f"Client {i+1}" for i in range(len(log_files))]

if not log_files:
    st.warning("No client logs found. Please run the federated learning pipeline first.")
else:
    # --- DROPDOWNS AT TOP ---
    col1, col2, col3 = st.columns([2,2,2])
    with col1:
        selected_client = st.selectbox("Select client", client_names, key="main_client")
    with col2:
        selected_metric = st.selectbox("Select metric", ["Loss", "Accuracy"], key="main_metric")
    with col3:
        shap_client = st.selectbox("Select client for SHAP", client_names, key="shap_client")
    client_idx = client_names.index(selected_client)
    log_file = log_files[client_idx]
    
    # Check if log file exists and has content
    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        st.warning(f"No training data available for {selected_client}. The federated learning may still be in progress.")
        st.info("Please wait for the training to complete or check if the clients are running properly.")
    else:
        try:
            # Debug: Show file info
            # st.info(f"Reading log file: {log_file} (size: {os.path.getsize(log_file)} bytes)")
            df = pd.read_csv(log_file)
            # Debug: Show DataFrame info
            # st.info(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            # if not df.empty:
            #     st.info(f"First few rows: {df.head().to_dict()}")
            
            if df.empty:
                st.warning(f"Training data for {selected_client} is empty. Please wait for training to complete.")
            else:
                # --- THEME SETTINGS (always #111) ---
                chart_bg = "#111"
                font_color = "#fff"
                grid_color = "#333"

                # --- TITLE ---
                st.markdown(f"<h1 style='color:{font_color};font-size:2.8rem;'>🏥 Federated Health Risk Prediction Dashboard</h1>", unsafe_allow_html=True)
                st.markdown("<hr style='margin:1.5rem 0; border-color: #222;'>", unsafe_allow_html=True)

                # --- TRAINING ROUNDS CHART ---
                st.markdown(f"<h2 style='color:{font_color};'>Training Rounds: {selected_client} - {selected_metric}</h2>", unsafe_allow_html=True)
                st.caption("Shows the selected metric for the chosen client across federated training rounds.")
                y_col = selected_metric.lower()
                
                # Validate data before plotting
                if y_col not in df.columns:
                    st.error(f"Column '{y_col}' not found in the data. Available columns: {list(df.columns)}")
                else:
                    # Filter out invalid values
                    valid_df = df[np.isfinite(df[y_col])].copy()
                    if valid_df.empty or not np.isfinite(valid_df[y_col]).any():
                        st.warning(f"No valid {selected_metric} data available for {selected_client}.")
                        st.stop()
                    chart = alt.Chart(valid_df).mark_line(point=alt.OverlayMarkDef(color="#e76f51", size=80)).encode(
                        x=alt.X('round:Q', title='Round', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                        y=alt.Y(f'{y_col}:Q', title=selected_metric, axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                        tooltip=['round', y_col]
                    ).configure(
                        background=chart_bg,
                        axis=alt.AxisConfig(gridColor=grid_color),
                        title=alt.TitleConfig(color=font_color)
                    ).properties(width=700, height=350)
                    st.altair_chart(chart, use_container_width=True)
                st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

                # --- AVERAGE METRIC ACROSS CLIENTS ---
                st.markdown(f"<h2 style='color:{font_color};'>Average {selected_metric} Across Clients</h2>", unsafe_allow_html=True)
                st.caption(f"Average {selected_metric} across all clients for each round.")
                
                # Read all log files with error handling
                all_dfs = []
                for f in log_files:
                    try:
                        if os.path.exists(f) and os.path.getsize(f) > 0:
                            df_temp = pd.read_csv(f)
                            if not df_temp.empty:
                                all_dfs.append(df_temp)
                    except Exception as e:
                        st.warning(f"Error reading log file {f}: {str(e)}")
                        continue
                
                if all_dfs:
                    min_rounds = min(len(df) for df in all_dfs)
                    
                    # Calculate average with validation
                    avg_metric = []
                    for i in range(min_rounds):
                        values = []
                        for df in all_dfs:
                            if y_col in df.columns and i < len(df) and np.isfinite(df.loc[i, y_col]):
                                values.append(df.loc[i, y_col])
                        if values:
                            avg_metric.append(sum(values) / len(values))
                        else:
                            avg_metric.append(np.nan)
                    
                    # Filter out invalid averages
                    valid_avgs = [(i, val) for i, val in enumerate(avg_metric) if np.isfinite(val)]
                    
                    if valid_avgs:
                        avg_df = pd.DataFrame({"round": [i for i, _ in valid_avgs], f"avg_{y_col}": [val for _, val in valid_avgs]})
                        if not avg_df.empty and np.isfinite(avg_df[f"avg_{y_col}"].values).any():
                            avg_chart = alt.Chart(avg_df).mark_line(point=alt.OverlayMarkDef(color="#2a9d8f", size=80)).encode(
                                x=alt.X('round:Q', title='Round', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                                y=alt.Y(f'avg_{y_col}:Q', title=f'Average {selected_metric}', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                                tooltip=['round', f'avg_{y_col}']
                            ).configure(
                                background=chart_bg,
                                axis=alt.AxisConfig(gridColor=grid_color),
                                title=alt.TitleConfig(color=font_color)
                            ).properties(width=700, height=350)
                            st.altair_chart(avg_chart, use_container_width=True)
                        else:
                            st.warning(f"No valid average {selected_metric} data available across clients.")
                            st.stop()
                    else:
                        st.warning(f"No valid average {selected_metric} data available across clients.")
                        st.stop()
                else:
                    st.warning("No valid log files found. Please wait for training to complete.")
                    st.stop()
                st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

                # --- PERSONALIZED RISK SCORES ---
                st.markdown(f"<h2 style='color:{font_color};'>Personalized Risk Scores (Final Accuracy)</h2>", unsafe_allow_html=True)
                st.caption("Shows the final accuracy for each client as a bar chart.")
                
                # Calculate risk scores with validation
                risk_scores = {}
                for i, df in enumerate(all_dfs):
                    if "accuracy" in df.columns and len(df) > 0:
                        final_acc = df["accuracy"].iloc[-1]
                        if np.isfinite(final_acc):
                            risk_scores[client_names[i]] = final_acc
                
                if risk_scores:
                    risk_df = pd.DataFrame({"Client": list(risk_scores.keys()), "Final Accuracy": list(risk_scores.values())})
                    if not risk_df.empty and np.isfinite(risk_df['Final Accuracy'].values).any():
                        bar_chart = alt.Chart(risk_df).mark_bar(color="#e76f51").encode(
                            x=alt.X('Client:N', title='Client', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                            y=alt.Y('Final Accuracy:Q', title='Final Accuracy', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                            tooltip=['Client', 'Final Accuracy']
                        ).configure(
                            background=chart_bg,
                            axis=alt.AxisConfig(gridColor=grid_color),
                            title=alt.TitleConfig(color=font_color)
                        ).properties(width=700, height=350)
                        st.altair_chart(bar_chart, use_container_width=True)
                    else:
                        st.warning("No valid final accuracy data available for any client.")
                        st.stop()
                else:
                    st.warning("No valid final accuracy data available for any client.")
                    st.stop()
                st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

                # --- SHAP FEATURE IMPORTANCES ---
                st.markdown(f"<h2 style='color:{font_color};'>Feature Importances (SHAP) - {shap_client}</h2>", unsafe_allow_html=True)
                st.caption("Shows the mean absolute SHAP value for each feature for the selected client.")
                if shap_files:
                    shap_idx = client_names.index(shap_client)
                    shap_file = shap_files[shap_idx]
                    
                    try:
                        # First try to read as regular CSV
                        try:
                            shap_df = pd.read_csv(shap_file)
                            if not shap_df.empty and len(shap_df.columns) == 5:
                                # Regular CSV format
                                shap_values = shap_df.iloc[0].values
                                feature_names = shap_df.columns.tolist()
                            else:
                                raise ValueError("Not in expected CSV format")
                        except:
                            # Handle malformed array format
                            with open(shap_file, 'r') as f:
                                lines = f.readlines()
                            
                            if len(lines) >= 2:
                                feature_names = lines[0].strip().split(',')
                                
                                # Parse the malformed array data
                                all_shap_values = []
                                for line in lines[1:]:
                                    try:
                                        # Split by '],[' to separate arrays
                                        arrays_str = line.strip().split('],[')
                                        arrays_str = [arr.replace('[', '').replace(']', '') for arr in arrays_str]
                                        
                                        # Parse each array
                                        for arr_str in arrays_str:
                                            if arr_str.strip():
                                                values = [float(x.strip()) for x in arr_str.split(',') if x.strip()]
                                                if len(values) == len(feature_names):
                                                    all_shap_values.append(values)
                                    except Exception as e:
                                        continue
                                
                                if all_shap_values:
                                    # Convert to numpy array and compute mean absolute values
                                    all_shap_values = np.array(all_shap_values)
                                    shap_values = np.abs(all_shap_values).mean(axis=0)
                                else:
                                    raise ValueError("Could not parse SHAP values")
                            else:
                                raise ValueError("Invalid SHAP file format")
                        
                        # Create DataFrame for plotting
                        plot_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": shap_values
                        })
                        
                        # Filter out any invalid values
                        plot_df = plot_df[np.isfinite(plot_df['Importance'])]
                        
                        if not plot_df.empty and np.isfinite(plot_df['Importance'].values).any():
                            shap_chart = alt.Chart(plot_df).mark_bar(color="#264653").encode(
                                x=alt.X('Feature:N', title='Feature', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                                y=alt.Y('Importance:Q', title='Mean |SHAP value|', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                                tooltip=['Feature', 'Importance']
                            ).configure(
                                background=chart_bg,
                                axis=alt.AxisConfig(gridColor=grid_color),
                                title=alt.TitleConfig(color=font_color)
                            ).properties(width=700, height=350)
                            st.altair_chart(shap_chart, use_container_width=True)
                        else:
                            st.warning("No valid SHAP values found for this client.")
                            st.stop()
                            
                    except Exception as e:
                        st.error(f"Error reading SHAP data: {str(e)}")
                        st.info("SHAP data may be in an unexpected format. Please regenerate the SHAP files.")
                        st.stop()
                else:
                    st.info("No SHAP logs found. Run the clients to generate SHAP feature importances.")
                    st.stop()
                st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

                # --- RISK DRIFT OVER TIME ---
                st.markdown(f"<h2 style='color:{font_color};'>Risk Drift Over Time (Average Loss)</h2>", unsafe_allow_html=True)
                st.caption("Shows the average loss across all clients for each round, visualized as an area chart.")
                
                # Calculate average loss with validation
                avg_loss = []
                for i in range(min_rounds):
                    values = []
                    for df in all_dfs:
                        if "loss" in df.columns and i < len(df) and np.isfinite(df.loc[i, "loss"]):
                            values.append(df.loc[i, "loss"])
                    if values:
                        avg_loss.append(sum(values) / len(values))
                    else:
                        avg_loss.append(np.nan)
                
                # Filter out invalid averages
                valid_losses = [(i, val) for i, val in enumerate(avg_loss) if np.isfinite(val)]
                
                if valid_losses:
                    drift_df = pd.DataFrame({"round": [i for i, _ in valid_losses], "avg_loss": [val for _, val in valid_losses]})
                    if not drift_df.empty and np.isfinite(drift_df['avg_loss'].values).any():
                        drift_chart = alt.Chart(drift_df).mark_area(color="#a8dadc", opacity=0.6).encode(
                            x=alt.X('round:Q', title='Round', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                            y=alt.Y('avg_loss:Q', title='Average Loss', axis=alt.Axis(labelColor=font_color, titleColor=font_color)),
                            tooltip=['round', 'avg_loss']
                        ).configure(
                            background=chart_bg,
                            axis=alt.AxisConfig(gridColor=grid_color),
                            title=alt.TitleConfig(color=font_color)
                        ).properties(width=700, height=300)
                        st.altair_chart(drift_chart, use_container_width=True)
                    else:
                        st.warning("No valid average loss data available across clients.")
                        st.stop()
                else:
                    st.warning("No valid average loss data available across clients.")
                    st.stop()

                st.markdown("<hr style='margin:2rem 0; border-color: #222;'>", unsafe_allow_html=True)
                st.caption("Built with ❤️ using Streamlit, Flower, and PyTorch. UI enhanced by Altair.")
        except Exception as e:
            st.error(f"Error reading training data: {str(e)}")
            st.info("Please wait for the federated learning training to complete and generate log files.") 