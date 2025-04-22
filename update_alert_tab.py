import streamlit as st
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
from uuid import uuid4

# --- Função de atualização de alertas do usuário ---
def display_alert_update_tab(sheet_id):
    with st.expander("🔄 Update an Alert"):
        if "user" not in st.session_state:
            st.info("Please log in to view and update your submitted alerts.")
            return

        scope = ["https://www.googleapis.com/auth/spreadsheets"]
        credentials = Credentials.from_service_account_info(st.secrets["gcp_service_account"], scopes=scope)
        client = gspread.authorize(credentials)
        sheets = client.open_by_key(sheet_id)

        try:
            df_alerts = pd.DataFrame(sheets.worksheet("Alerts").get_all_records())
        except Exception as e:
            st.error(f"❌ Failed to load Alerts: {e}")
            return

        user_alerts = df_alerts[df_alerts["Author"] == st.session_state["user"]]

        if user_alerts.empty:
            st.info("You haven't submitted any alerts yet.")
            return

        selected_alert = st.selectbox("Select one of your alerts:", user_alerts["Title"].tolist())
        alert_data = user_alerts[user_alerts["Title"] == selected_alert].iloc[0]
        alert_id = alert_data["ID"]

        st.markdown("### ✏️ Edit Alert Fields")
        with st.form("edit_alert_fields"):
            new_title = st.text_input("Title", value=alert_data["Title"])
            new_description = st.text_area("Description", value=alert_data["Description"])
            categories = ["Species", "Country", "Marketplace", "Operation", "Policy", "Other"]
            risk_levels = ["Low", "Medium", "High"]
            new_category = st.selectbox("Category", categories, index=categories.index(alert_data["Category"]) if alert_data["Category"] in categories else 0)
            new_risk = st.selectbox("Risk Level", risk_levels, index=risk_levels.index(alert_data["Risk Level"]) if alert_data["Risk Level"] in risk_levels else 0)
            new_species = st.text_input("Species", value=alert_data["Species"])
            new_country = st.text_input("Country", value=alert_data["Country"])
            new_source = st.text_input("Source Link", value=alert_data["Source Link"])
            make_public = st.checkbox("Make this alert public?", value=str(alert_data["Public"]).strip().upper() == "TRUE")

            submitted_edit = st.form_submit_button("📏 Save Changes")

            if submitted_edit:
                try:
                    ws = sheets.worksheet("Alerts")
                    row_number = df_alerts[df_alerts["ID"] == alert_id].index[0] + 2
                    ws.update(f"D{row_number}:K{row_number}", [[
                        new_title, new_description, new_category, new_species,
                        new_country, new_risk, new_source
                    ]])
                    ws.update(f"L{row_number}", [[str(make_public).upper()]])
                    st.success("✅ Alert updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Failed to update alert: {e}")

        st.markdown("### 📌 Add a Timeline Update")
        with st.form("form_add_timeline"):
            timeline_text = st.text_area("Describe what happened next (e.g. sent to authorities, animal released, etc.)")
            submit_update = st.form_submit_button("➕ Submit Update")

            if submit_update and timeline_text.strip():
                try:
                    update_ws = sheets.worksheet("Alert Updates")
                except gspread.exceptions.WorksheetNotFound:
                    update_ws = sheets.add_worksheet(title="Alert Updates", rows="1000", cols="4")
                    update_ws.append_row(["Alert ID", "Timestamp", "User", "Update Text"])

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                update_ws.append_row([alert_id, timestamp, st.session_state["user"], timeline_text.strip()])
                st.success("✅ Timeline update added!")
                st.rerun()
