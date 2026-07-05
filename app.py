import streamlit as st
import requests
import os
from api.reports import generate_csv_report, generate_pdf_report

st.set_page_config(page_title="QC-Dashboard | Nornickel", page_icon="🔬", layout="wide")

st.title("Automated Ore Microstructure Analysis")
st.markdown("Upload a panoramic OM image to detect talc inclusions and classify the ore.")

API_URL = "http://127.0.0.1:8000"
WORKSPACE_DIR = os.path.join("api", "workspace")

uploaded_file = st.file_uploader("Select TIFF/PNG/JPG file", type=["tif", "tiff", "png", "jpg", "jpeg"])

if st.button("Start Analysis", type="primary") and uploaded_file:
    with st.spinner("Analyzing image... Please wait."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(f"{API_URL}/analyze", files=files)
            if response.status_code == 200:
                result = response.json()
                task_id = result["task_id"]

                st.divider()
                st.subheader("Results")

                col1, col2, col3 = st.columns(3)
                col1.metric("Ore Class", result["ore_class"])
                col2.metric("Talc Percent", f"{result['talc_percent']}%")
                col3.metric("Classifier Confidence", f"{result['clf_confidence']}%")

                img_path = os.path.join(WORKSPACE_DIR, result["result_image_path"])
                if os.path.exists(img_path):
                    st.image(img_path, caption="Segmentation Result (Blue = Talc)", use_column_width=True)

                csv_path = os.path.join(WORKSPACE_DIR, f"report_{task_id}.csv")
                pdf_path = os.path.join(WORKSPACE_DIR, f"report_{task_id}.pdf")

                generate_csv_report(result, csv_path)
                generate_pdf_report(result, img_path, pdf_path)

                dl1, dl2 = st.columns(2)
                with dl1:
                    with open(csv_path, "rb") as f:
                        st.download_button("📥 Download CSV", f, f"report_{task_id}.csv", "text/csv")
                with dl2:
                    with open(pdf_path, "rb") as f:
                        st.download_button("📥 Download PDF", f, f"report_{task_id}.pdf", "application/pdf")
            else:
                st.error(f"Error from server: {response.text}")
        except Exception as e:
            st.error(f"Could not connect to FastAPI: {e}")