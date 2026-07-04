import streamlit as st
import requests
import time
import os
from reports import generate_csv_report, generate_pdf_report


st.set_page_config(page_title="QC-Дашборд | Норникель", page_icon="🔬", layout="wide")

st.title("Автоматизированный анализ микроструктур руды")
st.markdown("""
Загрузите панорамный OM-снимок полированного шлифа. 
Система автоматически проведет классификацию, сегментирует включения талька и сформирует отчет.
""")

API_URL = "http://127.0.0.1:8000"

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "result_data" not in st.session_state:
    st.session_state.result_data = None
if "task_id" not in st.session_state:
    st.session_state.task_id = None

uploaded_file = st.file_uploader("Выберите TIFF/PNG/JPEG файл", type=["tif", "tiff", "png", "jpg", "jpeg"])


if uploaded_file and st.session_state.task_id and not uploaded_file.name in st.session_state.get("last_uploaded_name",""):
    st.session_state.analysis_done = False
    st.session_state.result_data = None
    st.session_state.task_id = None
if uploaded_file:
    st.session_state.last_uploaded_name = uploaded_file.name


trigger_analysis = st.button("Запустить анализ", type="primary", disabled=uploaded_file is None)

if trigger_analysis:
    with st.status("Отправка данных на сервер...", expanded=True) as status:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        try:
            response = requests.post(f"{API_URL}/upload", files=files)
            if response.status_code == 200:
                task_id = response.json()["task_id"]
                st.session_state.task_id = task_id
                st.write(f"Файл принят. ID задачи: `{task_id}`")

                st.write("ML-модели обрабатывают снимок...")
                while True:
                    res = requests.get(f"{API_URL}/status/{task_id}")
                    if res.status_code == 200:
                        data = res.json()
                        if data["status"] == "completed":
                            status.update(label="Анализ успешно завершен!", state="complete", expanded=False)
                            st.session_state.result_data = data["result"]
                            st.session_state.analysis_done = True
                            break
                        elif data["status"] == "failed":
                            status.update(label="Ошибка при анализе", state="error")
                            st.error(data.get("error", "Неизвестная ошибка"))
                            st.stop()
                    time.sleep(2.0)
            else:
                status.update(label="Ошибка сервера", state="error")
                st.error(f"Не удалось связаться с API (Код {response.status_code})")
                st.stop()
        except requests.exceptions.ConnectionError:
            status.update(label="Ошибка подключения", state="error")
            st.error("FastAPI сервер не запущен или недоступен!")
            st.stop()

if st.session_state.analysis_done and st.session_state.result_data:
    st.divider()
    st.subheader("Результаты анализа")

    result_data = st.session_state.result_data
    task_id = st.session_state.task_id
    image_path = result_data["result_image_path"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Присвоенный класс руды", value=result_data["ore_class"])
    with col2:
        st.metric(label="Доля талька", value=f"{result_data['talc_percent']}%")

    if os.path.exists(image_path):
        st.image(image_path, caption="Результат сегментации: Исходное изображение с наложенной маской талька",
                 use_column_width=True)
    else:
        st.warning("Файл с маской не найден на диске бэкенда.")

    st.subheader("Экспорт отчетов")

    csv_path = os.path.join("workspace", f"report_{task_id}.csv")
    pdf_path = os.path.join("workspace", f"report_{task_id}.pdf")

    generate_csv_report(result_data, csv_path)

    pdf_success = False
    if os.path.exists(image_path):
        try:
            generate_pdf_report(result_data, image_path, pdf_path)
            pdf_success = True
        except Exception as e:
            st.error(f"Ошибка компиляции PDF-отчета (LaTeX): {e}")

    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        if os.path.exists(csv_path):
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📄 Скачать таблицу (CSV)",
                    data=f,
                    file_name=f"ore_analysis_{task_id}.csv",
                    mime="text/csv",
                    key="csv_download"
                )

    with col_dl2:
        if pdf_success and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📑 Скачать отчет (PDF)",
                    data=f,
                    file_name=f"ore_report_{task_id}.pdf",
                    mime="application/pdf",
                    key="pdf_download"
                )