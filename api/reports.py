import csv
import os
from fpdf import FPDF


def generate_csv_report(result_data: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Parameter", "Value"])
        writer.writerow(["Ore Class", result_data.get("ore_class")])
        writer.writerow(["Talc Content (%)", result_data.get("talc_percent")])
    return output_path


def generate_pdf_report(result_data: dict, image_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Ore Microstructure Analysis Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Assigned Ore Class: {result_data.get('ore_class')}", ln=True)
    pdf.cell(200, 10, txt=f"Talc Concentration: {result_data.get('talc_percent')}%", ln=True)
    pdf.ln(5)

    if os.path.exists(image_path):
        pdf.image(image_path, x=10, y=None, w=190)

    pdf.output(output_path)
    return output_path