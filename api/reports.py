import csv
import os
import subprocess


def generate_csv_report(result_data: dict, output_path: str) -> str:
    """Генерирует CSV файл с метриками, совместимый с Excel."""

    with open(output_path, mode='w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerow(["Параметр", "Значение"])
        writer.writerow(["Класс руды", result_data.get("ore_class", "N/A")])

        talc_str = str(result_data.get("talc_percent", 0.0)).replace('.', ',')
        writer.writerow(["Содержание талька (%)", talc_str])

    return output_path


def generate_pdf_report(result_data: dict, image_path: str, output_path: str) -> str:
    """Генерирует строгий PDF отчет через компиляцию LaTeX и удаляет временные файлы."""
    tex_path = output_path.replace('.pdf', '.tex')
    aux_path = output_path.replace('.pdf', '.aux')
    log_path = output_path.replace('.pdf', '.log')

    safe_image_path = image_path.replace("\\", "/")

    tex_content = f"""
\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[russian]{{babel}}
\\usepackage{{graphicx}}
\\usepackage[margin=2cm]{{geometry}}
\\usepackage{{helvet}}
\\renewcommand{{\\familydefault}}{{\\sfdefault}}

\\begin{{document}}

\\begin{{center}}
    \\Large\\textbf{{Отчет микроструктурного анализа руды}} \\\\
    \\vspace{{0.5cm}}
    \\large Автоматизированная система контроля (Норникель)
\\end{{center}}

\\vspace{{1cm}}

\\section*{{1. Результаты классификации}}
\\begin{{itemize}}
    \\item \\textbf{{Присвоенный класс руды:}} {result_data.get('ore_class')}
    \\item \\textbf{{Оценочная доля талька:}} {result_data.get('talc_percent')}\\%
\\end{{itemize}}

\\vspace{{1cm}}

\\section*{{2. Визуализация шлифа с маской сегментации}}
\\begin{{center}}
    \\includegraphics[width=0.9\\textwidth]{{{safe_image_path}}}
\\end{{center}}

\\vspace{{1cm}}
\\noindent\\rule{{\\textwidth}}{{0.4pt}}
\\small{{\\textit{{Документ сгенерирован автоматически ML-пайплайном.}}}}

\\end{{document}}
    """

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(tex_content)

    output_dir = os.path.dirname(output_path)
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", f"-output-directory={output_dir}", tex_path],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    for temp_file in [tex_path, aux_path, log_path]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError as e:
            print(f"Не удалось удалить временный файл {temp_file}: {e}")

    return output_path