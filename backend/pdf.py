import os
from fpdf import FPDF

SOURCE_FOLDER = "."
OUTPUT_PDF = "all_python_files.pdf"

class PDF(FPDF):
    def header(self):
        self.set_font("Courier", "B", 10)
        self.cell(0, 8, "Python Source Code Compilation")
        self.ln(10)

def safe_text(text: str) -> str:
    """Remove unicode that FPDF can't handle"""
    return text.encode("latin-1", "ignore").decode("latin-1")

def add_file_to_pdf(pdf, filepath):
    pdf.add_page()
    pdf.set_font("Courier", size=8)

    # File header
    pdf.write(5, safe_text(f"File: {filepath}\n\n"))

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            safe_line = safe_text(line.rstrip())
            pdf.write(5, safe_line + "\n")

def main():
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for root, dirs, files in os.walk(SOURCE_FOLDER):
        dirs[:] = [d for d in dirs if d not in {".venv", "venv", "__pycache__", ".git"}]

        for file in files:
            if file.endswith(".py"):
                add_file_to_pdf(pdf, os.path.join(root, file))

    pdf.output(OUTPUT_PDF)
    print(f"✅ PDF created successfully: {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
