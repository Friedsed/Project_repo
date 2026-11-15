import re
import pandas as pd
from pypdf import PdfReader

def extract_pdf_to_excel(pdf_path, excel_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    
    # Extract text starting from page 2 (skipping the title/TOC page)
    for page in reader.pages[1:]:
        full_text += page.extract_text() + "\n"

    # Clean up page headers like "Analytical Mechanics Solutions Manual (Chapters 8-11)"
    full_text = re.sub(r"Analytical Mechanics Solutions Manual \(Chapters 8-11\)\s*\d*", "", full_text)

    # Split text by question identifiers (e.g., Q8.1., Q9.12., Q11.16.)
    # Pattern looks for 'Q' followed by digits, a dot, digits, and a dot
    pattern = re.compile(r"(Q\d+\.\d+\.)\s*(.*?)(?=(?:Q\d+\.\d+\.|\Z))", re.DOTALL)
    matches = pattern.findall(full_text)

    chapter_map = {
        "8": "Chapter 8: Rigid Body Mechanics (Kinematics & Dynamics)",
        "9": "Chapter 9: Impulsive Dynamics of Rigid Bodies (Collisions)",
        "10": "Chapter 10: Oscillators & Small Oscillations",
        "11": "Chapter 11: Continuous Media & Hamilton-Jacobi"
    }

    records = []
    for q_id, content in matches:
        clean_id = q_id.rstrip(".")
        chap_num = clean_id.split(".")[0].replace("Q", "")
        chapter_name = chapter_map.get(chap_num, "General")

        # Clean excessive newlines inside the block
        content = re.sub(r"\s+", " ", content).strip()

        # Split the first sentence (Topic/Question Title) from the rest (Answer)
        parts = content.split(". ", 1)
        if len(parts) == 2:
            question_text = parts[0] + "."
            answer_text = parts[1]
        else:
            question_text = content
            answer_text = "See manual details."

        records.append({
            "Chapter": chapter_name,
            "ID": clean_id,
            "Question": question_text,
            "Answer": answer_text
        })

    # Save to Excel
    df = pd.DataFrame(records)
    df.to_excel(excel_path, index=False, engine="openpyxl")
    print(f"Successfully extracted {len(records)} Q&A items to '{excel_path}'!")

if __name__ == "__main__":
    extract_pdf_to_excel("solutions_cours_chap891011.pdf", "questions_answers.xlsx")