import tkinter as tk
from tkinter import messagebox, filedialog
import pandas as pd
import os

class QnAApp:
    def __init__(self, root, excel_file="questions_answers.xlsx"):
        self.root = root
        self.root.title("Analytical Mechanics Q&A Study App")
        self.root.geometry("780x560")
        self.root.minsize(650, 480)
        
        self.questions_data = self.load_from_excel(excel_file)
        if not self.questions_data:
            self.root.destroy()
            return

        self.current_index = 0
        self.answer_visible = False

        # --- UI Styling ---
        self.bg_color = "#f4f6f9"
        self.card_color = "#ffffff"
        self.primary_color = "#2c3e50"
        self.accent_color = "#2980b9"
        
        self.root.configure(bg=self.bg_color)

        # Header / Progress Label
        self.progress_label = tk.Label(
            root, text="", font=("Helvetica", 11, "bold"), bg=self.bg_color, fg="#7f8c8d"
        )
        self.progress_label.pack(pady=(15, 5))

        # Main Card Frame
        self.card_frame = tk.Frame(root, bg=self.card_color, bd=1, relief="solid")
        self.card_frame.pack(fill="both", expand=True, padx=30, pady=10)

        # Chapter Tag Label
        self.chapter_label = tk.Label(
            self.card_frame, text="", font=("Helvetica", 10, "italic"), bg=self.card_color, fg=self.accent_color
        )
        self.chapter_label.pack(anchor="w", padx=20, pady=(15, 5))

        # Question ID & Text
        self.question_label = tk.Label(
            self.card_frame, text="", font=("Helvetica", 13, "bold"), bg=self.card_color, 
            fg=self.primary_color, wraplength=680, justify="left"
        )
        self.question_label.pack(anchor="w", padx=20, pady=(5, 15))

        # Separator Line inside Card
        self.separator = tk.Frame(self.card_frame, height=2, bg="#ecf0f1")
        self.separator.pack(fill="x", padx=20, pady=5)

        # Answer Title / Text Area
        self.answer_title_label = tk.Label(
            self.card_frame, text="Answer:", font=("Helvetica", 11, "bold"), bg=self.card_color, fg="#27ae60"
        )
        self.answer_title_label.pack(anchor="w", padx=20, pady=(10, 0))

        self.answer_text = tk.Text(
            self.card_frame, font=("Helvetica", 11), bg="#f9fbfc", fg="#34495e", 
            wrap="word", relief="flat", height=10, state="disabled"
        )
        self.answer_text.pack(fill="both", expand=True, padx=20, pady=(5, 20))

        # --- Control Button Panel ---
        self.btn_frame = tk.Frame(root, bg=self.bg_color)
        self.btn_frame.pack(fill="x", padx=30, pady=20)

        self.prev_btn = tk.Button(
            self.btn_frame, text="⬅ Previous", font=("Helvetica", 11), bg="#bdc3c7", fg="#2c3e50",
            padx=15, pady=8, relief="raised", command=self.prev_question
        )
        self.prev_btn.pack(side="left")

        self.reveal_btn = tk.Button(
            self.btn_frame, text="Show Answer", font=("Helvetica", 11, "bold"), bg=self.accent_color, fg="white",
            padx=20, pady=8, relief="raised", command=self.toggle_answer
        )
        self.reveal_btn.pack(side="left", expand=True)

        self.next_btn = tk.Button(
            self.btn_frame, text="Next ➡", font=("Helvetica", 11), bg="#bdc3c7", fg="#2c3e50",
            padx=15, pady=8, relief="raised", command=self.next_question
        )
        self.next_btn.pack(side="right")

        # Load first question
        self.load_question()

    def load_from_excel(self, filepath):
        if not os.path.exists(filepath):
            filepath = filedialog.askopenfilename(
                title="Select Q&A Excel File",
                filetypes=[("Excel files", "*.xlsx *.xls")]
            )
            if not filepath:
                return []
        
        df = pd.read_excel(filepath)
        # Convert rows into a list of dictionaries matching the app structure
        data = []
        for _, row in df.iterrows():
            data.append({
                "chapter": str(row["Chapter"]),
                "id": str(row["ID"]),
                "question": str(row["Question"]),
                "answer": str(row["Answer"])
            })
        return data

    def load_question(self):
        q_data = self.questions_data[self.current_index]
        
        self.progress_label.config(text=f"Question {self.current_index + 1} of {len(self.questions_data)}")
        self.chapter_label.config(text=q_data["chapter"])
        self.question_label.config(text=f"{q_data['id']}. {q_data['question']}")
        
        self.answer_visible = False
        self.reveal_btn.config(text="Show Answer", bg=self.accent_color)
        
        self.answer_text.config(state="normal")
        self.answer_text.delete("1.0", tk.END)
        self.answer_text.insert("1.0", "Click 'Show Answer' to reveal the solution step.")
        self.answer_text.config(state="disabled", fg="#95a5a6")

    def toggle_answer(self):
        q_data = self.questions_data[self.current_index]
        self.answer_text.config(state="normal")
        self.answer_text.delete("1.0", tk.END)
        
        if not self.answer_visible:
            self.answer_text.insert("1.0", q_data["answer"])
            self.answer_text.config(fg="#34495e")
            self.reveal_btn.config(text="Hide Answer", bg="#e67e22")
            self.answer_visible = True
        else:
            self.answer_text.insert("1.0", "Click 'Show Answer' to reveal the solution step.")
            self.answer_text.config(fg="#95a5a6")
            self.reveal_btn.config(text="Show Answer", bg=self.accent_color)
            self.answer_visible = False
            
        self.answer_text.config(state="disabled")

    def next_question(self):
        if self.current_index < len(self.questions_data) - 1:
            self.current_index += 1
            self.load_question()
        else:
            messagebox.showinfo("End of Deck", "You have reached the last question in this study session!")

    def prev_question(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_question()
        else:
            messagebox.showinfo("Beginning of Deck", "You are already at the first question.")

if __name__ == "__main__":
    root = tk.Tk()
    app = QnAApp(root)
    root.mainloop()