import json
import string
import tkinter as tk
from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
with open("dataset.json", "r") as file:
    data = json.load(file)

questions = [item["question"] for item in data]
answers = [item["answer"] for item in data]

# Preprocess text
def preprocess(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    return text

processed_questions = [preprocess(q) for q in questions]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(processed_questions)

def get_response(user_input):
    user_input = preprocess(user_input)
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    index = similarity.argmax()
    return answers[index]

# GUI Setup
def send_message():
    user_input = entry.get()
    if user_input.strip() == "":
        return
    chat_window.insert(tk.END, "You: " + user_input + "\n")
    response = get_response(user_input)
    chat_window.insert(tk.END, "AI: " + response + "\n\n")
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("AI Smart Study Assistant")
root.geometry("500x500")

chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD)
chat_window.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

entry = tk.Entry(root)
entry.pack(padx=10, pady=5, fill=tk.X)

send_button = tk.Button(root, text="Send", command=send_message)
send_button.pack(pady=5)

root.mainloop()