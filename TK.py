import tkinter as tk
from tkinter import messagebox
import numpy as np
from gensim.models import KeyedVectors
from keras.models import load_model
import nltk
import re
from nltk.corpus import stopwords

# Load the pre-trained model and Word2Vec embeddings
lstm_model = load_model('final_lstm.h5')
word2vec_model = KeyedVectors.load_word2vec_format("word2vecmodel.bin", binary=True)

# Preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub("[^A-Za-z]", " ", text)
    text = text.lower()
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# Convert text to vector
def text_to_vector(text):
    words = preprocess_text(text)
    vec = np.zeros((300,), dtype="float32")
    no_of_words = 0
    for word in words:
        if word in word2vec_model:
            vec = np.add(vec, word2vec_model[word])
            no_of_words += 1
    if no_of_words > 0:
        vec /= no_of_words
    return vec

# Get score
def get_score(text):
    text_vec = text_to_vector(text)
    text_vec = np.reshape(text_vec, (1, 1, text_vec.shape[0]))
    score = lstm_model.predict(text_vec)[0][0]
    return round(score, 2)

# Callback function for scoring button
def score_essay():
    essay_text = text_entry.get("1.0", "end-1c")
    if essay_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter an essay!")
        return
    score = get_score(essay_text)
    messagebox.showinfo("Score", f"The score of the essay is: {score}")

# Create GUI
root = tk.Tk()
root.title("Automated Essay Scoring")
root.configure(bg="#2b2b2b")

label = tk.Label(root, text="Enter your essay:", fg="white", bg="#2b2b2b", font=("Helvetica", 14, "bold"))
label.pack(pady=(20, 5))

text_entry = tk.Text(root, height=10, width=50, bg="#3b3b3b", fg="white", font=("Helvetica", 12))
text_entry.pack()

score_button = tk.Button(root, text="Get Score", command=score_essay, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"))
score_button.pack(pady=(10, 20))

root.mainloop()
