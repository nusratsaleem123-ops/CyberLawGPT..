# ⚖️ CyberlawsGPT

CyberlawsGPT is a Retrieval-Augmented Generation (RAG) application designed to help users understand cyber laws and legal provisions related to Pakistan.

The application automatically downloads a legal PDF, extracts its text, creates local embeddings, retrieves relevant legal passages, and uses the Grok API to generate a context-aware answer.

---

## 🚀 Features

- Automatic PDF download on startup
- PDF text extraction
- Local embeddings using Sentence Transformers
- RAG-based legal question answering
- Grok API integration
- Beginner, Intermediate, and Expert technical levels
- Short, Medium, and Detailed response modes
- English, Urdu, and Roman Urdu support
- Adjustable number of retrieved legal sources
- Adjustable response creativity
- Conversation history
- Retrieved legal source display
- Legal information disclaimer
- Secure API key handling
- No hard-coded API keys
- No external vector database required

---

# 🧠 How CyberlawsGPT Works

CyberlawsGPT follows a RAG pipeline:

1. Download the cyber law PDF.
2. Extract text from the PDF.
3. Split the text into smaller overlapping chunks.
4. Create embeddings using a local Sentence Transformer model.
5. Store embeddings in application memory.
6. Convert the user's question into an embedding.
7. Find the most relevant legal passages using similarity search.
8. Send the question and retrieved legal context to Grok.
9. Generate an answer grounded in the retrieved legal text.

---

# 📁 Project Structure

Only three files are required:

```text
CyberlawsGPT/
│
├── app.py
├── requirement.txt
└── README.md# CyberLawGPT..
