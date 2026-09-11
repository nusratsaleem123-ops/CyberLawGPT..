import os
import re
import tempfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# =========================================================
# CYBERLAWSGPT CONFIGURATION
# =========================================================

APP_NAME = "CyberlawsGPT"

PDF_URL = (
    "https://drive.google.com/uc?export=download"
    "&id=1UG3QhN70CfcKCh7kJPbZHFpmbwvvNP0X"
)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_GROK_MODEL = "grok-4.6"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

MAX_HISTORY = 6


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="CyberlawsGPT",
    page_icon="⚖️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 700;
        }

        .subtitle {
            color: #666;
            font-size: 1.05rem;
        }

        .source-box {
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #ddd;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# API KEY
# =========================================================

def get_api_key():
    """
    Reads the Grok API key securely.

    Priority:
    1. Streamlit secrets
    2. Environment variable
    """

    try:
        if "XAI_API_KEY" in st.secrets:
            return st.secrets["XAI_API_KEY"]
    except Exception:
        pass

    return os.getenv("XAI_API_KEY")


# =========================================================
# DOWNLOAD PDF
# =========================================================

@st.cache_data(show_spinner=False)
def download_law_pdf():
    """
    Download the legal PDF.

    The file is stored in a temporary directory.
    """

    temp_dir = Path(tempfile.gettempdir())

    pdf_path = temp_dir / "cyberlaws_pakistan.pdf"

    if pdf_path.exists() and pdf_path.stat().st_size > 5000:
        return str(pdf_path)

    response = requests.get(
        PDF_URL,
        timeout=60,
        allow_redirects=True
    )

    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if (
        "text/html" in content_type
        or len(response.content) < 5000
    ):
        raise ValueError(
            "The Google Drive link did not return a valid PDF. "
            "Please verify that the file is publicly accessible."
        )

    pdf_path.write_bytes(response.content)

    return str(pdf_path)


# =========================================================
# EXTRACT PDF TEXT
# =========================================================

@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_path):
    """
    Extract text from the PDF page by page.
    """

    reader = PdfReader(pdf_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    if not pages:
        raise ValueError(
            "No readable text was found in the PDF."
        )

    return pages


# =========================================================
# CHUNKING
# =========================================================

def chunk_text(text, chunk_size, overlap):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


@st.cache_data(show_spinner=False)
def create_chunks(pages):
    """
    Create chunks while preserving page metadata.
    """

    documents = []

    for page_data in pages:

        page_number = page_data["page"]

        page_chunks = chunk_text(
            page_data["text"],
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1
        ):

            documents.append(
                {
                    "page": page_number,
                    "chunk": chunk_number,
                    "text": chunk
                }
            )

    return documents


# =========================================================
# LOAD LOCAL EMBEDDING MODEL
# =========================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# CREATE EMBEDDINGS
# =========================================================

@st.cache_resource(show_spinner=False)
def build_vector_index(documents):

    model = load_embedding_model()

    texts = [
        document["text"]
        for document in documents
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    return embeddings


# =========================================================
# RETRIEVAL
# =========================================================

def retrieve_documents(
    question,
    documents,
    embeddings,
    top_k=5
):
    """
    Retrieve the most relevant legal passages
    using cosine similarity.

    Because embeddings are normalized,
    dot product = cosine similarity.
    """

    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding[0],
        dtype=np.float32
    )

    scores = embeddings @ query_embedding

    top_indices = np.argsort(
        scores
    )[-top_k:][::-1]

    results = []

    for index in top_indices:

        document = documents[index].copy()

        document["score"] = float(
            scores[index]
        )

        results.append(document)

    return results


# =========================================================
# BUILD LEGAL CONTEXT
# =========================================================

def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
SOURCE:
PDF Page: {result["page"]}
Chunk: {result["chunk"]}

LEGAL TEXT:
{result["text"]}
""".strip()
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# =========================================================
# GROK RESPONSE
# =========================================================

def ask_grok(
    question,
    context,
    technical_level,
    response_size,
    language,
    temperature,
    model_name,
    chat_history
):
    """
    Send the question and retrieved RAG context
    to Grok.
    """

    api_key = get_api_key()

    if not api_key:
        raise ValueError(
            "XAI_API_KEY is missing. "
            "Add it to Streamlit secrets or "
            "your environment variables."
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.x.ai/v1"
    )

    history_text = ""

    for item in chat_history[-MAX_HISTORY:]:

        role = item["role"].upper()

        history_text += (
            f"{role}: {item['content']}\n"
        )

    system_prompt = f"""
You are CyberlawsGPT, an AI legal information
assistant focused on cyber laws in Pakistan.

Your primary task is to answer questions using the
RETRIEVED LEGAL CONTEXT provided below.

IMPORTANT RULES:

1. Treat the retrieved legal context as the primary
   source of truth.

2. Do not invent sections, penalties, amendments,
   cases, legal requirements, or legal authorities.

3. If the answer is not supported by the retrieved
   context, clearly say:
   "The provided legal document does not contain
   enough information to answer this confidently."

4. Explain the answer according to this user level:
   {technical_level}

5. Response size:
   {response_size}

6. Respond in:
   {language}

7. When possible, mention the relevant PDF page
   number from the provided context.

8. Distinguish between:
   - information from the legal text
   - general explanation or interpretation

9. Do not claim to be a lawyer.

10. Add this short disclaimer when appropriate:
    "This is general legal information, not legal advice.
    For a legal dispute or official matter, consult a
    qualified lawyer or relevant Pakistani authority."

11. Do not provide instructions that facilitate
    cybercrime, unauthorized access, malware,
    credential theft, fraud, evasion of law enforcement,
    or other illegal activity.

12. If a user asks how to perform illegal cyber activity,
    refuse the operational instructions and instead provide
    legal consequences, prevention, defensive guidance,
    or lawful alternatives.

13. The source document may not include every future
    amendment. If the user asks about a change that is
    not present in the retrieved text, say so clearly.

Use clear headings when useful.

RETRIEVED LEGAL CONTEXT:

{context}
""".strip()

    user_prompt = f"""
QUESTION:
{question}

RECENT CONVERSATION:
{history_text}

Answer the question using the retrieved legal context.
""".strip()

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature
    )

    return response.choices[0].message.content


# =========================================================
# INITIALIZE RAG
# =========================================================

@st.cache_resource(show_spinner=False)
def initialize_rag():

    pdf_path = download_law_pdf()

    pages = extract_pdf_text(
        pdf_path
    )

    documents = create_chunks(
        pages
    )

    embeddings = build_vector_index(
        documents
    )

    return pages, documents, embeddings


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Response Settings")

    technical_level = st.selectbox(
        "Technical Level",
        [
            "Beginner",
            "Intermediate",
            "Expert"
        ],
        index=0
    )

    response_size = st.selectbox(
        "Response Size",
        [
            "Short",
            "Medium",
            "Detailed"
        ],
        index=1
    )

    language = st.selectbox(
        "Response Language",
        [
            "English",
            "Urdu",
            "Roman Urdu"
        ]
    )

    top_k = st.slider(
        "Legal Sources to Retrieve",
        min_value=2,
        max_value=8,
        value=5
    )

    temperature = st.slider(
        "Response Creativity",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help=(
            "Lower values produce more consistent "
            "and focused answers."
        )
    )

    st.divider()

    model_name = st.text_input(
        "Grok Model",
        value=DEFAULT_GROK_MODEL,
        help=(
            "Change this if your xAI account has "
            "access to another Grok model."
        )
    )

    st.divider()

    show_sources = st.checkbox(
        "Show Retrieved Legal Sources",
        value=True
    )

    if st.button("🗑️ Clear Conversation"):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">⚖️ CyberlawsGPT</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    AI-powered RAG assistant for understanding
    cyber laws and legal provisions related to Pakistan.
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    """
    **Legal Information Notice:** CyberlawsGPT provides
    general information based primarily on the connected
    legal PDF. It is not a substitute for professional
    legal advice, court interpretation, or official
    government guidance.
    """
)


# =========================================================
# LOAD RAG SYSTEM
# =========================================================

try:

    with st.spinner(
        "Loading CyberlawsGPT and creating legal embeddings..."
    ):

        pages, documents, embeddings = initialize_rag()

except Exception as error:

    st.error(
        f"Unable to initialize the legal knowledge base: {error}"
    )

    st.stop()


# =========================================================
# STATUS
# =========================================================

st.caption(
    f"📄 Knowledge Base: {len(pages)} pages | "
    f"🧩 Legal Chunks: {len(documents)} | "
    f"🧠 Embeddings: Local"
)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about cyber laws in Pakistan..."
)


if question:

    # Display user question
    st.chat_message("user").markdown(
        question
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Retrieve relevant context
    retrieved_results = retrieve_documents(
        question=question,
        documents=documents,
        embeddings=embeddings,
        top_k=top_k
    )

    context = build_context(
        retrieved_results
    )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner(
            "Analyzing relevant legal provisions..."
        ):

            try:

                answer = ask_grok(
                    question=question,
                    context=context,
                    technical_level=technical_level,
                    response_size=response_size,
                    language=language,
                    temperature=temperature,
                    model_name=model_name,
                    chat_history=st.session_state.messages
                )

                st.markdown(answer)

            except Exception as error:

                answer = (
                    "⚠️ **Error:** "
                    f"{error}"
                )

                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # =====================================================
    # SHOW SOURCES
    # =====================================================

    if show_sources:

        with st.expander(
            "📚 Retrieved Legal Context"
        ):

            for number, result in enumerate(
                retrieved_results,
                start=1
            ):

                st.markdown(
                    f"""
### Source {number}

**PDF Page:** {result["page"]}  
**Relevance Score:** {result["score"]:.3f}

{result["text"]}
                    """
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "CyberlawsGPT | RAG-powered legal information assistant "
    "| Pakistan cyber law knowledge base"
)
