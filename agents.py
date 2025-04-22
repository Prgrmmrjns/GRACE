from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import os
import csv
from params import DATASET_PATH, DATASET_NAME, TARGET_COL, MODEL
import json
from langchain.schema import Document
from pydantic import BaseModel, Field
from langgraph.prebuilt import create_react_agent

PDF_DIR = "rag_sources_mimic"
VECTOR_DB_PATH = "faiss_index"

def load_dataset_description(dataset_name: str) -> str:
    """Load dataset description from the corresponding info file"""
    info_file = f"dataset_info/{dataset_name.lower()}_info.txt"
    if not os.path.exists(info_file):
        raise FileNotFoundError(f"Dataset info file not found: {info_file}")
    with open(info_file, 'r') as f:
        return f.read().strip()
    
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    feature_names = [col for col in header if col.lower() != TARGET_COL.lower()]

# 2. Load dataset description
dataset_description = load_dataset_description(DATASET_NAME)

# Gather all PDF files in the directory
pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]

all_docs = []
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    splitted_docs = splitter.split_documents(pages)
    all_docs.extend(splitted_docs)

# Also load PubMed abstracts from metadata.jsonl
metadata_path = os.path.join(PDF_DIR, 'metadata.jsonl')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
    abstract_docs = [Document(page_content=rec['abstract'], metadata={'pmid': rec['pmid'], 'source': rec.get('source')}) for rec in records]
    splitted_abstracts = splitter.split_documents(abstract_docs)
    all_docs.extend(splitted_abstracts)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Check if FAISS index exists, if so, load it, else create and save it
if os.path.exists(VECTOR_DB_PATH):
    vector_db = FAISS.load_local(VECTOR_DB_PATH, embeddings_model, allow_dangerous_deserialization=True)
else:
    vector_db = FAISS.from_documents(documents=all_docs, embedding=embeddings_model)
    vector_db.save_local(VECTOR_DB_PATH)

# --- Langgraph tool definition and agent logic ---

from langchain_core.tools import tool

class RetrieverToolInput(BaseModel):
    query: str = Field(description="A search term for finding relevant documents.")

@tool
def retriever_tool(input: RetrieverToolInput) -> str:
    """
    Retrieve the most relevant scientific document for a given query, returning its source and a supporting text quote.
    """
    query = input.query
    docs = vector_db.similarity_search(query, k=4)
    # Return the most relevant doc with its source and a quote
    if docs:
        doc = docs[0]
        source = doc.metadata.get('source', 'unknown')
        text_quote = doc.page_content[:300]  # first 300 chars as a quote
        return json.dumps({
            'source': source,
            'text_quote': text_quote,
            'full_text': doc.page_content
        })
    else:
        return json.dumps({'source': None, 'text_quote': '', 'full_text': ''})

# Define the structured output schema
class RetrievalAnswer(BaseModel):
    answer: str = Field(description="A concise answer to the user's question.")
    source: str = Field(description="The source (article or document) where the answer was found.")
    text_quote: str = Field(description="A direct quote from the source supporting the answer.")

# System prompt for the agent
system_prompt = (
    "You are a scientific assistant. Use the retriever_tool to find the most relevant scientific article or abstract. "
    "Answer the user's question concisely, cite the source, and provide a direct quote from the article or abstract that supports your answer. "
    "Return your answer in the structured output format."
)

llm = ChatOpenAI(model=MODEL)

# Create the ReAct agent with structured output
agent = create_react_agent(
    llm,
    tools=[retriever_tool],
    response_format=RetrievalAnswer
)

# Run the agent with system prompt as the first message
inputs = {"messages": [
    ("system", system_prompt),
    ("user", "What are major mechanisms contributing to intensive care mortality?")
]}
response = agent.invoke(inputs)

# Print the structured output
print("\nStructured Output:")
print(response["structured_response"])