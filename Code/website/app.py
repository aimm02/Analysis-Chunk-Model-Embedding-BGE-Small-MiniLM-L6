import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
app = FastAPI()

# Model BGE Small
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

db_path = "./vector_stores/chunk_size_512_BGE Small"
vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

# Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

system_prompt = (
    "Anda adalah asisten medis Klinik Pratama Nusa Putra. "
    "Gunakan konteks berikut untuk menjawab pertanyaan. "
    "Jika tidak ada dalam konteks, katakan Anda tidak tahu.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 2}), qa_chain)

@app.get("/chat")
async def chat(query: str):
    try:
        response = rag_chain.invoke({"input": query})
        return {"answer": response["answer"]}
    except Exception as e:
        return {"answer": f"Backend Error: {str(e)}"}