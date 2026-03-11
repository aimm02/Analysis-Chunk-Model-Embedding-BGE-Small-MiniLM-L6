from __future__ import annotations
import json
import os
import pandas as pd
import time
from typing import Dict, List
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision
from ragas.run_config import RunConfig
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

# Configuration
TEST_DATASET_PATH = "test_dataset.json" 

def load_test_dataset(file_path: str):
    """Memuat dataset pengujian dari JSON."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan!")
    with open(file_path, 'r') as f:
        return json.load(f)

def run_ragas_evaluation(experiments_metadata: Dict, test_dataset: List[Dict]) -> pd.DataFrame:
    """Menjalankan RAGAS dengan mode 'Sabar' (Anti-Rate Limit)."""
    
    # Gemini 2.5 Flash Lite
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0,
        request_timeout=120
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Run Configuration
    run_config = RunConfig(
        timeout=1200,
        max_retries=3,
        max_workers=1
    )
    
    all_scores = {}
    
    # Load Data
    ragas_questions = [item['question'] for item in test_dataset]
    ragas_gt = []
    for item in test_dataset:
        gt = item.get('ground_truth_context', "")
        ragas_gt.append("\n".join(gt) if isinstance(gt, list) else str(gt))

    for exp_name, meta in experiments_metadata.items():
        print(f"\nPengujian Metrik RAGAS] Eksperimen: {exp_name}")
        
        try:
            vector_store = Chroma(
                persist_directory=meta["db_path"], 
                embedding_function=meta["embedding_model"]
            )
            retriever = vector_store.as_retriever(search_kwargs={"k": 5}) 
            
            print(f"> Melakukan retrieval untuk {len(ragas_questions)} pertanyaan...")
            retrieved_contexts = []
            for q in ragas_questions:
                docs = retriever.invoke(q)
                retrieved_contexts.append([doc.page_content for doc in docs])
                time.sleep(2)

            data_dict = {
                    "question": ragas_questions,
                    "contexts": retrieved_contexts,
                    "ground_truth": ragas_gt,
                    "answer": [""] * len(ragas_questions)
            }
            evaluation_dataset = Dataset.from_dict(data_dict)
            
            print("> Menghitung skor dengan Gemini..")
            
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=[context_recall, context_precision],
                llm=llm,
                embeddings=embeddings,
                run_config=run_config
            )
            
            scores = result.to_pandas().mean(numeric_only=True).fillna(0).to_dict()
            all_scores[exp_name] = scores
            print(f"Selesai! Skor: {scores}")

        except Exception as e:
            print(f"Gagal/Limit Habis pada {exp_name}. Error: {e}")
            all_scores[exp_name] = {"context_recall": 0.0, "context_precision": 0.0}
            time.sleep(60)

    return pd.DataFrame(all_scores).T

def generate_final_report(scores_df: pd.DataFrame):
    print("\nAnalisis Komparatif Skor RAGAS")
    
    if not scores_df.empty:
        scores_df['average_score'] = scores_df.mean(axis=1)
        print(scores_df.sort_values(by='average_score', ascending=False)) 
        
        optimal_config = scores_df['average_score'].idxmax()
        print(f"Konfigurasi Data Kualitas Tinggi Adalah: {optimal_config}")