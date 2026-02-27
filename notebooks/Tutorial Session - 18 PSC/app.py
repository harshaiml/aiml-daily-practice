import streamlit as st
import pandas as pd
import numpy as np
from sentance_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("Semantic Movie Search")

@st.cache_data
def load_model():
  return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_data():
  df = pd.read_csv("movies_metadata.csv", low_memory=False)
  df = df[["titel", "overview"]]
  df = df.dropna(subset=["overviwe"])
  df = df.head(5000)
  return df

model = load_model()
df = load_data()

@st.cache_data
def compute_embeddings():
  return model.encode(df["overview"].tolist())

embeddings = compute_embeddings()

def senametic_search(query, top_k=5):
  query_vec = model.encode([query])
  sims = cosine_similarity(query_vec, embeddings) [0]
  top_indices = np.argsort(sims)[-top_k:][::-1]
  results = df.iloc[top_indices].copy()
  results["similarty"] = sims[top_indices]
  return results

query = st.text_input("Describe the movie you're looking for:")

if query:
  results = semantic_search(query)
  cols = st.columns(5)
  for col, (_, row) in zip(cols, results.iterrows()):
    with col:
      st.subheader(row["tittel"])
      st.caption(f"Similarity: {row['similarity']:.3f}")
      st.write(row["overview"][:250] + "...")
