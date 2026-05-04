import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import faiss
import pickle
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

DATA_PATH = "data/uniprot_human.csv"
ESM_INDEX_PATH = "data/esm_sequence.index"
ESM_META_PATH = "data/esm_metadata.pkl"

MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


def clean_sequence(seq):
    if pd.isna(seq):
        return ""
    return str(seq).replace(" ", "").replace("\n", "").strip()


def embed_sequence(sequence, tokenizer, model, device):
    inputs = tokenizer(
        sequence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask
    summed = masked_embeddings.sum(dim=1)
    counts = attention_mask.sum(dim=1)

    embedding = summed / counts

    return embedding.squeeze().cpu().numpy()


def main():
    df = pd.read_csv(DATA_PATH).fillna("")
    df = df.rename(columns={
        "Entry": "accession",
        "Protein names": "protein_name",
        "Gene Names": "gene_name",
        "Organism": "organism",
        "Function [CC]": "function",
        "Sequence": "sequence"
    })

    required_columns = [
        "accession",
        "protein_name",
        "gene_name",
        "organism",
        "function",
        "sequence"
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    df["sequence"] = df["sequence"].apply(clean_sequence)
    df = df[df["sequence"] != ""].reset_index(drop=True)

    # Test with fewer proteins first
    df = df.head(500)

    device = "cpu"
    torch.set_num_threads(1)

    print(f"Using device: {device}")
    print(f"Embedding {len(df)} sequences")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    embeddings = []

    for seq in tqdm(df["sequence"], desc="Embedding sequences with ESM-2"):
        try:
            emb = embed_sequence(seq, tokenizer, model, device)
            embeddings.append(emb)
        except Exception as e:
            print(f"Skipping one sequence because of error: {e}")

    embeddings = np.vstack(embeddings).astype("float32")

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, ESM_INDEX_PATH)

    with open(ESM_META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print(f"Built ESM-2 FAISS index with {len(embeddings)} proteins.")
    print(f"Saved index to {ESM_INDEX_PATH}")
    print(f"Saved metadata to {ESM_META_PATH}")


if __name__ == "__main__":
    main()