import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/uniprot_human.csv"
INDEX_PATH = "data/uniprot_text.index"
META_PATH = "data/protein_metadata.pkl"


def clean_text(value):
    if pd.isna(value):
        return ""
    return str(value).replace("\n", " ").strip()


def build_text(row):
    return f"""
    Accession: {clean_text(row.get('accession'))}
    Protein: {clean_text(row.get('protein_name'))}
    Gene: {clean_text(row.get('gene_name'))}
    Organism: {clean_text(row.get('organism'))}
    Function: {clean_text(row.get('function'))}
    Sequence preview: {clean_text(row.get('sequence'))[:300]}
    """


def main():
    df = pd.read_csv(DATA_PATH)
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

    df = df.fillna("")

    texts = df.apply(build_text, axis=1).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print(f"Built FAISS index with {len(df)} proteins.")
    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {META_PATH}")


if __name__ == "__main__":
    main()