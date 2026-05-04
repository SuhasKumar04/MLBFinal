import faiss
import pickle
import ollama
from sentence_transformers import SentenceTransformer

TEXT_INDEX_PATH = "data/uniprot_text.index"
TEXT_META_PATH = "data/protein_metadata.pkl"

ESM_INDEX_PATH = "data/esm_sequence.index"
ESM_META_PATH = "data/esm_metadata.pkl"


class BiologyQASystem:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        self.text_index = faiss.read_index(TEXT_INDEX_PATH)
        with open(TEXT_META_PATH, "rb") as f:
            self.text_metadata = pickle.load(f)

        self.esm_index = faiss.read_index(ESM_INDEX_PATH)
        with open(ESM_META_PATH, "rb") as f:
            self.esm_metadata = pickle.load(f)

        self.accession_to_esm_idx = {
            protein["accession"]: i
            for i, protein in enumerate(self.esm_metadata)
        }

    def retrieve_text(self, question, top_k=5):
        query_embedding = self.embedder.encode(
            [question],
            normalize_embeddings=True
        )

        scores, ids = self.text_index.search(query_embedding, top_k)

        results = []

        for score, idx in zip(scores[0], ids[0]):
            protein = self.text_metadata[idx].copy()
            protein["text_score"] = float(score)
            protein["source"] = "Direct UniProt text match"
            results.append(protein)

        return results

    def retrieve_esm_candidates(self, seed_proteins, top_k_per_seed=3):
        esm_results = []

        for seed in seed_proteins:
            accession = seed.get("accession")

            if accession not in self.accession_to_esm_idx:
                continue

            seed_idx = self.accession_to_esm_idx[accession]
            seed_vector = self.esm_index.reconstruct(seed_idx).reshape(1, -1)

            scores, ids = self.esm_index.search(seed_vector, top_k_per_seed + 1)

            for score, idx in zip(scores[0], ids[0]):
                candidate = self.esm_metadata[idx].copy()

                if candidate.get("accession") == accession:
                    continue

                candidate["esm_score"] = float(score)
                candidate["source"] = "ESM-2 sequence-similar candidate"
                candidate["similar_to_accession"] = accession
                candidate["similar_to_name"] = seed.get("protein_name", "")
                candidate["similar_to_gene"] = seed.get("gene_name", "")

                esm_results.append(candidate)

        return esm_results

    def deduplicate_esm_candidates(self, direct_results, esm_results):
        direct_accessions = {
            protein.get("accession")
            for protein in direct_results
        }

        seen = set()
        unique_esm = []

        for protein in esm_results:
            accession = protein.get("accession")

            if accession in direct_accessions:
                continue

            if accession in seen:
                continue

            seen.add(accession)
            unique_esm.append(protein)

        return unique_esm

    def format_direct_evidence(self, direct_results):
        evidence = ""

        for i, protein in enumerate(direct_results, start=1):
            evidence += f"""
Direct Match {i}
Accession: {protein.get('accession', '')}
Protein Name: {protein.get('protein_name', '')}
Gene: {protein.get('gene_name', '')}
Organism: {protein.get('organism', '')}
Function: {protein.get('function', '')}
Text Retrieval Score: {protein.get('text_score', 'N/A')}
"""

        return evidence

    def format_esm_evidence(self, esm_results):
        evidence = ""

        for i, protein in enumerate(esm_results, start=1):
            evidence += f"""
ESM Candidate {i}
Accession: {protein.get('accession', '')}
Protein Name: {protein.get('protein_name', '')}
Gene: {protein.get('gene_name', '')}
Organism: {protein.get('organism', '')}
Function: {protein.get('function', '')}
ESM Similarity Score: {protein.get('esm_score', 'N/A')}
Similar To: {protein.get('similar_to_name', '')}
Similar To Gene: {protein.get('similar_to_gene', '')}
Similar To Accession: {protein.get('similar_to_accession', '')}
"""

        return evidence

    def generate_answer(self, question, direct_results, esm_results):
        direct_evidence = self.format_direct_evidence(direct_results)
        esm_evidence = self.format_esm_evidence(esm_results)

        prompt = f"""
You are a biology question-answering assistant.

The user is asking:
{question}

You are given two types of retrieved proteins:

1. DIRECT UNIPROT TEXT MATCHES:
These proteins were retrieved from UniProt annotations using semantic text search.
These are the strongest evidence for the user's requested function.

2. ESM-2 SEQUENCE-SIMILAR CANDIDATES:
These proteins were found because their amino acid sequences are similar to the direct UniProt matches.
These may share related functions, but they should be described as possible candidates, not confirmed hits.

Rules:
- Separate the answer into two sections.
- First list the proteins directly related to the requested function from UniProt annotations.
- Then list additional possible related proteins found using ESM-2 sequence similarity.
- Do not claim that ESM-2 candidates definitely have the function unless their UniProt function text also supports it.
- For each direct UniProt match, explain why it matches the requested function.
- For each ESM-2 candidate, say which seed protein it was similar to.
- Be concise but specific.
- Use only the provided evidence.
- If evidence is weak, say so.

DIRECT UNIPROT TEXT MATCHES:
{direct_evidence}

ESM-2 SEQUENCE-SIMILAR CANDIDATES:
{esm_evidence}

Final answer:
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    def ask(self, question, text_top_k=5, esm_top_k_per_seed=3):
        direct_results = self.retrieve_text(
            question,
            top_k=text_top_k
        )

        esm_results = self.retrieve_esm_candidates(
            direct_results,
            top_k_per_seed=esm_top_k_per_seed
        )

        esm_results = self.deduplicate_esm_candidates(
            direct_results,
            esm_results
        )

        answer = self.generate_answer(
            question,
            direct_results,
            esm_results
        )

        all_results = {
            "direct_uniprot_matches": direct_results,
            "esm_sequence_candidates": esm_results
        }

        return answer, all_results