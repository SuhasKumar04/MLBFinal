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

    def _short_function(self, text, limit=160):
        if not text:
            return ""
        text = str(text).replace("\n", " ").strip()
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0] + "..."
        return text

    def format_direct_evidence(self, direct_results):
        lines = []
        for protein in direct_results:
            gene = protein.get("gene_name", "") or ""
            name = protein.get("protein_name", "") or ""
            func = self._short_function(protein.get("function", ""))
            lines.append(f"- {gene} ({name}): {func}")
        return "\n".join(lines)

    def format_esm_evidence(self, esm_results):
        lines = []
        for protein in esm_results:
            gene = protein.get("gene_name", "") or ""
            name = protein.get("protein_name", "") or ""
            func = self._short_function(protein.get("function", ""))
            seed = protein.get("similar_to_gene") or protein.get("similar_to_name", "")
            lines.append(f"- {gene} ({name}) [similar to {seed}]: {func}")
        return "\n".join(lines)

    def generate_answer(self, question, direct_results, esm_results):
        direct_evidence = self.format_direct_evidence(direct_results)
        esm_evidence = self.format_esm_evidence(esm_results)

        prompt = f"""Answer the biology question using only the proteins listed below.

Question: {question}

Direct UniProt matches (strong evidence):
{direct_evidence}

ESM-2 sequence-similar candidates (possible related proteins):
{esm_evidence}

Write two short sections: "Direct matches" and "Possible related (ESM-2)". For each protein, give one sentence on why it fits the question. Be concise. Use only the proteins above.
"""

        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response["message"]["content"]

    def retrieve(
        self,
        question,
        text_top_k=30,
        esm_top_k_per_seed=1,
        esm_seed_count=5,
    ):
        direct_results = self.retrieve_text(
            question,
            top_k=text_top_k
        )

        esm_seeds = direct_results[:esm_seed_count]

        esm_results = self.retrieve_esm_candidates(
            esm_seeds,
            top_k_per_seed=esm_top_k_per_seed
        )

        esm_results = self.deduplicate_esm_candidates(
            direct_results,
            esm_results
        )

        return {
            "direct_uniprot_matches": direct_results,
            "esm_sequence_candidates": esm_results
        }

    def ask(
        self,
        question,
        text_top_k=30,
        esm_top_k_per_seed=1,
        esm_seed_count=5,
    ):
        all_results = self.retrieve(
            question,
            text_top_k=text_top_k,
            esm_top_k_per_seed=esm_top_k_per_seed,
            esm_seed_count=esm_seed_count,
        )

        answer = self.generate_answer(
            question,
            all_results["direct_uniprot_matches"],
            all_results["esm_sequence_candidates"]
        )

        return answer, all_results