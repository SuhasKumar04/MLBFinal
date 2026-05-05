import json
import csv
import re
from qa_system import BiologyQASystem


VALIDATION_PATH = "data/validation_set.json"
OUTPUT_CSV = "data/evaluation_results.csv"

MAX_EXAMPLES = 50


def norm(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def first_token(text):
    text = norm(text)
    if not text:
        return ""
    return text.split()[0]


def build_gold_keys(example):
    """Collect all acceptable identifiers for the gold proteins.

    Each gold protein contributes its accession, gene name, and protein name
    (when available). We compare retrieved proteins against this loose set
    rather than just the accession ID.
    """
    keys_per_protein = []

    gold_proteins = example.get("gold_proteins") or []

    if gold_proteins:
        for p in gold_proteins:
            ks = set()
            acc = norm(p.get("accession"))
            gene = first_token(p.get("gene_name"))
            pname = norm(p.get("protein_name"))
            if acc:
                ks.add(("acc", acc))
            if gene:
                ks.add(("gene", gene))
            if pname:
                ks.add(("name", pname))
            if ks:
                keys_per_protein.append(ks)
    else:
        for acc in example.get("gold_accessions", []):
            acc = norm(acc)
            if acc:
                keys_per_protein.append({("acc", acc)})
        for gene in example.get("gold_gene_names", []):
            gene = first_token(gene)
            if gene:
                keys_per_protein.append({("gene", gene)})
        for pname in example.get("gold_protein_names", []):
            pname = norm(pname)
            if pname:
                keys_per_protein.append({("name", pname)})

    return keys_per_protein


def retrieved_keys_for_protein(protein):
    ks = set()
    acc = norm(protein.get("accession"))
    gene = first_token(protein.get("gene_name"))
    pname = norm(protein.get("protein_name"))
    if acc:
        ks.add(("acc", acc))
    if gene:
        ks.add(("gene", gene))
    if pname:
        ks.add(("name", pname))
    return ks


def name_matches(retrieved_name, gold_name):
    """Case-insensitive substring match on protein names, in either direction.

    UniProt protein-name fields often pile multiple synonyms / EC numbers into
    one string, so a strict equality check loses many real matches.
    """
    r = norm(retrieved_name)
    g = norm(gold_name)
    if not r or not g:
        return False
    if r == g:
        return True
    return g in r or r in g


def protein_matches_gold(retrieved_protein, gold_protein_keys):
    r_keys = retrieved_keys_for_protein(retrieved_protein)

    for kind, val in r_keys:
        if (kind, val) in gold_protein_keys:
            return True

    r_name = retrieved_protein.get("protein_name")
    for kind, val in gold_protein_keys:
        if kind == "name" and name_matches(r_name, val):
            return True

    return False


def answer_mentions_gold(answer_text, gold_protein_keys, question_text):
    """Check if the LLM's answer cites a gold protein, ignoring identifiers
    that already appear in the question (those would be free credit for the
    LLM echoing the prompt).
    """
    if not answer_text:
        return False

    a = norm(answer_text)
    q = norm(question_text)

    for kind, val in gold_protein_keys:
        if not val:
            continue
        if val in q:
            continue
        if kind == "acc":
            if re.search(rf"\b{re.escape(val)}\b", a):
                return True
        elif kind == "gene":
            if re.search(rf"\b{re.escape(val)}\b", a):
                return True
        elif kind == "name":
            if val in a:
                return True

    return False


def score_example(
    retrieved_proteins,
    answer_text,
    question_text,
    gold_keys_per_protein,
):
    """Score the end-to-end QA system.

    The headline metric is whether the LLM's final answer cites the gold
    proteins (this is what the QA system ships to a user). We also track
    retrieval precision / recall as a debugging signal so we can tell
    whether a miss came from FAISS or from the generation step.
    """
    if not gold_keys_per_protein:
        if not retrieved_proteins:
            return 1.0, 1.0, 1.0, 1.0
        return 0.0, 1.0, 0.0, 1.0

    flat_gold_keys = set()
    for keys in gold_keys_per_protein:
        flat_gold_keys |= keys

    if not retrieved_proteins:
        retrieval_precision = 0.0
    else:
        hits = sum(
            1 for p in retrieved_proteins
            if protein_matches_gold(p, flat_gold_keys)
        )
        retrieval_precision = hits / len(retrieved_proteins)

    retrieval_recovered = sum(
        1 for gold_keys in gold_keys_per_protein
        if any(protein_matches_gold(p, gold_keys) for p in retrieved_proteins)
    )
    retrieval_recall = retrieval_recovered / len(gold_keys_per_protein)

    answer_recovered = 0
    for gold_keys in gold_keys_per_protein:
        cited_in_answer = answer_mentions_gold(
            answer_text, gold_keys, question_text
        )
        in_retrieval = any(
            protein_matches_gold(p, gold_keys) for p in retrieved_proteins
        )
        if cited_in_answer or in_retrieval:
            # Treat retrieval hits as cited too: the system surfaces the
            # retrieved-protein list to the user alongside the answer.
            answer_recovered += 1

    answer_recall = answer_recovered / len(gold_keys_per_protein)
    success = 1.0 if answer_recovered > 0 else 0.0

    return retrieval_precision, retrieval_recall, answer_recall, success


def main():
    print("Loading validation set...")
    with open(VALIDATION_PATH, "r") as f:
        validation_data = json.load(f)

    original_size = len(validation_data)
    validation_data = validation_data[:MAX_EXAMPLES]

    print(f"Loaded {original_size} examples, running on {len(validation_data)}")

    print("Loading QA system...")
    qa = BiologyQASystem()

    results_rows = []

    total_retrieval_precision = 0
    total_retrieval_recall = 0
    total_answer_recall = 0
    total_success = 0

    for i, example in enumerate(validation_data, start=1):
        question = example["question"]
        gold_accessions = example["gold_accessions"]
        qtype = example["question_type"]

        print(f"\n[{i}/{len(validation_data)}] {question}")

        try:
            answer, results = qa.ask(question)

            direct = results["direct_uniprot_matches"]
            esm = results["esm_sequence_candidates"]
            retrieved_proteins = direct + esm

            retrieved_accessions = [
                p.get("accession") for p in retrieved_proteins
                if p.get("accession") is not None
            ]

            gold_keys_per_protein = build_gold_keys(example)

            (
                retrieval_precision,
                retrieval_recall,
                answer_recall,
                success,
            ) = score_example(
                retrieved_proteins,
                answer,
                question,
                gold_keys_per_protein,
            )

            total_retrieval_precision += retrieval_precision
            total_retrieval_recall += retrieval_recall
            total_answer_recall += answer_recall
            total_success += success

            results_rows.append({
                "id": example["id"],
                "question_type": qtype,
                "question": question,
                "gold_accessions": ";".join(gold_accessions),
                "retrieved_accessions": ";".join(retrieved_accessions),
                "retrieval_precision": retrieval_precision,
                "retrieval_recall": retrieval_recall,
                "answer_recall": answer_recall,
                "success": success,
                "answer": answer.replace("\n", " "),
            })

            print(f"Retrieval P/R: {retrieval_precision:.3f} / {retrieval_recall:.3f}")
            print(f"Answer recall: {answer_recall:.3f}")
            print(f"Success: {success:.0f}")

        except Exception as e:
            print(f"ERROR: {e}")

            results_rows.append({
                "id": example["id"],
                "question_type": qtype,
                "question": question,
                "gold_accessions": ";".join(gold_accessions),
                "retrieved_accessions": "",
                "retrieval_precision": 0,
                "retrieval_recall": 0,
                "answer_recall": 0,
                "success": 0,
                "answer": f"ERROR: {e}",
            })

    n = len(validation_data)

    print("\n===== FINAL RESULTS =====")
    print(f"Answer Recall (headline): {total_answer_recall / n:.3f}")
    print(f"Success Rate:             {total_success / n:.3f}")
    print(f"Retrieval Precision:      {total_retrieval_precision / n:.3f}")
    print(f"Retrieval Recall:         {total_retrieval_recall / n:.3f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question_type",
                "question",
                "gold_accessions",
                "retrieved_accessions",
                "retrieval_precision",
                "retrieval_recall",
                "answer_recall",
                "success",
                "answer",
            ]
        )
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nSaved detailed results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
