import json
import csv
from qa_system import BiologyQASystem


VALIDATION_PATH = "data/validation_set.json"
OUTPUT_CSV = "data/evaluation_results.csv"

MAX_EXAMPLES = 50  


def precision_at_k(retrieved, gold):
    if len(retrieved) == 0:
        return 0.0
    return len(set(retrieved) & set(gold)) / len(retrieved)


def recall_at_k(retrieved, gold):
    if len(gold) == 0:
        return 1.0 if len(retrieved) == 0 else 0.0
    return len(set(retrieved) & set(gold)) / len(gold)


def success_at_k(retrieved, gold):
    if len(gold) == 0:
        return 1.0 if len(retrieved) == 0 else 0.0
    return 1.0 if len(set(retrieved) & set(gold)) > 0 else 0.0


def main():
    print("Loading validation set...")
    with open(VALIDATION_PATH, "r") as f:
        validation_data = json.load(f)

    # 🔥 Limit dataset size
    original_size = len(validation_data)
    validation_data = validation_data[:MAX_EXAMPLES]

    print(f"Loaded {original_size} examples, running on {len(validation_data)}")

    print("Loading QA system...")
    qa = BiologyQASystem()

    results_rows = []

    total_precision = 0
    total_recall = 0
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

            retrieved_accessions = []

            for p in direct:
                retrieved_accessions.append(p.get("accession"))

            for p in esm:
                retrieved_accessions.append(p.get("accession"))

            retrieved_accessions = [
                x for x in retrieved_accessions if x is not None
            ]

            precision = precision_at_k(retrieved_accessions, gold_accessions)
            recall = recall_at_k(retrieved_accessions, gold_accessions)
            success = success_at_k(retrieved_accessions, gold_accessions)

            total_precision += precision
            total_recall += recall
            total_success += success

            results_rows.append({
                "id": example["id"],
                "question_type": qtype,
                "question": question,
                "gold_accessions": ";".join(gold_accessions),
                "retrieved_accessions": ";".join(retrieved_accessions),
                "precision": precision,
                "recall": recall,
                "success": success,
                "answer": answer.replace("\n", " ")
            })

            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"Success: {success:.0f}")

        except Exception as e:
            print(f"ERROR: {e}")

            results_rows.append({
                "id": example["id"],
                "question_type": qtype,
                "question": question,
                "gold_accessions": ";".join(gold_accessions),
                "retrieved_accessions": "",
                "precision": 0,
                "recall": 0,
                "success": 0,
                "answer": f"ERROR: {e}"
            })

    n = len(validation_data)

    print("\n===== FINAL RESULTS =====")
    print(f"Average Precision: {total_precision / n:.3f}")
    print(f"Average Recall: {total_recall / n:.3f}")
    print(f"Success Rate: {total_success / n:.3f}")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "question_type",
                "question",
                "gold_accessions",
                "retrieved_accessions",
                "precision",
                "recall",
                "success",
                "answer"
            ]
        )
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nSaved detailed results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()