import json
import random
from pathlib import Path

import pandas as pd


DATA_PATH = "data/uniprot_human.csv"
OUTPUT_PATH = "data/validation_set.json"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

MAX_EXAMPLES = 300


FUNCTION_CATEGORIES = {
    "apoptosis": ["apoptosis", "apoptotic"],
    "DNA repair": ["dna repair", "double-strand break", "repair"],
    "cell cycle": ["cell cycle", "mitosis", "checkpoint"],
    "transcription": ["transcription", "transcriptional"],
    "kinase activity": ["kinase", "phosphorylation", "phosphorylates"],
    "immune response": ["immune", "cytokine", "inflammatory"],
    "signal transduction": ["signal", "signaling", "receptor"],
    "oxidative stress": ["oxidative", "reactive oxygen", "antioxidant"],
    "metabolism": ["metabolic", "metabolism", "catalyzes"],
    "protein transport": ["transport", "trafficking", "vesicle"],
}


FUNCTION_TO_PROTEIN_TEMPLATES = [
    "Which human proteins are involved in {category}?",
    "Name proteins associated with {category}.",
    "List proteins that play a role in {category}.",
    "What proteins are related to {category}?",
    "Find proteins connected to {category}.",
]

PROTEIN_TO_FUNCTION_TEMPLATES = [
    "What does {gene} do?",
    "What is the function of {protein}?",
    "Explain the biological role of {gene}.",
    "What is {protein} involved in?",
]

GENE_TO_PROTEIN_TEMPLATES = [
    "What protein is encoded by the gene {gene}?",
    "Which protein corresponds to {gene}?",
    "Identify the protein associated with gene {gene}.",
]

ACCESSION_TO_PROTEIN_TEMPLATES = [
    "What protein has the UniProt accession {accession}?",
    "Identify the protein with accession ID {accession}.",
]

NEGATIVE_TEMPLATES = [
    "Which human proteins are involved in photosynthesis?",
    "Which human proteins are involved in chlorophyll production?",
    "Which human proteins perform bacterial flagellar rotation?",
    "Which human proteins are responsible for nitrogen fixation?",
]


def normalize_columns(df):
    return df.rename(columns={
        "Entry": "accession",
        "Protein names": "protein_name",
        "Gene Names": "gene_name",
        "Organism": "organism",
        "Function [CC]": "function",
        "Sequence": "sequence",
    })


def first_gene(gene_field):
    gene_field = str(gene_field).strip()
    if not gene_field:
        return ""
    return gene_field.split()[0]


def clean_text(x):
    return str(x).replace("\n", " ").strip()


def matches_any(function_text, keywords):
    function_text = str(function_text).lower()
    return any(k.lower() in function_text for k in keywords)


def add_example(examples, question, qtype, gold_rows, category=None):
    gold_rows = gold_rows.copy()

    proteins = []
    for _, row in gold_rows.iterrows():
        proteins.append({
            "accession": clean_text(row["accession"]),
            "gene_name": first_gene(row["gene_name"]),
            "protein_name": clean_text(row["protein_name"]),
            "function_evidence": clean_text(row["function"])[:600],
        })

    examples.append({
        "id": f"val_{len(examples) + 1:04d}",
        "question": question,
        "question_type": qtype,
        "category": category,
        "gold_accessions": [p["accession"] for p in proteins],
        "gold_gene_names": [p["gene_name"] for p in proteins if p["gene_name"]],
        "gold_protein_names": [p["protein_name"] for p in proteins],
        "gold_proteins": proteins,
        "source": "UniProt",
    })


def build_validation_set():
    print("Loading UniProt data...")
    df = pd.read_csv(DATA_PATH).fillna("")
    df = normalize_columns(df)

    required = ["accession", "protein_name", "gene_name", "function"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["gene_clean"] = df["gene_name"].apply(first_gene)

    df = df[
        (df["accession"].astype(str).str.len() > 0)
        & (df["protein_name"].astype(str).str.len() > 0)
        & (df["function"].astype(str).str.len() > 30)
    ].reset_index(drop=True)

    print(f"Usable proteins: {len(df)}")

    examples = []

    # 1. Function -> protein list questions
    print("\nBuilding function-to-protein examples...")
    for category, keywords in FUNCTION_CATEGORIES.items():
        subset = df[df["function"].apply(lambda x: matches_any(x, keywords))]

        print(f"{category}: {len(subset)} matches")

        if len(subset) < 3:
            continue

        for _ in range(8):
            sampled = subset.sample(
                n=min(5, len(subset)),
                random_state=random.randint(0, 999999)
            )

            template = random.choice(FUNCTION_TO_PROTEIN_TEMPLATES)
            question = template.format(category=category)

            add_example(
                examples,
                question,
                "function_to_protein",
                sampled,
                category
            )

    # 2. Protein/gene -> function questions
    print("\nBuilding protein-to-function examples...")
    protein_rows = df[df["gene_clean"] != ""].sample(
        n=min(80, len(df[df["gene_clean"] != ""])),
        random_state=RANDOM_SEED
    )

    for _, row in protein_rows.iterrows():
        one_row = pd.DataFrame([row])

        if random.random() < 0.5:
            template = random.choice(PROTEIN_TO_FUNCTION_TEMPLATES)
            question = template.format(
                gene=row["gene_clean"],
                protein=clean_text(row["protein_name"])
            )
            qtype = "protein_to_function"
        else:
            template = random.choice(GENE_TO_PROTEIN_TEMPLATES)
            question = template.format(gene=row["gene_clean"])
            qtype = "gene_to_protein"

        add_example(examples, question, qtype, one_row)

    # 3. Accession ID -> protein questions
    print("\nBuilding accession-to-protein examples...")
    accession_rows = df.sample(n=min(40, len(df)), random_state=RANDOM_SEED + 1)

    for _, row in accession_rows.iterrows():
        template = random.choice(ACCESSION_TO_PROTEIN_TEMPLATES)
        question = template.format(accession=row["accession"])

        add_example(
            examples,
            question,
            "accession_to_protein",
            pd.DataFrame([row])
        )

    # 4. Sequence-length / metadata style questions
    # These test whether system can retrieve a protein by name and discuss it.
    print("\nBuilding metadata-style examples...")
    meta_rows = df[df["gene_clean"] != ""].sample(
        n=min(40, len(df[df["gene_clean"] != ""])),
        random_state=RANDOM_SEED + 2
    )

    for _, row in meta_rows.iterrows():
        question = random.choice([
            f"Give information about the human protein {row['gene_clean']}.",
            f"What is known about {clean_text(row['protein_name'])}?",
            f"Summarize the UniProt annotation for {row['gene_clean']}.",
        ])

        add_example(
            examples,
            question,
            "protein_summary",
            pd.DataFrame([row])
        )

    # 5. Negative examples
    # These should ideally return no confident human protein hits.
    print("\nBuilding negative examples...")
    for q in NEGATIVE_TEMPLATES:
        examples.append({
            "id": f"val_{len(examples) + 1:04d}",
            "question": q,
            "question_type": "negative",
            "category": "out_of_scope",
            "gold_accessions": [],
            "gold_gene_names": [],
            "gold_protein_names": [],
            "gold_proteins": [],
            "source": "UniProt",
        })

    random.shuffle(examples)

    if len(examples) > MAX_EXAMPLES:
        examples = examples[:MAX_EXAMPLES]

    Path("data").mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(examples, f, indent=2)

    print(f"\nSaved {len(examples)} examples to {OUTPUT_PATH}")

    type_counts = {}
    for ex in examples:
        type_counts[ex["question_type"]] = type_counts.get(ex["question_type"], 0) + 1

    print("\nQuestion type counts:")
    for qtype, count in type_counts.items():
        print(f"- {qtype}: {count}")


if __name__ == "__main__":
    build_validation_set()