from qa_system import BiologyQASystem

qa = BiologyQASystem()

while True:
    question = input("\nAsk a protein biology question: ")

    if question.lower() in ["quit", "exit"]:
        break
    
    answer, results = qa.ask(question)

    print("\nAnswer:")
    print(answer)

    print("\nDirect UniProt matches:")
    for p in results["direct_uniprot_matches"]:
        print(f"- {p.get('protein_name')} ({p.get('gene_name')}), score={p.get('text_score'):.3f}")

    print("\nESM-2 sequence candidates:")
    for p in results["esm_sequence_candidates"]:
        print(
            f"- {p.get('protein_name')} ({p.get('gene_name')}), "
            f"similar to {p.get('similar_to_name')}, "
            f"ESM score={p.get('esm_score'):.3f}"
        )