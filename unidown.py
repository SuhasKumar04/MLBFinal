import requests
import pandas as pd
from io import StringIO

query = "reviewed:true AND organism_id:9606"

url = "https://rest.uniprot.org/uniprotkb/search"

params = {
    "query": query,
    "format": "tsv",
    "fields": "accession,protein_name,gene_names,organism_name,cc_function,sequence",
    "size": 500
}

response = requests.get(url, params=params)

# Read directly into pandas
df = pd.read_csv(StringIO(response.text), sep="\t")

# Save as CSV
df.to_csv("data/uniprot_human.csv", index=False)

print(df.head())