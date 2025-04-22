import os
import argparse
import requests
import json
import xml.etree.ElementTree as ET
from params import (
    DATASET_NAME as DEFAULT_DATASET_NAME,
    PREDICTION_TASK as DEFAULT_PREDICTION_TASK,
    ARTICLES_MAX as DEFAULT_MAX_ARTICLES,
)

def make_output_dir(dataset_name: str) -> str:
    output_dir = f'rag_sources_{dataset_name.lower()}'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def pubmed_search(search_terms: str, max_results: int = DEFAULT_MAX_ARTICLES) -> list:
    # use NCBI E-utilities to search PubMed
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {'db': 'pubmed', 'term': search_terms, 'retmax': max_results, 'retmode': 'json'}
    response = requests.get(url, params=params, timeout=30)
    data = response.json()
    return data.get('esearchresult', {}).get('idlist', [])

def fetch_pubmed_abstracts(id_list: list) -> list:
    if not id_list:
        return []
    url = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    params = {'db': 'pubmed', 'id': ','.join(id_list), 'retmode': 'xml'}
    response = requests.get(url, params=params, timeout=60)
    root = ET.fromstring(response.text)
    records = []
    for art in root.findall('.//PubmedArticle'):
        pmid = art.findtext('.//PMID') or ''
        title = art.findtext('.//ArticleTitle') or ''
        abstract_text = ''.join([t.text or '' for t in art.findall('.//AbstractText')])
        journal = art.findtext('.//Journal/Title') or ''
        year = art.findtext('.//PubDate/Year') or art.findtext('.//PubDate/MedlineDate') or ''
        records.append({'pmid': pmid, 'title': title, 'journal': journal, 'year': year, 'abstract': abstract_text, 'source': 'pubmed'})
    return records

def save_metadata(records: list, output_dir: str):
    meta_file = os.path.join(output_dir, 'metadata.jsonl')
    mode = 'a' if os.path.exists(meta_file) else 'w'
    with open(meta_file, mode, encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')

def main():
    parser = argparse.ArgumentParser(description='Fetch PubMed abstracts and save as JSONL.')
    parser.add_argument('--dataset_name', default=DEFAULT_DATASET_NAME)
    parser.add_argument('--prediction_task', default=DEFAULT_PREDICTION_TASK)
    parser.add_argument('--max_articles', type=int, default=DEFAULT_MAX_ARTICLES)
    args = parser.parse_args()

    output_dir = make_output_dir(args.dataset_name)
    search_terms = f'{args.dataset_name} {args.prediction_task}'
    pmids = pubmed_search(search_terms, args.max_articles)
    records = fetch_pubmed_abstracts(pmids)
    save_metadata(records, output_dir)
    print(f'Saved {len(records)} abstracts from PubMed in {output_dir}/metadata.jsonl')

if __name__ == '__main__':
    main()
