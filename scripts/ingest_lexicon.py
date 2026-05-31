import argparse
import os
import sys
import requests
import xml.etree.ElementTree as ET

# Fix for Windows console unicode printing
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import chromadb
from chromadb.utils import embedding_functions

def get_chroma_collection(db_path="./chroma_db"):
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    collection = client.get_or_create_collection(
        name="lanes_lexicon",
        embedding_function=emb_fn
    )
    return collection

def fetch_xml_files(api_url):
    response = requests.get(api_url)
    response.raise_for_status()
    items = response.json()
    return [item for item in items if item["name"].endswith(".xml") and item["name"] != "__contents__.xml"]

def parse_tei_xml(xml_content):
    entries = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        print(f"Parse error: {e}")
        return []
        
    for entry in root.findall('.//entryFree'):
        orth = entry.find('.//orth')
        if orth is not None and orth.text:
            root_word = orth.text.strip()
            text_content = "".join(entry.itertext()).strip()
            
            if root_word and text_content:
                entry_id = entry.get("id", f"lane_root_{root_word}")
                entries.append({
                    "root": root_word,
                    "definition": text_content,
                    "id": entry_id
                })
    return entries

def ingest_lexicon(api_url, db_path, dry_run=False):
    print(f"Fetching XML file list from {api_url}")
    xml_files = fetch_xml_files(api_url)
    
    if dry_run:
        print(f"Dry run: Found {len(xml_files)} files. Only parsing the first one.")
        xml_files = xml_files[:1]
        
    all_entries = []
    seen_ids = set()
    for file_info in xml_files:
        print(f"Downloading {file_info['name']}...")
        resp = requests.get(file_info['download_url'])
        if resp.status_code == 200:
            entries = parse_tei_xml(resp.content)
            
            # Deduplicate IDs
            for e in entries:
                original_id = e["id"]
                unique_id = original_id
                counter = 1
                while unique_id in seen_ids:
                    unique_id = f"{original_id}_{counter}"
                    counter += 1
                seen_ids.add(unique_id)
                e["id"] = unique_id
                
            print(f"Extracted {len(entries)} entries from {file_info['name']}.")
            all_entries.extend(entries)
        else:
            print(f"Failed to download {file_info['name']}")
            
    if dry_run:
        print("Dry run complete. Sample entry:")
        if all_entries:
            print(all_entries[0])
        return

    print(f"Starting ingestion of {len(all_entries)} entries into ChromaDB at {db_path}...")
    collection = get_chroma_collection(db_path)
    
    batch_size = 500
    for i in range(0, len(all_entries), batch_size):
        batch = all_entries[i:i+batch_size]
        documents = []
        metadatas = []
        ids = []
        
        for entry in batch:
            doc = f"Arabic Word/Root: {entry['root']}\nDefinition: {entry['definition']}"
            documents.append(doc)
            metadatas.append({"root": entry["root"], "source": "Lane's Lexicon"})
            ids.append(entry["id"])
            
        print(f"Upserting batch {i//batch_size + 1}/{(len(all_entries) + batch_size - 1)//batch_size} ({len(batch)} entries)...")
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    print("Ingestion complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest Lane's Lexicon into a local ChromaDB for RAG")
    parser.add_argument(
        "--api-url", 
        type=str, 
        default="https://api.github.com/repos/alpheios-project/lan/contents/db/lexica/ara/lan", 
        help="GitHub API URL for the lexicon TEI XML files"
    )
    parser.add_argument(
        "--db-path", 
        type=str, 
        default=os.path.join(os.path.dirname(__file__), "..", "chroma_db"), 
        help="Path to local ChromaDB directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only download and parse the first file, do not ingest into ChromaDB"
    )
    
    args = parser.parse_args()
    ingest_lexicon(args.api_url, args.db_path, args.dry_run)
