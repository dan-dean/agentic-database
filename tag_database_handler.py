from huggingface_hub import hf_hub_download
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gc

MODELS_DIR = './models/embedding'
DATABASE_DIR = './databases/tags'
model_name = 'openai/text-embedding-ada-002'
embedding_dim = 1536

'''
The vector tag database handler module for agentic database. Provides insertion and nearest neighbor search operations
for the tag database. Vector indices are paired with matching named json dictionaries to map index to string. Intended to match near-tags from LLM analysis with real-tags present in the document SQL database.
The module provides the following functions:

- def load_model(model_name): Loads the SentenceTransformer model with the given name from the Hugging Face model hub.
- def create_database(title): Creates a new database with the given title and returns True if the database was created, False if it already exists.
- def delete_database(title): Deletes the database with the given title and returns True if the database was deleted, False if it didn't exist.
- def add_entry_to_database(title, embedding, tag): Adds an entry to the database with the given title.
    returns True if the entry was added, False if the database didn't exist.
- def get_nearest_neighbors(title, embedding, tag, k=20): Returns the k nearest neighbors to the given tag in the database with the given title.
    returns a list of lists of tags.
- def delete_entry_from_database(title, tags): Deletes the given tags from the database with the given title. Will remove the tag from 
    the index and the json tag map.
'''

def download_model(model_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        hf_hub_download(model_name, model_path)
    return model_path

def load_model(model_name):
    model_path = download_model(model_name)
    return SentenceTransformer(model_path)

def unload_model(model):
    del download_model
    gc.collect()

def create_database_dir():
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

def create_database(title):
    db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
    json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
    
    if not os.path.exists(db_path):
        create_database_dir()
        index = faiss.IndexFlatL2(embedding_dim)
        faiss.write_index(index, db_path)
        
        with open(json_path, 'w') as f:
            json.dump({}, f)
        
        return True
    return False

def delete_database(title):
    db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
    json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
    
    if os.path.exists(db_path):
        os.remove(db_path)
        os.remove(json_path)
        return True
    return False

def load_index_to_tag_map(title):
    json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}

def save_index_to_tag_map(title, index_to_tag):
    json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
    with open(json_path, 'w') as f:
        json.dump(index_to_tag, f)

def add_entry_to_database(title, embedding, tag):
    db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
    if os.path.exists(db_path):
        index = faiss.read_index(db_path)
        index_to_tag = load_index_to_tag_map(title)

        if isinstance(tag, str):
            tag = [tag]

        for t in tag:
            vector = embedding.encode([t])[0].astype('float32')
            index.add(np.array([vector]))
            index_id = index.ntotal - 1
            
            index_to_tag[index_id] = t

        faiss.write_index(index, db_path)
        save_index_to_tag_map(title, index_to_tag)
        return True
    return False

def get_nearest_neighbors(title, embedding, tag, k=20):
    db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
    index = faiss.read_index(db_path)

    if isinstance(tag, str):
        tag = [tag]

    tag_vectors = np.array([embedding.encode([t])[0].astype('float32') for t in tag])
    
    D, I = index.search(tag_vectors, k)


    index_to_tag = load_index_to_tag_map(title)
    neighbors = [[index_to_tag.get(str(i), "Unknown") for i in row] for row in I]
    
    return neighbors

def reverse_dict(original_dict):
    return {v: k for k, v in original_dict.items()}

def delete_entry_from_database(title, tags):
    db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
    index = faiss.read_index(db_path)
    index_to_tag = load_index_to_tag_map(title)
    tag_to_index = reverse_dict(index_to_tag)

    if isinstance(tags, str):
        tags = [tags]

    for t in tags:
        if t in tag_to_index:
            tag_index = int(tag_to_index[t])

            index.remove_ids(np.array([tag_index], dtype=np.int64))

            del index_to_tag[str(tag_index)]

    faiss.write_index(index, db_path)
    save_index_to_tag_map(title, index_to_tag)