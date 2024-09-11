from huggingface_hub import hf_hub_download
import os
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import gc
import warnings

MODELS_DIR = './models/embedding'
DATABASE_DIR = './databases/tags'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model_path = os.path.join(MODELS_DIR, model_name)
embedding_dim = 384

'''
The vector tag database handler module for agentic database. Provides insertion and nearest neighbor search operations
for the tag database. Vector indices are paired with matching named json dictionaries to map index to string. Intended to match near-tags from LLM analysis with real-tags present in the document SQL database.
The module provides the following functions:

- def get_model(model_name): loads the SentenceTransformer model or retrieves it if it already exists.
    return: SentenceTransformer object
- def release_model(model): releases SentenceTransformer model from memory.
    return: None
- def create_database(title): creates a new database with the given title and returns True if the database was created, False if it already exists.
    return: bool
- def delete_database(title): deletes the database with the given title and returns True if the database was deleted, False if it didn't exist.
    return: bool
- def add_entry_to_database(title, embedding, tag): adds an entry to the database with the given title.
    return: bool
- def get_nearest_neighbors(title, embedding, tag, k=20): returns the k nearest neighbors to the given tag(s) in the database with the given title.
    return: [[str]]
- def delete_entry_from_database(title, tags): deletes the given tags from the database with the given title. Will remove the tag from 
    the index and the json tag map.
    return: None
'''
class TagDatabaseHandler:

    #singleton model
    _model = None

    def __init__(self):
        print("Downloading or loading the embedding model locally...")
        self.get_model()

    def get_model(self):
        if self._model is None:
            # Ensure the directory exists
            if not os.path.exists(MODELS_DIR):
                os.makedirs(MODELS_DIR)
            # Load the model and cache it locally in MODELS_DIR
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*")

            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder=MODELS_DIR)

            self._model.tokenizer.clean_up_tokenization_spaces = True
        return self._model

    def release_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()

    def create_database_dir(self):
        if not os.path.exists(DATABASE_DIR):
            os.makedirs(DATABASE_DIR)

    def create_database(self, title):
        db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
        json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
        
        if not os.path.exists(db_path):
            self.create_database_dir()

            # Use IndexIDMap to allow custom ID handling
            index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
            faiss.write_index(index, db_path)
            
            with open(json_path, 'w') as f:
                json.dump({}, f)
            
            return True
        return False

    def delete_database(self, title):
        db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
        json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
        deleted_ids_json_path = os.path.join(DATABASE_DIR, f"{title}-deleted-ids.json")
        
        if os.path.exists(db_path):
            os.remove(db_path)
            os.remove(json_path)
            os.remove(deleted_ids_json_path)
            return True
        return False

    def load_index_to_tag_map(self, title):
        json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}

    def save_index_to_tag_map(self, title, index_to_tag):
        json_path = os.path.join(DATABASE_DIR, f"{title}-tags.json")
        with open(json_path, 'w') as f:
            json.dump(index_to_tag, f)

    def add_entry_to_database(self, title, tag):
        db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
        if os.path.exists(db_path):
            model = self.get_model()
            index = faiss.read_index(db_path)
            index_to_tag = self.load_index_to_tag_map(title)
            tag_to_index = self.reverse_dict(index_to_tag)
            deleted_ids = self.load_deleted_ids(title)  # Track deleted IDs

            if isinstance(tag, str):
                tag = [tag]

            for t in tag:
                if t in tag_to_index:
                    continue  # Tag already exists, no need to add it again

                # Calculate the embedding directly using the loaded model
                vector = model.encode([t])[0].astype('float32')

                # Reuse a deleted ID if available, otherwise add a new vector
                if deleted_ids:
                    reuse_id = deleted_ids.pop(0)  # Reuse the first available deleted ID
                    index.add_with_ids(np.array([vector]), np.array([reuse_id]))
                    index_id = reuse_id
                else:
                    index_id = index.ntotal - 1  # Assign the next available ID
                    index.add_with_ids(np.array([vector]), index_id)

                index_to_tag[index_id] = t  # Update the tag map with the new ID

            faiss.write_index(index, db_path)
            self.save_index_to_tag_map(title, index_to_tag)
            self.save_deleted_ids(title, deleted_ids)  # Save updated list of deleted IDs
            return True
        return False


    def get_nearest_neighbors(self, title, tag, k=20):
        db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
        neighbors = []
        if os.path.exists(db_path):
            model = self.get_model()
            index = faiss.read_index(db_path)

            if isinstance(tag, str):
                tag = [tag]

            # Calculate embeddings directly using the loaded model
            tag_vectors = np.array([model.encode([t])[0].astype('float32') for t in tag])

            num_vectors = index.ntotal
            k = min(k, num_vectors)
            if k == 0:
                return []
            
            D, I = index.search(tag_vectors, k)

            index_to_tag = self.load_index_to_tag_map(title)
            neighbors = [[index_to_tag.get(str(i), "Unknown") for i in row] for row in I]
            
        return neighbors

    def reverse_dict(self, original_dict):
        return {v: k for k, v in original_dict.items()}

    def save_deleted_ids(self, title, deleted_ids):
        deleted_ids_path = os.path.join(DATABASE_DIR, f"{title}-deleted-ids.json")
        with open(deleted_ids_path, 'w') as f:
            json.dump(deleted_ids, f)

    def load_deleted_ids(self, title):
        deleted_ids_path = os.path.join(DATABASE_DIR, f"{title}-deleted-ids.json")
        if os.path.exists(deleted_ids_path):
            with open(deleted_ids_path, 'r') as f:
                return json.load(f)
        return []

    def delete_entry_from_database(self, title, tags):
        db_path = os.path.join(DATABASE_DIR, f"{title}.bin")
        index = faiss.read_index(db_path)
        index_to_tag = self.load_index_to_tag_map(title)
        tag_to_index = self.reverse_dict(index_to_tag)
        deleted_ids = self.load_deleted_ids(title)  # Track deleted IDs

        if isinstance(tags, str):
            tags = [tags]

        # Prepare a list of tag indices to remove
        indices_to_remove = [int(tag_to_index[t]) for t in tags if t in tag_to_index]

        if indices_to_remove:
            id_selector = faiss.IDSelectorBatch(np.array(indices_to_remove, dtype=np.int64))
            index.remove_ids(id_selector)

            # Remove tags from index-to-tag map and track the deleted IDs
            for t in tags:
                if t in tag_to_index:
                    tag_index = tag_to_index[t]
                    del index_to_tag[tag_index]  # Remove from tag map
                    deleted_ids.append(tag_index)  # Track deleted ID for reuse

            faiss.write_index(index, db_path)
            self.save_index_to_tag_map(title, index_to_tag)
            self.save_deleted_ids(title, deleted_ids)  # Save updated list of deleted IDs

