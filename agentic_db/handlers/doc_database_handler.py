import os
import sqlite3
import uuid
from datetime import datetime

DATABASE_DIR = 'databases/docs'

'''
The SQL doc database handler module for agentic database. Provides simple operations with some specificity to the 
system as a whole (such as table creation). The module provides the following functions:

- create_database: creates a new database with the given title and returns the path to the database file.
    return: str

- get_existing_databases: returns a list of existing databases with their metadata.
    return: [{'file': str, 'last_modified': str, 'title': str}]
- update_database_title: updates the title of the database with the given file name.
    return: bool
- delete_database: deletes the database with the given file name.
    return: bool

- add_entry_to_database: adds an entry to the database with the given file name. Increments all tag instances.
    return: bool
- get_all_tags: returns a list of all tags in the database with the given file name.
    return: [str] | None
- get_document_uuid_tags_from_tag: returns a list of document UUIDs and a list of of tags that have the given tag(s).
    return: [str], [str] | None
- get_document_text_from_uuid: returns the text of the document with the given UUID.
    return: str | None
- get_original_document_from_document_uuid: returns the original document of the document with the given UUID.
    return: (str, str, str, str) | None

- get_original_documents_from_textual_match: returns a list of original documents that contain the given text.
    return: [(str, str, str, str)] | None

- get_documents_uuids_from_original_document: returns a list of document UUIDs that are derived from the original document with the given UUID.
    return: [str] | None
- remove_original_document: removes the original document with the given UUID from the database with the given file name.
    It is a deep delete, removing all sub-documents as well. Also removes all tags associated with the document that 
    hit zero instances. Returns a list of tags that were deleted or None if the document didn't exist.
    return: [str] | None

Databases have metadata stored in the metadata table about modification date and title for ordered display
purposes.

'''

def create_database_dir():
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)

def create_database(title):
    create_database_dir()
    db_uuid = str(uuid.uuid4())
    db_path = os.path.join(DATABASE_DIR, f"{db_uuid}.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute('''
    CREATE TABLE original_documents (
        uuid TEXT PRIMARY KEY,
        text TEXT,
        pdf TEXT,
        youtube_url TEXT,
        document_type TEXT,
        file_path TEXT
    )''')
    
    cursor.execute('''
    CREATE TABLE documents (
        uuid TEXT PRIMARY KEY,
        original_uuid TEXT,
        text TEXT,
        FOREIGN KEY (original_uuid) REFERENCES original_documents (uuid)
    )''')
    
    cursor.execute('''
    CREATE TABLE tags (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag TEXT UNIQUE,
        instances INTEGER DEFAULT 0
    )''')
    
    cursor.execute('''
    CREATE TABLE document_tags (
        document_uuid TEXT,
        tag_id INTEGER,
        PRIMARY KEY (document_uuid, tag_id),
        FOREIGN KEY (document_uuid) REFERENCES documents (uuid),
        FOREIGN KEY (tag_id) REFERENCES tags (id)
    )''')
    
    cursor.execute('''
    CREATE TABLE metadata (
        last_modified TEXT,
        title TEXT,
        custom_prompt TEXT
    )''')
    
    # Insert initial metadata
    last_modified = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO metadata (last_modified, title) VALUES (?, ?)
    ''', (last_modified, title))
    
    conn.commit()
    conn.close()
    
    return db_uuid + ".db"

def update_last_modified(db_file, cursor):
    last_modified = datetime.now().isoformat()
    cursor.execute('''
    UPDATE metadata SET last_modified = ? WHERE 1
    ''', (last_modified,))

def set_custom_prompt(db_file, custom_prompt):
    db_path = os.path.join(DATABASE_DIR, db_file)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
    UPDATE metadata SET custom_prompt = ? WHERE 1
    ''', (custom_prompt,))

    conn.commit()
    conn.close()

def get_custom_prompt(db_file):
    db_path = os.path.join(DATABASE_DIR, db_file)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT custom_prompt FROM metadata')
    custom_prompt = cursor.fetchone()
    conn.close()
    return custom_prompt[0] if custom_prompt else None



def get_existing_databases():
    databases = []
    
    if os.path.exists(DATABASE_DIR):
        for db_file in os.listdir(DATABASE_DIR):
            if db_file.endswith(".db"):
                db_path = os.path.join(DATABASE_DIR, db_file)
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute('SELECT last_modified, title FROM metadata')
                metadata = cursor.fetchone()
                
                if metadata:
                    last_modified, title = metadata
                    databases.append({
                        'file': db_file,
                        'last_modified': last_modified,
                        'title': title
                    })
                
                conn.close()
    
    # Sort by last modified date, most recent first
    databases.sort(key=lambda x: x['last_modified'], reverse=True)
    return databases

def update_database_title(db_file, new_title):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        last_modified = datetime.now().isoformat()
        cursor.execute('''
        UPDATE metadata SET title = ?, last_modified = ? WHERE 1
        ''', (new_title, last_modified))
        
        conn.commit()
        conn.close()
        return True
    else:
        return False

def delete_database(db_file):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        return True
    else:
        return False
    
def add_entry_to_database(db_file, text, sub_docs, pdf="null", youtube_url="null", document_type="text", file_path="null"):
    print(db_file)
    print(DATABASE_DIR)
    db_path = os.path.join(DATABASE_DIR, db_file)
    print(file_path)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        uuid_str = str(uuid.uuid4())
        cursor.execute('''
        INSERT INTO original_documents (uuid, text, pdf, youtube_url, document_type, file_path) VALUES (?, ?, ?, ?, ?, ?)
        ''', (uuid_str, text, pdf, youtube_url, document_type, file_path))

        for sub_doc in sub_docs:
            sub_doc_uuid = str(uuid.uuid4())
            cursor.execute('''
            INSERT INTO documents (uuid, original_uuid, text) VALUES (?, ?, ?)
            ''', (sub_doc_uuid, uuid_str, sub_doc['subdoc_text']))

            for tag in sub_doc['tags']:
                cursor.execute('''
                SELECT id, instances FROM tags WHERE tag = ?
                ''', (tag,))
                
                tag_row = cursor.fetchone()

                if tag_row:
                    tag_id, instances = tag_row
                    cursor.execute('''
                    UPDATE tags SET instances = ? WHERE id = ?
                    ''', (instances + 1, tag_id))
                else:
                    cursor.execute('''
                    INSERT INTO tags (tag, instances) VALUES (?, ?)
                    ''', (tag, 1))
                    tag_id = cursor.lastrowid


                cursor.execute('''
                INSERT INTO document_tags (document_uuid, tag_id) VALUES (?, ?)
                ''', (sub_doc_uuid, tag_id))
        
        update_last_modified(db_file, cursor)
        
        conn.commit()
        conn.close()
        return True
    else:
        return False
    
def get_all_tags(db_file):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT tag FROM tags')
        tags = cursor.fetchall()
        
        conn.close()
        return [tag[0] for tag in tags]
    else:
        return None

def get_all_original_document_file_paths(db_file):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT file_path FROM original_documents')
        file_paths = cursor.fetchall()
        
        conn.close()
        return [file_path[0] for file_path in file_paths]
    else:
        return None

# tag may be a string or a list of strings
def get_document_uuid_tags_from_tags(db_file, tags):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        uuids_with_tags = {}

        if isinstance(tags, list) and len(tags) > 0:
            primary_tag = tags[0]  # First tag is the primary one
            all_tags = tags  # Consider all tags for filtering
            
            placeholder = ', '.join(['?'] * len(all_tags))
            query = f'''
            SELECT documents.uuid, tags.tag FROM documents
            JOIN document_tags ON documents.uuid = document_tags.document_uuid
            JOIN tags ON document_tags.tag_id = tags.id
            WHERE tags.tag IN ({placeholder})
            '''
            cursor.execute(query, tuple(all_tags))
            results = cursor.fetchall()

            for doc_uuid, tag in results:
                if doc_uuid not in uuids_with_tags:
                    uuids_with_tags[doc_uuid] = []
                
                uuids_with_tags[doc_uuid].append(tag)

            filtered_uuids_with_tags = {uuid: tags for uuid, tags in uuids_with_tags.items() if primary_tag in tags}

            sorted_documents = sorted(filtered_uuids_with_tags.items(), key=lambda x: len(x[1]), reverse=True)

            uuids = [doc[0] for doc in sorted_documents]
            tags_matched = [doc[1] for doc in sorted_documents]

            conn.close()
            return uuids, tags_matched
        
        else:
            raise ValueError("Tags must be a non-empty list")
    else:
        return None, None

def get_document_text_from_uuid(db_file, uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT text FROM documents WHERE uuid = ?
        ''', (uuid,))
        
        text = cursor.fetchone()
        conn.close()
        
        return text[0] if text else None
    else:
        return None

def get_original_document_from_document_uuid(db_file, document_uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT original_uuid FROM documents WHERE uuid = ?
        ''', (document_uuid,))
        
        original_uuid = cursor.fetchone()
        conn.close()
        
        return get_original_document_from_uuid(db_file, original_uuid[0]) if original_uuid else None
    else:
        return None
    
def get_original_document_from_uuid(db_file, uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT text, pdf, youtube_url, document_type FROM original_documents WHERE uuid = ?
        ''', (uuid,))
        
        original_document = cursor.fetchone()
        conn.close()
        
        return original_document if original_document else None
    else:
        return None

def get_original_documents_from_textual_match(db_file, matching_text):
    db_path = os.path.join(DATABASE_DIR, db_file)

    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT uuid, text, pdf, youtube_url, document_type 
        FROM original_documents 
        WHERE text LIKE ?
        ''', ('%' + matching_text + '%',))

        original_documents = cursor.fetchall()
        conn.close()

        return original_documents if original_documents else []
    else:
        return []

def get_documents_uuids_from_original_document(db_file, original_uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT uuid FROM documents
        WHERE original_uuid = ?
        ''', (original_uuid,))
        
        uuids = cursor.fetchall()
        conn.close()
        
        return [uuid[0] for uuid in uuids] if uuids else []
    else:
        return None

def remove_document(db_file, uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT tag_id FROM document_tags WHERE document_uuid = ?
        ''', (uuid,))
        tag_ids = cursor.fetchall()
        
        deleted_tags = []
        
        for tag_id_tuple in tag_ids:
            tag_id = tag_id_tuple[0]

            cursor.execute('''
            SELECT tag, instances FROM tags WHERE id = ?
            ''', (tag_id,))
            tag_row = cursor.fetchone()
            if not tag_row:
                continue
            
            tag_name, instances = tag_row
            
            if instances > 1:
                cursor.execute('''
                UPDATE tags SET instances = ? WHERE id = ?
                ''', (instances - 1, tag_id))
            else:
                deleted_tags.append(tag_name)
                
                cursor.execute('''
                DELETE FROM tags WHERE id = ?
                ''', (tag_id,))

        cursor.execute('''
        DELETE FROM document_tags WHERE document_uuid = ?
        ''', (uuid,))

        cursor.execute('''
        DELETE FROM documents WHERE uuid = ?
        ''', (uuid,))
        
        conn.commit()
        conn.close()
        
        return deleted_tags
    else:
        return None

def remove_original_document(db_file, original_uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        sub_document_uuids = get_documents_uuids_from_original_document(db_file, original_uuid)

        all_deleted_tags = []
        
        if sub_document_uuids:
            for sub_uuid in sub_document_uuids:
                deleted_tags = remove_document(db_file, sub_uuid)
                if deleted_tags:
                    all_deleted_tags.extend(deleted_tags)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            DELETE FROM original_documents
            WHERE uuid = ?
            ''', (original_uuid,))

            update_last_modified(db_file, cursor)
            
            conn.commit()
            conn.close()
            return all_deleted_tags
        return None
    
def get_all_document_uuids_from_tag(db_file, tag):
        db_path = os.path.join(DATABASE_DIR, db_file)
        
        if os.path.exists(db_path):
            if isinstance(tag, str):
                tag = [tag]

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            placeholder = ', '.join(['?'] * len(tag))
            query = f'''
            SELECT documents.uuid FROM documents
            JOIN document_tags ON documents.uuid = document_tags.document_uuid
            JOIN tags ON document_tags.tag_id = tags.id
            WHERE tags.tag IN ({placeholder})
            '''

            cursor.execute(query, tuple(tag))
            results = cursor.fetchall()
            conn.close()

            return [result[0] for result in results]
        else:
            return None

def get_tags_from_document_uuid(db_file, uuid):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT tags.tag FROM document_tags
        JOIN tags ON document_tags.tag_id = tags.id
        WHERE document_tags.document_uuid = ?
        ''', (uuid,))
        tags = cursor.fetchall()
        conn.close()

        return [tag[0] for tag in tags]
    else:
        return None
    
def get_number_of_original_documents(db_file):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT COUNT(*) FROM original_documents
        ''')
        count = cursor.fetchone()[0]
        conn.close()

        return count
    else:
        return None
    
def get_number_of_documents(db_file):
    db_path = os.path.join(DATABASE_DIR, db_file)
    
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
        SELECT COUNT(*) FROM documents
        ''')
        count = cursor.fetchone()[0]
        conn.close()

        return count
    else:
        return None
    