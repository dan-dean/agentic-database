import queue
import threading
import time
from datetime import datetime
from orchestrator import Orchestrator

class MultiQueueProcessor:
    def __init__(self):
        self.document_queue = queue.Queue()
        self.prompt_queue = queue.Queue()
        self.currently_processing = None
        self.processing_start_time = None
        self.lock = threading.Lock()
        self.processing_thread = None  # No processing thread initially
        self.orchestrator = Orchestrator()
        self.default_database = None
    
    def add_document(self, document, callback=None):
        """Add a document to the document queue with an optional callback."""

        if not isinstance(document, (list, tuple)):
            if self.default_database is not None:
                document = (document, self.default_database)
            else:
                raise ValueError("Database must be provided if default database is not set.")
        
        with self.lock:
            self.document_queue.put((document, callback))
            self.get_process()
    
    def add_prompt(self, prompt, callback=None):
        """Add a prompt to the prompt queue with an optional callback."""
        if not isinstance(prompt, (list, tuple)):
            if self.default_database is not None:
                prompt = (prompt, self.default_database)
            else:
                raise ValueError("Database must be provided if default database is not set.")

        with self.lock:
            self.prompt_queue.put((prompt, callback))
            self.get_process()
    
    def queue_size(self):
        """Return the combined size of both queues."""
        return self.document_queue.qsize(), self.prompt_queue.qsize()

    def status(self):
        """Return the status of the current task being processed."""
        if self.currently_processing:
            time_spent = datetime.now() - self.processing_start_time
            return {
                "status": "Processing",
                "current_task": self.currently_processing,
                "time_spent": str(time_spent)
            }
        else:
            return {"status": "Idle"}

    def get_process(self):
        """Start the processing thread if it's not running."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            print("Starting a new processing thread.")
            self.processing_thread = threading.Thread(target=self.process_queues)
            self.processing_thread.daemon = True  # Ensure the thread ends with the main program
            self.processing_thread.start()

    def split(self, queue_item):
        return queue_item[0], queue_item[1]

    def get_existing_databases(self):
        return orchestrator.get_existing_databases()
    
    def update_database_title(self, db_file, new_title):
        return orchestrator.update_database_title(db_file, new_title)
    
    def delete_database(self, db_file):
        return orchestrator.delete_database(db_file)

    def get_number_of_documents(self, db_file):
        return orchestrator.get_number_of_documents(database_title)

    def get_original_documents_from_textual_match(db_file, search_text):
        return orchestrator.get_original_documents_from_textual_match(db_file, search_text)

    def remove_original_document(self, db_file, doc_uuid):
        return orchestrator.remove_original_document(db_file, doc_uuid)

    def get_all_tags(self, db_file):
        return orchestrator.get_all_tags(db_file)
    
    def set_default_database(self, db_file):
        self.default_database = db_file

    def process_queues(self):
        """Main processing method that checks the prompt queue first, then documents."""
        while True:
            # Check for prompts first
            if not self.prompt_queue.empty():
                prompt_text, database_title = None, None
                prompt, callback = self.prompt_queue.get()
                with self.lock:
                    prompt_text, database_title = self.split(prompt)

                    self.currently_processing = f"Prompt: {prompt_text}"
                    self.processing_start_time = datetime.now()

                if database_title is None:
                    database_title = self.default_database

                answer, context = self.orchestrator.process_prompt(prompt_text, database_title)

                # create a response object
                response = {
                    "prompt": prompt_text,
                    "response": answer,
                    "context": context
                }

                # Call the callback function if provided
                if callback:
                    callback(response)
            
            # Check for documents if no prompts
            elif not self.document_queue.empty():
                document_text, database_title = None, None
                document, callback = self.document_queue.get()
                with self.lock:
                    document_text, database_title = self.split(document)
                    self.currently_processing = f"Document: {document_text}"
                    self.processing_start_time = datetime.now()

                if database_title is None:
                    database_title = self.default_database

                self.orchestrator.process_document(document_text, database_title)
                
                # Call the callback function if provided
                if callback:
                    callback(f"Processed document: {document_text}")
            
            else:
                # Both queues are empty, so shut down the thread
                print("Both queues are empty. Shutting down processor.")
                self.llm_handler.release_model()
                self.tag_handler.release_model()
                self.currently_processing = None
                self.processing_start_time = None
                return