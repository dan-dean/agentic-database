import queue
import threading
from datetime import datetime
from agentic_db.orchestrator import Orchestrator


class AsyncAgenticDatabase:
    def __init__(self):
        self.document_queue = queue.Queue()
        self.prompt_queue = queue.Queue()
        self.currently_processing = None
        self.processing_start_time = None
        self.lock = threading.Lock()
        self.processing_thread = None  # No processing thread initially
        self.orchestrator = Orchestrator()
        self.default_database = None
    
    def add_document(self, document, db_file=None, callback=None):
        """Add a document to the document queue with an optional callback."""

        if db_file is None:
            if self.default_database is not None:
                db_file = self.default_database
            else:
                raise ValueError("Database must be provided if default database is not set.")
        
        document_queue_object = (document[0], db_file, document[1])
        
        with self.lock:
            self.document_queue.put((document_queue_object, callback))
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

    def set_new_system_prompt(self, new_prompt):
        return self.orchestrator.set_new_system_prompt(new_prompt)

    def get_existing_databases(self):
        return self.orchestrator.get_existing_databases()
    
    def get_all_original_documents(self, db_file = None):
        if db_file is None:
            db_file = self.default_database
        return self.orchestrator.get_all_original_documents(db_file)
    
    def get_database_custom_prompt(self, db_file = None):
        if db_file is None:
            db_file = self.default_database
        return self.orchestrator.get_database_custom_prompt(db_file)
    
    def set_database_custom_prompt(self, new_prompt, db_file = None):
        if db_file is None:
            db_file = self.default_database
        return self.orchestrator.set_database_custom_prompt(db_file, new_prompt)
    
    def update_database_title(self, db_file, new_title):
        return self.orchestrator.update_database_title(db_file, new_title)
    
    def delete_database(self, db_file):
        return self.orchestrator.delete_database(db_file)

    def create_database(self, title):
        return self.orchestrator.create_database(title)

    def get_number_of_documents(self, db_file):
        return self.orchestrator.get_number_of_documents(db_file)

    def get_original_documents_from_textual_match(self, db_file, search_text):
        return self.orchestrator.get_original_documents_from_textual_match(db_file, search_text)

    def remove_original_document(self, db_file, doc_uuid):
        return self.orchestrator.remove_original_document(db_file, doc_uuid)

    def get_all_tags(self, db_file):
        return self.orchestrator.get_all_tags(db_file)
    
    def set_default_database(self, db_file):
        self.default_database = db_file
        print(self.default_database)

    def change_mode(self, mode):
        self.orchestrator.change_mode(mode)
        if mode == "single_query":
            if self.processing_thread is not None and self.status()["status"] == "Idle":
                self.processing_thread.join()
                self.processing_thread = None
    
    def get_mode(self):
        return self.orchestrator.get_mode()

    def clear_conversation_history(self):
        return self.orchestrator.clear_conversation_history()

    def load_conversation_history(self):
        return self.orchestrator.load_conversation_history()

    def process_queues(self):
        """Main processing method that checks the prompt queue first, then documents."""
        self.orchestrator.llm_handler.get_model()
        while True:
            # Check for prompts first
            if not self.prompt_queue.empty():
                prompt_text, database_title = None, None
                prompt, callback = self.prompt_queue.get()
                with self.lock:
                    prompt_text, database_title = prompt[0], prompt[1]

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
                document_queue_object, callback = self.document_queue.get()
                with self.lock:
                    document_text, database_title, file_path = document_queue_object[0], document_queue_object[1], document_queue_object[2]
                    self.currently_processing = f"Document: {file_path}"
                    self.processing_start_time = datetime.now()

                if database_title is None:
                    database_title = self.default_database
                print(database_title)

                self.orchestrator.process_document(document_text, database_title, file_path)
                
                # Call the callback function if provided
                response = {
                    "document": file_path,
                    "document_text" : document_text[:20],
                    "time_spent": str(datetime.now() - self.processing_start_time)
                }
                if callback:
                    callback(response)
            
            elif self.orchestrator.get_mode() == "single_query":
                # Both queues are empty, so shut down the thread
                print("Both queues are empty. Shutting down processor.")
                self.orchestrator.llm_handler.release_model()
                self.orchestrator.tag_handler.release_model()
                self.currently_processing = None
                self.processing_start_time = None
                return