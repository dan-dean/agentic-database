from llm_handler import LLMHandler
from tag_database_handler import TagDatabaseHandler
from doc_database_handler import *
from collections import Counter


class Orchestrator:
    def __init__(self):
        self.llm_handler = LLMHandler()
        self.tag_handler = TagDatabaseHandler()
        self.mode = "single_query"
        self.system_prompt = '''You are a knowledgeable chatbot that answers questions and assists users. You have access to a hybrid database tool built with SQL and a Vector DB.
        Your database uses agentic LLM models that can create roadmaps to answer problems. When retrieving data from the database, if the answer is not present in the provided
        data, candidly state as much. Be clear, effective, and succinct in your responses while also fully explaining requested concepsts.'''
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def get_existing_databases(self):
        return get_existing_databases()

    def set_new_system_prompt(self, new_prompt):
        self.system_prompt = new_prompt
        self.clear_conversation_history()
    
    def update_database_title(self, db_file, new_title):
        return update_database_title(db_file, new_title)
    
    def delete_database(self, db_file):
        self.tag_handler.delete_database(db_file)
        return delete_database(db_file)
    
    def create_database(self, title):
        db_file = create_database(title)
        self.tag_handler.create_database(db_file)
        return db_file

    def get_number_of_documents(self, db_file):
        return get_number_of_documents(db_file)

    def get_original_documents_from_textual_match(db_file, search_text):
        return get_original_documents_from_textual_match(db_file, search_text)

    def remove_original_document(self, db_file, doc_uuid):
        return remove_original_document(db_file, doc_uuid)

    def get_all_tags(self, db_file):
        return get_all_tags(db_file)

    def process_document(self, document, db_file):
        self.tag_handler.release_model()
        subdocs = self.llm_handler.break_up_and_summarize_text(document)

        subdocs_tags = []

        for subdoc in subdocs:
            for tag in subdoc["tags"]:
                subdocs_tags.append(tag)
        
        self.llm_handler.release_model()

        self.tag_handler.add_entry_to_database(db_file, subdocs_tags)

        add_entry_to_database(db_file, document, subdocs)

    def change_mode(self, mode):
        if mode not in ["chat_mode", "single_query"]:
            raise ValueError("Mode must be either 'chat_mode' or 'single_query'")
        if mode == "chat_mode":
            self.clear_conversation_history()
        self.mode = mode

    def get_mode(self):
        return self.mode

    def database_query(self, conversation_history, prompt, database_title):
        roadmap = self.llm_handler.generate_roadmap(prompt)

        context = []

        for step in roadmap:
            print(step[1])

            #get real tags from prospective
            real_tags = self.tag_handler.get_nearest_neighbors(database_title, step[0], 10)

            self.tag_handler.release_model()

            #get relevant tags from real

            real_tags_pool = []

            for tag_list in real_tags:
                for tag in tag_list:
                    real_tags_pool.append(tag)

            relevant_tags = self.llm_handler.return_relevant_tags(step[1], real_tags_pool)

            if len(relevant_tags) == 0 or ( len(relevant_tags) == 1 and relevant_tags[0] == "nothing" ):
                continue

            print("relevant tags: ", relevant_tags)

            #get doc uuids from relevant tags

            doc_uuids, doc_tags = get_document_uuid_tags_from_tags(database_title, relevant_tags)

            print("found ", len(doc_uuids), " documents with relevant tags")
            print("reading document with tags: ", doc_tags[0])

            doc_text = get_document_text_from_uuid(database_title, doc_uuids[0])

            context.append(doc_text)
        
        answer = self.llm_handler.generate_response_with_context(conversation_history, context)

        return answer, context

    def clear_conversation_history(self):
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def load_conversation_history(self, conversation_history):
        self.conversation_history = conversation_history

    def process_prompt(self, prompt, database_title):
        self.tag_handler.release_model()

        answer, context = None, None

        if self.mode == "single_query":
            query_history = self.conversation_history+[{"role": "user", "content": prompt}]
        
            answer, context = self.database_query(query_history, prompt, database_title)

        elif self.mode == "chat_mode":
            self.conversation_history.append({"role": "user", "content": prompt})
            if self.llm_handler.decide_to_respond_or_use_tool(self.conversation_history) == "no":
                answer, context = self.database_query(self.conversation_history, prompt, database_title)
            else:
                answer = self.llm_handler.generate_response(self.conversation_history)
            
            self.conversation_history.append({"role": "assistant", "content": answer})
        
        return answer, context