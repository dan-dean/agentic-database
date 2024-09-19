import shlex
import os
from agentic_db import async_agentic_database

async_agentic_database = async_agentic_database.AsyncAgenticDatabase()

async_agentic_database.set_new_system_prompt('''You are a knowledgeable chatbot that answers questions and assists users. You have access to a hybrid database tool built with SQL and a Vector DB.
        Your database uses agentic LLM models that can create roadmaps to answer problems. When retrieving data from the database, if the answer is not present in the provided
        data, candidly state as much. Defer with complete adherence to the information retrieved from the database over your own general knowledge. 
        Be clear, effective, and succinct in your responses while also fully explaining requested concepts. If the user asks how to exit the chat interface, tell them
        that the command they need to enter is 'exit'
        ''')
# Functions for each command
def help_command():
    print("""
Available commands:
- help (Show this help message)
- ls (Lists all databases)
- ls_db [database_number] (List all documents in a database)
- prompt [database_number] (Show the custom prompt for a database)
- set_prompt [database_number] [new_prompt] (Set a custom prompt for a database)
- set_db [database_number] (Set the default database)
- mk_db [database_name] (Create a new database)
- rm_db [database_number] (Delete a database)
- add [database_number] [document_path] (Add a document to the database)
- ask [query] [database_number] (Send a single query to the database)
- q_size (Get the size of the document and prompt queues)
- status (Get the current status of the system)
- thread [database_number] (Start a chat thread in the database)
- exit
    """)

def list_databases():
    databases = async_agentic_database.get_existing_databases()
    for i in range(len(databases)):
        print(f"{i + 1}. {databases[i]["title"]} - {databases[i]["last_modified"]}")

def list_docs(db_number=None):
    databases = async_agentic_database.get_existing_databases()

    if db_number is not None:
        try:
            db_number = int(db_number) - 1
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["file"]
    else:
        if db_number is not None:
            print(f"Database '{db_number}' not found.")
            return
    if db_number is not None:
        docs = async_agentic_database.get_all_original_documents(db_file)
    else:
        docs = async_agentic_database.get_all_original_documents()
    for i in range(len(docs)):
        print(f"{i + 1}. {docs[i]}")

def show_prompt(db_number=None):
    databases = async_agentic_database.get_existing_databases()

    if db_number is not None:
        try:
            db_number = int(db_number) - 1
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["file"]
    else:
        if db_number is not None:
            print(f"Database '{db_number}' not found.")
            return
        else: 
            db_file = async_agentic_database.default_database
    if db_number is not None:
        prompt = async_agentic_database.get_database_custom_prompt(db_file)
    else:
        prompt = async_agentic_database.get_database_custom_prompt()
    print(f"Prompt for database '{databases[db_number]['title']}': {prompt}")

def set_prompt(db_number=None, new_prompt=None):
    databases = async_agentic_database.get_existing_databases()

    if db_number is not None:
        try:
            db_number = int(db_number) - 1
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["file"]
    else:
        if db_number is not None:
            print(f"Database '{db_number}' not found.")
            return
        elif db_number is None:
            list_databases()
            db_number = int(input("Enter the database # to modify: ")) - 1

            
    print(f"Setting prompt for database '{databases[db_number]['title']}'.")
        
    if not new_prompt:
        new_prompt = input("Enter the new prompt: ")

    if db_number is not None:
        async_agentic_database.set_database_custom_prompt(new_prompt, db_file)
    else:
        async_agentic_database.set_database_custom_prompt(new_prompt)

    print(f"Prompt for database '{databases[db_number]['title']}' set to: {new_prompt}")

def set_default_database(db_number=None):
    global default_database
    if not db_number:
        list_databases()
        db_number = int(input("Enter the database # to set as default: ")) - 1
    else:
        db_number=int(db_number) - 1

    databases = async_agentic_database.get_existing_databases()
    
    if db_number in range(len(databases)):
        async_agentic_database.set_default_database(databases[db_number]["file"])
        print(f"Default database set to {databases[db_number]["title"]}.")
    else:
        print(f"Database '{db_number}' not found.")

def create_database(db_name=None):
    if not db_name:
        db_name = input("Enter the name for the new database: ")
    
    async_agentic_database.create_database(db_name)

def delete_database(db_number=None):
    if not db_number:
        list_databases()
        db_number = int(input("Enter the database # to delete: ")) - 1
    else:
        db_number=int(db_number) - 1

    databases = async_agentic_database.get_existing_databases()

    if db_number in range(len(databases)):
        confirm = input(f"Are you sure you want to delete database '{databases[db_number]['title']}'? (y/n)")
        if confirm.lower() == "y":
            db_name = databases[db_number]["title"]
            async_agentic_database.delete_database(databases[db_number]["file"])
            print(f"Database '{db_name}' deleted.")
        else:
            print("Deletion cancelled.")
    else:
        print(f"Database '{db_number}' not found.")

def add_document(db_number=None, doc_name=None):
    databases = async_agentic_database.get_existing_databases()
    db_file = None

    if db_number is not None:
        try:
            db_number = int(db_number) - 1
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["file"]
    elif db_number:
        print(f"Database '{db_number}' not found.")
        return

    if not doc_name:
        doc_name = input("Enter the document filepath: ")

    if not os.path.isfile(doc_name):
        print(f"File '{doc_name}' does not exist.")
        return

    try:
        with open(doc_name, 'r', encoding='utf-8', errors='ignore') as file:
            document_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    def callback(response):
        doc_name = response.get("document", "N/A")
        time_spent = response.get("time_spent", "N/A")
        doc_text = response.get("document_text", "N/A")

        print(f"Document added: {doc_name}")
        print(doc_text)
        print(f"Time spent: {time_spent}")

    document = [document_text, doc_name]
    if db_file is not None:
        async_agentic_database.add_document(document, db_file=db_file, callback=callback)
    else:
        try:
            async_agentic_database.add_document(document, callback=callback)
        except Exception as e:
            print(f"Error adding document: {e}")

def queue_size():
    queue_sizes = async_agentic_database.queue_size()

    print(f"Document queue size: {queue_sizes[0]}")
    print(f"Prompt queue size: {queue_sizes[1]}")

def status():
    print(async_agentic_database.status())

def send_query(query=None, db_number=None):
    databases = async_agentic_database.get_existing_databases()

    # Validate and handle db_number input
    if db_number is not None:
        try:
            db_number = int(db_number)
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["file"]
    else:
        if db_number is not None:
            print(f"Database '{db_number}' not found.")
            return

    if not query:
        query = input("Enter your query: ")

    def callback(response):
        prompt_text = response.get("prompt", "N/A")
        answer = response.get("response", "No response available")
        # context = response.get("context", "No context available")

        print(f"Prompt: {prompt_text}")
        print(f"Response: {answer}")
        # print(f"Context: {context}\n")

    if db_number is not None:
        packaged_query = [query, db_file]
        async_agentic_database.add_prompt(packaged_query, callback)
    else:
        try:
            async_agentic_database.add_prompt(query, callback)
        except Exception as e:
            print(f"Error sending query: {e}")


def start_thread(db_number=None):
    databases = async_agentic_database.get_existing_databases()

    if db_number is not None:
        try:
            db_number = int(db_number) - 1
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["file"]
        print(f"Starting thread in database: {databases[db_number]['title']}")
    else:
        if db_number is not None:
            print(f"Database '{db_number}' not found.")
            return
        print("Starting thread in default database.")

    print("Type 'exit' to exit chat mode.")

    # chat mode
    async_agentic_database.change_mode("chat_mode")
    # load the llm model in expectation of a prompt
    async_agentic_database.get_process()
    while True:
        user_input = input().strip()
        
        if user_input.lower() == "exit":
            print("Exiting chat mode.")
            async_agentic_database.clear_conversation_history()
            async_agentic_database.change_mode("single_query")
            break

        # Pass the user input as a message to the conversation in the agentic database system
        def chat_callback(response):
            prompt_text = response.get("prompt", "N/A")
            reply = response.get("response", "No response available")
            print(f"Agent: {reply}")

        if db_number is not None:
            chat_input = [user_input, db_file]  # Package message with the db_file
            async_agentic_database.add_prompt(chat_input, chat_callback)
        else:
            async_agentic_database.add_prompt(user_input, chat_callback)

    

    
    

# Command dispatcher
def handle_command(command_input):
    args = shlex.split(command_input)  # Safely parse input with quoted arguments
    if not args:
        return

    command = args[0]

    # Match commands
    if command == "help":
        help_command()
    elif command == "ls":
        list_databases()
    elif command == "ls_db":
        list_docs(args[1] if len(args) > 1 else None)
    elif command == "prompt":
        show_prompt(args[1] if len(args) > 1 else None)
    elif command == "set_prompt":
        set_prompt(args[1] if len(args) > 1 else None, args[2] if len(args) > 2 else None)
    elif command == "set_db":
        set_default_database(args[1] if len(args) > 1 else None)
    elif command == "mk_db":
        create_database(args[1] if len(args) > 1 else None)
    elif command == "rm_db":
        delete_database(args[1] if len(args) > 1 else None)
    elif command == "add":
        if len(args) == 3:
            add_document(args[1], args[2])
        elif len(args) == 2:
            add_document(args[1], None)
        else:
            add_document(None, None)
    elif command == "q_size":
        queue_size()
    elif command == "status":
        status()
    elif command == "ask":
        send_query(" ".join(args[1:]) if len(args) > 1 else None)
    elif command == "thread":
        start_thread(" ".join(args[1:]) if len(args) > 1 else None)
    elif command == "exit":
        if async_agentic_database.status()["status"] == "Processing":
            print("Please wait for the current process to finish.")
            return True
        print("Exiting...")
        return False
    else:
        print(f"Unknown command: {command}")
    return True

# Main loop to handle CLI input
def main():
    print("CLI started. Type 'help' for a list of commands.")
    while True:
        command_input = input("> ").strip()
        if not handle_command(command_input):
            break

if __name__ == "__main__":
    main()