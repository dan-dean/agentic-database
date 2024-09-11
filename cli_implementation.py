import shlex
import os
import async_agentic_database

async_agentic_database = async_agentic_database.AsyncAgenticDatabase()

# Functions for each command
def help_command():
    print("""
Available commands:
- help
- list_databases
- set_default_database [database_number]
- create_database [database_name]
- delete_database [database_number]
- add_document [document_path] [database_number]
- send_query [query] [database_number]
- queue_size
- status
- start_thread [database_number]
- exit
    """)

def list_databases():
    database = async_agentic_database.get_existing_databases()
    for db in database:
        print("1. ", db["title"], " - ", db["last_modified"])

def set_default_database(db_number=None):
    global default_database
    if not db_number:
        list_databases()
        db_number = int(input("Enter the database # to set as default: ")) - 1
    else:
        db_number=int(db_number) - 1

    databases = async_agentic_database.get_existing_databases()
    
    if db_number in range(len(databases)):
        default_database = databases[db_number]["db_file"]
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
        if confirm.lower() != "y":
            db_name = databases[db_number]["title"]
            async_agentic_database.delete_database(databases[db_number]["db_file"])
            print(f"Database '{db_name}' deleted.")
    else:
        print(f"Database '{db_number}' not found.")

def add_document(doc_name=None, db_number=None):
    databases = async_agentic_database.get_existing_databases()

    if db_number is not None:
        try:
            db_number = int(db_number) - 1
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number and 0 <= db_number < len(databases):
        db_file = databases[db_number]["db_file"]
    else:
        print(f"Database '{db_number}' not found.")
        return

    if not doc_name:
        doc_name = input("Enter the document filepath: ")

    if not os.path.isfile(doc_name):
        print(f"File '{doc_name}' does not exist.")
        return

    try:
        with open(doc_name, 'r') as file:
            document_text = file.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    def callback(message):
        print(message,"\n")

    if db_number is not None:
        document = [document_text, db_file]
        async_agentic_database.add_document(document, callback)
    else:
        async_agentic_database.add_document(document_text, callback)

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
        db_file = databases[db_number]["db_file"]
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
        async_agentic_database.add_prompt(query, callback)


def start_thread(db_number=None):
    databases = async_agentic_database.get_existing_databases()

    if db_number is not None:
        try:
            db_number = int(db_number)
        except ValueError:
            print("Invalid database number. Please enter a valid number.")
            return

    if db_number is not None and 0 <= db_number < len(databases):
        db_file = databases[db_number]["db_file"]
        print(f"Starting thread in database: {databases[db_number]['title']}")
    else:
        if db_number is not None:
            print(f"Database '{db_number}' not found.")
            return
        print("Starting thread in default database.")

    print("Type 'exit' to exit chat mode.")

    # chat mode
    async_agentic_database.change_mode("chat_mode")
    while True:
        user_input = input("You: ").strip()
        
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
    elif command == "list_databases":
        list_databases()
    elif command == "set_default_database":
        set_default_database(args[1] if len(args) > 1 else None)
    elif command == "create_database":
        create_database(args[1] if len(args) > 1 else None)
    elif command == "delete_database":
        delete_database(args[1] if len(args) > 1 else None)
    elif command == "add_document":
        if len(args) == 3:
            add_document(args[1], args[2])
        elif len(args) == 2:
            add_document(args[1], None)
        else:
            add_document(None, None)
    elif command == "queue_size":
        queue_size()
    elif command == "status":
        status()
    elif command == "send_query":
        send_query(" ".join(args[1:]) if len(args) > 1 else None)
    elif command == "start_thread":
        start_thread(" ".join(args[1:]) if len(args) > 1 else None)
    elif command == "exit":
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