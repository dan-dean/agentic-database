import llama_cpp
from huggingface_hub import hf_hub_download
import os
import contextlib
import json

MODELS_DIR = "models\\llm"

model_file_name = os.path.join(MODELS_DIR, "model_file_name.json")

subdoc_char_limit = 5000

'''
The LLMHandler class is a wrapper for the LLM model. It provides methods to interact with the model relevant to the larger agentic database use case
such as generating tags, generating roadmaps, and generating responses with context. The class also handles the downloading of the model and
the creation of the model object. The model object is a singleton object that is created once and then reused for all subsequent requests. 
A function to unload the model from memory is made available to free up resources when the model is no longer needed or when other modes of the 
application are being used. 

Grammars in llama are on a token basis and so the grammar needs to be constructed from the tokens that are generated from the model.
The module provides the following functions:

- get_model: returns the model object, creates one and loads it into memory if it does not exist
    return: llama_cpp.Llama object
- release_model: releases the model from memory
    return: None

- get_token_count: returns the number of tokens in a given text
    return: int

- return_relevant_tags: returns the relevant actual tags for a given text based on a list of possibly related or unrelated actual tags
    return: [str]
- generate_tags: generates tags for a given text. Returns a list of tags
    return: [str]

- generate_roadmap: generates a roadmap for a given text. Returns a list of steps to retrieve information from the database. Steps are made up of
a list of tags to search for and an explanation of the query.
    return: [[[str], str]]

- generate_response_with_context: generates a response for a given text using the retrieved documents from the database. Returns the response.
    return: str
'''

generic_tag_grammar_text = """
root ::= tags
tags ::= tag ("," tag)*
tag ::= alphanumeric | "_"
alphanumeric ::= [a-zA-Z0-9]+
"""

roadmap_schema = {
  "type": "object",
  "properties": {
    "steps": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "query": {
            "type": "string",
            "pattern": "^[a-z0-9_]+(,[a-z0-9_]+)*$",
            "description": "A comma-delimited set of alphanumeric lowercase strings with underscores, representing database queries."
          },
          "explanation": {
            "type": "string",
            "description": "A string explaining the purpose of the query."
          }
        },
        "required": ["query", "explanation"]
      }
    }
  },
  "required": ["steps"]
}

choice_schema = {
  "type": "object",
  "properties": {
    "choice": {
      "type": "string",
      "enum": ["no", "yes"]
    }
  },
  "required": ["choice"],
  "additionalProperties": False
}

subject_list_schema = {
    "type": "object",
    "properties": {
        "subjects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "A subject or concept found in the document."
                    }
                },
                "required": ["subject"]
            }
        }
    },
    "required": ["subjects"]
}

subdoc_schema = {
    "type": "object",
    "properties": {
        "subdoc_text": {
            "type": "string",
            "maxLength": subdoc_char_limit,
            "description": "The subdoc text, mostly quoting from the source with minimal paraphrasing."
        },
        "tags": {
            "type": "array",
            "items": {
                "type": "string",
                "pattern": "^[a-z0-9_]+$",
                "description": "A tag describing the subject or concept found in the subdoc."
            },
            "minItems": 1
        }
    },
    "required": ["subdoc_text", "tags"]
}

class LLMHandler:

    #singleton model
    _model = None
    generic_tag_grammar = None
                              
    def __init__(self):
        if not os.path.exists(model_file_name):
            print("Downloading model...")
            llama_large_location = hf_hub_download(repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", 
                            filename="Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf", 
                            cache_dir=MODELS_DIR)
            # llama_large_location = hf_hub_download(repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF", 
            #     filename="Llama-3.2-3B-Instruct-f16.gguf", 
            #     cache_dir=MODELS_DIR)

            #store model location in a json file
            model_json = {"model_file": llama_large_location}
            with open(model_file_name, "w") as f:
                json.dump(model_json, f)
        else:
            print("Model already downloaded.")
        with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
            self.generic_tag_grammar = llama_cpp.LlamaGrammar.from_string(generic_tag_grammar_text)

#TODO: add big vs small model selection and figure out where smaller models could be used in functionality.
    def get_model(self, size="big"):
        if self._model is None:
            with open(model_file_name, "r") as f:
                model_json = json.load(f)
            self._model = llama_cpp.Llama(model_json["model_file"],
                                            n_gpu_layers=-1,
                                            n_ctx=30000,
                                            flash_attn=True,
                                            type_k=8,
                                            type_v=8,
                                            verbose=False
            )
            # # (huggingface reports 292 tensors when 291 for lmstudio's 3.1 8B)
            # self._model = llama_cpp.Llama("models\\llm\\models--lmstudio-community--Meta-Llama-3.1-8B-Instruct-GGUF\\snapshots\\8601e6db71269a2b12255ebdf09ab75becf22cc8\\Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
            #                                 n_gpu_layers=-1,
            #                                 n_ctx=30000,
            #                                 flash_attn=True,
            #                                 type_k=8,
            #                                 type_v=8,
            #                                 verbose=False
            #                             )

        return self._model
        
    def release_model(self):
        if self._model is not None:
            del self._model
            self._model = None
    
    def get_token_count(self, text):
        model = self.get_model()
        text_bytes = text.encode('utf-8')
        return len(model.tokenize(text_bytes))
    
    def get_token_sets(self, tags_actual):
            model = self.get_model()
            token_sets = []

            
            for tag in tags_actual:
                # Tokenize the tag
                tokenized_tag = model.tokenize(tag.encode('utf-8'))
                
                # Now process the tokenized tag as a sequence of tokens
                detokenized_tokens = [model.detokenize([token]).decode('utf-8') for token in tokenized_tag]
                
                # Append the list of tokens to the token_sets list
                token_sets.append(detokenized_tokens)
            
            return token_sets

    def construct_grammar_from_token_sets(self, token_sets):
        # Create grammar rule components for each tokenized tag
        tag_grammar_parts = []
        for token_set in token_sets:
            # Represent each token in the set, ensuring we use explicit token breaks
            tag_rule = ' '.join([f'"{token}"' for token in token_set])
            tag_grammar_parts.append(f"({tag_rule})")
        
        
        # Join the tag rules with | for alternation
        grammar_text = f"""
        root ::= tags
        tags ::= tag ("," tag)*
        tag ::= {' | '.join(tag_grammar_parts)}
        """
        
        return grammar_text


    def return_relevant_tags(self, text, tags_actual):
        model = self.get_model()
        
        # Ensure tags_actual is a list of token sets and include a "nothing" tag option
        token_sets = self.get_token_sets(tags_actual)  # Split tags into tokens
        token_sets.append(["nothing"])  # Add the "nothing" tag

        
        # Generate the grammar from token sets
        grammar_text = self.construct_grammar_from_token_sets(token_sets)

        with contextlib.redirect_stdout(open(os.devnull, 'w')), contextlib.redirect_stderr(open(os.devnull, 'w')):
            token_grammar = llama_cpp.LlamaGrammar.from_string(grammar_text)

        prompt = '''Given the following text and a list of possibly relevant valid tags in our database,
        return only the tag or tags that are relevant to the prompt being asked and may point towards documents in the database 
        that would help answer the prompt.  You are not trying to make a judgement call or answer the question. You should return a 
        comma delimited list. The first tag you return will be the Primary Tag, which documents are required to have to 
        be selected as context. It should be the most descriptive and relevant tag that is present in the list. Then, any subsequent 
        tags are Associated Tags, which may help refine the search if a document has more of those tags present. Once again, documents 
        have to have the primary tag, and they may have some number of secondary tags. Make sure the first document in your list is most 
        relevant.
        If none are applicable, return 'nothing'.\nText:\n'''

        constructed_prompt = prompt + text + "\nHere are the possibly relevant tags:\n" + ', '.join(tags_actual) + "\nThese are the actually relevant tags: \n"

        output = model(constructed_prompt, grammar=token_grammar)
        
        output_str = output["choices"][0]["text"]

        valid_tags = output_str.split(",")

        tags_trimmed = []

        for tag in valid_tags:
            if tag == "nothing":
                return []
            if tag in tags_trimmed or tag == "":
                continue
            tags_trimmed.append(tag)
        
        return tags_trimmed
    
    def generate_tags(self, text):
        model = self.get_model()
        prompt = '''Given the following text, generate a tag or a list of tags that describe subjects or the contents of the text. These 
        tags will be metadata associated with the text within a database and should fully describe subjects and concepts present in the text.
        The list should be comma-delimited.\nText\n'''

        constructed_prompt = prompt + text + "\nrelevant tags describing the contents, subjects, and concepts in the text:\n"

        output = model(constructed_prompt, grammar=self.generic_tag_grammar)

        output_str = output["choices"][0]["text"]

        text_tags = output_str.split(",")

        return text_tags
    
    def generate_roadmap(self, text):
        model = self.get_model()
        system_prompt = '''You are a knowledge base system orchestrator module. You are provided a prompt or query, you do not answer the prompt. 
        You are responsible for creating a functional set of steps to retrieve information from the knowledge base to best answer the prompt.
        You respond in json format with the steps to retrieve the information and explain what you are doing. You can query the database for 
        information. Say if the user asks for a comparison between two different concepts, you should make two individual calls to the database.
        Database queries are made via single tag or tag lists that will select matching documents about a subject or concept from the database. Tags are 
        lowercase alphanumeric strings with underscores. This is because sometimes it may take information from more than one specific subject or subjects to answer a problem or synthesize a response.
        Example of a single query plan: What is the population of Bangladesh?
        Response should be: { "steps": [ { "query": "population,bangladesh,bangladesh_population", "explanation": "Gathering information on the population of Bangladesh" } ] }
        Example of a multi query plan: What's the difference between the prime number theorem in and euler's theorem with coprimes?
        Response should be: { "steps": [ { "query": "prime_number_theorem", "explanation": "Gathering information on the prime number theorem" } , { "query": "euler_theorem,coprimes", "explanation": "Gathering information on euler's theorem" } ] }
        Do not mention the example concepts or tags in your response. Generate an answer specifically for your provided prompt as follows.'''
        roadmap_response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ],
            response_format={"type": "json_object", "schema": roadmap_schema}
        )

        roadmap_json = json.loads(roadmap_response["choices"][0]["message"]["content"]
                          .replace('\n', '\\n')
                          .replace('\r', '\\r')
                          .replace('\t', '\\t')
                          .encode('utf-8', 'ignore').decode('utf-8'))
        
        steps = roadmap_json["steps"]

        roadmap = [[step["query"].split(","), step["explanation"]] for step in steps]
        return roadmap
    
    def generate_response_with_context(self, conversation_history, context):
        model=self.get_model()

        context_str = "\n".join(context)
        combined_text = "\nRetrieved context:\n" + context_str

        conversation_history.append({"role": "system", "content": combined_text})

        response = model.create_chat_completion(
            messages=conversation_history,
        )

        return response["choices"][0]["message"]["content"]
    

    def finished_with_subdocs(self, messages, subject_list):
        model=self.get_model()

        system_prompt_finished = '''Have the generated sub-documents covered all the subjects or concepts in the document? There may be subjects in this list that are redundant or 
        unneccesary. If the sub-documents created so far have covered all subjects listed in the following list, return True. Else, return False. If you believe the entirety of the 
        information present in the source text hasn't been captured in the sub-docs yet, more sub-docs will be created.'''

        message_to_send = messages + [{"role": "system", "content": system_prompt_finished}]
        response = model.create_chat_completion(
            messages=message_to_send,
            response_format={"type": "json_object", "schema": {"type": "boolean"}}
        )

        return json.loads(response["choices"][0]["message"]["content"])

    
    
    def break_up_and_summarize_text(self, text):
        model = self.get_model()

        print("chunking text")
        
        system_prompt_subjects = '''You are provided with a document. Your task is to identify the major one or more subjects or concepts present in the document.
        List each subject or concept found in the document as a JSON array. Do not explain them. The subjects should be concise and accurately describe topics found in the text.
        Subjects should be as if you had to chunk up the given document into discrete chapters or topics. These subjects will be used to subdivide the text into documents to be 
        placed into a database, so smart chunking is crucial. For example, if a document is a list of 100 rapid-fire topics, one subject should probably be what describes the list as 
        a whole instead of one subject for every entry in the list. Not every single word or term needs to be its own subject.
        Avoid redundancy in similar subjects. If a subject is a subset of another subject, only list the broader subject. Like if 
        the document mentions serverless functions and then goes on to explain AWS Lambda, you would only list serverless functions as a subject, as that topic covers Lambda.
        You're not trying to reach a word count, think more in broad strokes, don't add every single detail or vocab word as a subject.'''

        # First, identify the subjects in the text
        subject_response = model.create_chat_completion(
            messages=[
                {"role": "user", "content": text},
                {"role": "system", "content": system_prompt_subjects}
            ],
            response_format={"type": "json_object", "schema": subject_list_schema}
        )

        print(subject_response["choices"][0]["message"]["content"])

        subject_list = json.loads(subject_response["choices"][0]["message"]["content"]
                            .replace('\n', '\\n')
                            .replace('\r', '\\r')
                            .replace('\t', '\\t')
                            .encode('utf-8', 'ignore').decode('utf-8'))["subjects"]



        subdocs = []

        messages = [
                {"role": "user", "content": text}
        ]

        for subject_item in subject_list:
            subject = subject_item["subject"]
            
            system_prompt_subdoc = f'''You are tasked with creating a sub-document for the subject: "{subject}". The sub-doc should mostly quote the original text,
            but you may paraphrase if necessary to abridge or clarify. It should explain the named subject in its entirety. Make sure not to add any external information that isn't found in the original document. 
            Include only the text relevant to the subject. After the sub-document text, list the tags that describe the subject or concept found in the sub-document.
            List only the tags that describe the contents of this sub-document. There may be one or two tags, or very many tags depending on the information conatined in the text
            of the sub-document created. Tags will be used as meta-data for each sub document in a database such that if someone wanted the information in the document, it could be 
            looked up by the tags, so design your tags for that use case. Tags are lowercase alphanumeric strings with underscores. Tags are single words or phrases. If there is a multi-word tag, 
            use underscores "_" as spaces. Avoid tags that are not relevant to the subject but found elsewhere in the text, unless this subject is a subset of a larger subject also defined elsewhere.'''

            # Build the messages, including the history
            messages.append({"role": "system", "content": system_prompt_subdoc})

            # Generate sub-document for this subject
            subdoc_response = model.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object", "schema": subdoc_schema}
            )
            
            print(subdoc_response["choices"][0]["message"]["content"])

            subdoc_data = json.loads(subdoc_response["choices"][0]["message"]["content"]
                            .replace('\n', '\\n')
                            .replace('\r', '\\r')
                            .replace('\t', '\\t')
                            .encode('utf-8', 'ignore').decode('utf-8'))
                          
            tags_possible = subdoc_data["tags"]
            tags_trimmed = []

            for tag in tags_possible:
                if tag in tags_trimmed or tag == "":
                    continue
                tags_trimmed.append(tag)
            

            subdocs.append({    
                "subdoc_text": subdoc_data["subdoc_text"],
                "tags": tags_trimmed
            })

            print(tags_trimmed)

            # Add this response to the message history for context in the next loop
            messages.append({
                "role": "assistant",
                "content": subdoc_response["choices"][0]["message"]["content"]
            })

            if (self.finished_with_subdocs(messages, subject_list)):
                print("LLM says all subjects already covered. Returning early.")
                break


        return subdocs

    def generate_response(self, conversation_history):
        model = self.get_model()

        #TODO: Figure out how to remove entries from conversation history such that we stay within a token limit
        # tokens = model.tokenize(conversation_history)
        # if len(tokens) > 28000:
        #     truncated_tokens = tokens[-28000:]
        #     conversation_history = self._model.detokenize(truncated_tokens).decode('utf-8')

        no_context_prompt = '''Based on the conversation history, you have elected that the user query can be answered without additional context from your database. Respond to the user.'''

        response = model.create_chat_completion(messages=conversation_history+[{"role": "system", "content": no_context_prompt}])

        return response["choices"][0]["message"]["content"]

    def decide_to_respond_or_use_tool(self, conversation_history):
        model = self.get_model()

        # tokens = model.tokenize(conversation_history)
        # if len(tokens) > 28000:
        #     truncated_tokens = tokens[-28000:]
        #     conversation_history = self._model.detokenize(truncated_tokens).decode('utf-8')

        system_prompt_choice = '''Decide if the current conversation history has the specific factual answer to the question being posed in the most recent user 
        message. If it does contain the information, say yes. If not, say no. If you are unsure at all, say no.You are not to base this decision on existing general knowledge.
        You only know information present in the context or database. Your output determines if a database lookup will be performed. If it does 
        not appear necessary to perform the database lookup, say yes. If it does appear necessary, say no.'''

        choice_response = model.create_chat_completion(
            messages=
                conversation_history+[{"role": "system", "content": system_prompt_choice}],
            response_format={"type": "json_object", "schema": choice_schema}
        )

        choice = json.loads(choice_response["choices"][0]["message"]["content"]
                          .replace('\n', '\\n')
                          .replace('\r', '\\r')
                          .replace('\t', '\\t')
                          .encode('utf-8', 'ignore').decode('utf-8'))["choice"]
        return choice
