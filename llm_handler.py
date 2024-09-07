import llama_cpp
from huggingface_hub import hf_hub_download
import os
import json

MODELS_DIR = "./models/llm"
model_name = "lmstudio"
model_file = os.path.join(MODELS_DIR, "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf")

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
tag ::= (alphanumeric | "_")+
tags ::= tag ("," tag)*
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

class LLMHandler:

    #singleton model
    _model = None
                              
    def __init__(self):
        if not os.path.exists(model_file):
            print("Downloading model...")
            hf_hub_download(repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF", 
                            filename="Meta-Llama-3.1-8B-Instruct-Q8_0.gguf", 
                            cache_dir=MODELS_DIR)
        else:
            print("Model already downloaded.")

        generic_tag_grammar = llama_cpp.LlamaGrammar.from_string(generic_tag_grammar_text)

    def get_model(self):
        if self._model is None:
            self._model = llama_cpp.Llama(model_file,
                                            n_gpu_layers=-1,
                                            n_ctx=30000,
                                            flash_attn=True,
                                            type_k=8,
                                            type_v=8,
                                        )
        return self._model
        
    def release_model(self):
        del self._model
        self._model = None
    
    def get_token_count(self, text):
        model = self.get_model()
        text_bytes = text.encode('utf-8')
        return len(model.tokenize(text_bytes))

    def construct_grammar_from_token_sets(token_sets):
    # Create grammar rule components for each tokenized tag
        tag_grammar_parts = []
        for token_set in token_sets:
            # Join tokens as a sequence for each tag (e.g., project_implementation_steps)
            tag_rule = ' '.join([f'"{token}"' for token in token_set])
            tag_grammar_parts.append(f"({tag_rule})")
        
        # Join the tag rules with commas or allow them to appear individually
        grammar_text = f"""
        tag ::= { ' | '.join(tag_grammar_parts) }
        tags ::= tag ("," tag)*
        """
        return grammar_text


    def return_relevant_tags(self, text, tags_actual):
        model = self.get_model()
        #add a nothing tag to tags_actual for a no-match
        tags_nothing = tags_actual + "nothing"
        grammar_text = self.construct_grammar_from_token_sets(tags_nothing)

        token_grammar = llama_cpp.LlamaGrammar.from_string(grammar_text)

        prompt = '''Given the following text and a list of possibly relevant valid tags in our database,
        return only the tag or tags that are relevant to the text and would point towards documents in the database 
        that would help answer the prompt. You should return a commma delimited list. If none are applicable, return 'nothing'.\nText:\n'''

        constructed_prompt = prompt + text + "\nHere are the possibly relevant tags:\n" + tags_actual + "\n These are the actually relevant tags: \n"

        output_str = model(constructed_prompt, grammar=token_grammar)

        valid_tags = output_str.split(",")
    
        return valid_tags
    
    def generate_tags(self, text):
        model = self.get_model()
        prompt = '''Given the following text, generate a tag or a list of tags that describe subjects or the contents of the text. These 
        tags will be metadata associated with the text within a database and should fully describe subjects and concepts present in the text.
        The list should be comma-delimited.\nText\n'''

        constructed_prompt = prompt + text + "\nrelevant tags describing the contents, subjects, and concepts in the text:\n"

        output_str = model(constructed_prompt, grammar=self.generic_tag_grammar)

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

        roadmap_json = json.loads(roadmap_response["choices"][0]["message"]["content"])
        
        steps = roadmap_json["steps"]

        roadmap = [[step["query"].split(","), step["explanation"]] for step in steps]
        return roadmap
    
    def generate_response_with_context(self, text):
        model=self.get_model()
        system_prompt = '''You are the culmination of a knowledge base system. Given the retrieved documents from the database focused around a certain domain
        and a user prompt or query, respond to the user to the best of your ability. Use the retrieved data in your answer. If the answer does not lie within
        the provided data, say as much. Be clear, effective, and succint in your responses while also fully explaining requested concepsts.'''

        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        return response["choices"][0]["message"]["content"]
    






        
