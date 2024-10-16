import llama_cpp
from pydantic import BaseModel, Field, StringConstraints
import instructor
from typing import List, Dict, Annotated
from huggingface_hub import hf_hub_download
import os, sys
import contextlib
import json
import shutil

MODELS_DIR = "models\\llm"

model_file_name = os.path.join(MODELS_DIR, "model_file_name.json")

subdoc_char_limit = 100

"""
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
"""

generic_tag_grammar_text = """
root ::= tags
tags ::= tag ("," tag)*
tag ::= alphanumeric | "_"
alphanumeric ::= [a-zA-Z0-9]+
"""


class Step(BaseModel):
    query: List[str] = Field(
        ...,
        description="A list of alphanumeric lowercase strings with underscores, representing database queries.",
    )
    explanation: str = Field(
        ..., description="A string explaining the purpose of the query."
    )


class Roadmap(BaseModel):
    steps: List[Step] = Field(
        ..., description="A list of steps containing queries and their explanations."
    )


class Choice(BaseModel):
    choice: str = Field(..., enum=["no", "yes"])

    class Config:
        extra = "forbid"


class Subject(BaseModel):
    subject: str = Field(..., description="A subject or concept found in the document.")


class SubjectList(BaseModel):
    subjects: List[Subject] = Field(
        ..., description="A list of subjects or concepts found in the document."
    )



TagType = Annotated[
    str,
    StringConstraints(pattern=r"^[a-z0-9_]+$")
]

class Subdoc(BaseModel):
    subdoc_text: str = Field(
        ...,
        description="The subdoc text, mostly quoting from the source with minimal paraphrasing.",
    )
    tags: List[TagType] = Field(
        ...,
        description="A list of tags describing the subject or concept found in the subdoc.",
    )


class LLMHandler:

    # singleton model
    _model = None
    generic_tag_grammar = None

    def __init__(self):
        if not os.path.exists(model_file_name):
            print("Downloading model...")
            llama_large_location = hf_hub_download(
                repo_id="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
                filename="Meta-Llama-3.1-8B-Instruct-Q3_K_L.gguf",
                cache_dir=MODELS_DIR,
            )
            # store model location in a json file
            model_json = {"model_file": llama_large_location}
            with open(model_file_name, "w") as f:
                json.dump(model_json, f)
        else:
            print("Model already downloaded.")
        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            self.generic_tag_grammar = llama_cpp.LlamaGrammar.from_string(
                generic_tag_grammar_text
            )

    # TODO: add big vs small model selection and figure out where smaller models could be used in functionality.
    def get_model(self, size="big"):
        if self._model is None:
            with open(model_file_name, "r") as f:
                model_json = json.load(f)

            if not os.path.exists(model_json["model_file"]):
                raise FileNotFoundError(
                    f"Model file {model_json['model_file']} not found on disk."
                )

            self._model = llama_cpp.Llama(
                model_json["model_file"],
                n_gpu_layers=-1,
                n_ctx=30000,
                flash_attn=True,
                type_k=8,
                type_v=8,
                verbose=False,
                repeat_penalty=1.1,
            )

            self._create = instructor.patch(
                create=self._model.create_chat_completion_openai_v1,
                mode=instructor.Mode.JSON_SCHEMA,
            )

        return self._model, self._create

    def release_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            del self._create
            self._create = None

    def get_token_count(self, text):
        model,_ = self.get_model()
        text_bytes = text.encode("utf-8")
        return len(model.tokenize(text_bytes))

    def get_token_sets(self, tags_actual):
        model,_ = self.get_model()
        token_sets = []

        for tag in tags_actual:
            # Tokenize the tag
            tokenized_tag = model.tokenize(tag.encode("utf-8"))

            # Now process the tokenized tag as a sequence of tokens
            detokenized_tokens = [
                model.detokenize([token]).decode("utf-8") for token in tokenized_tag
            ]

            # Append the list of tokens to the token_sets list
            token_sets.append(detokenized_tokens)

        return token_sets

    def construct_grammar_from_token_sets(self, token_sets):
        # Create grammar rule components for each tokenized tag
        tag_grammar_parts = []
        for token_set in token_sets:
            # Represent each token in the set, ensuring we use explicit token breaks
            tag_rule = " ".join([f'"{token}"' for token in token_set])
            tag_grammar_parts.append(f"({tag_rule})")

        # Join the tag rules with | for alternation
        grammar_text = f"""
        root ::= tags
        tags ::= tag ("," tag)*
        tag ::= {' | '.join(tag_grammar_parts)}
        """

        return grammar_text

    def return_relevant_tags(self, text, tags_actual):
        model,_ = self.get_model()

        # Ensure tags_actual is a list of token sets and include a "nothing" tag option
        token_sets = self.get_token_sets(tags_actual)  # Split tags into tokens
        token_sets.append(["nothing"])  # Add the "nothing" tag

        # Generate the grammar from token sets
        grammar_text = self.construct_grammar_from_token_sets(token_sets)

        with contextlib.redirect_stdout(
            open(os.devnull, "w")
        ), contextlib.redirect_stderr(open(os.devnull, "w")):
            token_grammar = llama_cpp.LlamaGrammar.from_string(grammar_text)

        prompt = """Given the following text and a list of possibly relevant valid tags in our database,
        return only the tag or tags that are relevant to the prompt being asked and may point towards documents in the database 
        that would help answer the prompt.  You are not trying to make a judgement call or answer the question. You should return a 
        comma delimited list. The first tag you return will be the Primary Tag, which documents are required to have to 
        be selected as context. It should be the most descriptive and relevant tag that is present in the list. Then, any subsequent 
        tags are Associated Tags, which may help refine the search if a document has more of those tags present. Once again, documents 
        have to have the primary tag, and they may have some number of secondary tags. Make sure the first document in your list is most 
        relevant.
        If none are applicable, return 'nothing'.\nText:\n"""

        constructed_prompt = (
            prompt
            + text
            + "\nHere are the possibly relevant tags:\n"
            + ", ".join(tags_actual)
            + "\nThese are the actually relevant tags: \n"
        )

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
        model,_ = self.get_model()
        prompt = """Given the following text, generate a tag or a list of tags that describe subjects or the contents of the text. These 
        tags will be metadata associated with the text within a database and should fully describe subjects and concepts present in the text.
        The list should be comma-delimited.\nText\n"""

        constructed_prompt = (
            prompt
            + text
            + "\nrelevant tags describing the contents, subjects, and concepts in the text:\n"
        )

        output = model(constructed_prompt, grammar=self.generic_tag_grammar)

        output_str = output["choices"][0]["text"]

        text_tags = output_str.split(",")

        return text_tags
    
    def get_num_lines(self, text):
        """Calculate the number of terminal lines the text occupies."""
        terminal_width = shutil.get_terminal_size().columns
        lines = text.split('\n')
        num_lines = 0
        for line in lines:
            # Calculate the number of terminal lines for each line
            line_length = len(line)
            num_terminal_lines = max(1, (line_length + terminal_width - 1) // terminal_width)
            num_lines += num_terminal_lines
        return num_lines

    def clear_lines(self, num_lines):
        """Move the cursor up num_lines and clear those lines."""
        for _ in range(num_lines):
            sys.stdout.write('\033[1A')  # Move cursor up one line
            sys.stdout.write('\033[2K')  # Clear entire line

    def get_structured_output(
        self,
        messages: List[Dict[str, str]],
        response_model: BaseModel,
        verbose: bool = False,
    ):
        """
        Streams the model output, updating the terminal line with partial results,
        and returns the accumulated data as a dictionary.

        Args:
            messages (List[Dict[str, str]]): The messages to send to the model.
            response_model (BaseModel): The Pydantic model class defining the expected output.
            verbose (bool): If True, updates the terminal with streaming output.

        Returns:
            Dict[str, Any]: The accumulated data as a dictionary.
        """

        _, create = self.get_model()

        extraction_stream = create(
            response_model=instructor.Partial[response_model],
            messages=messages,
            stream=True,
        )
        
        accumulated_data = {}
        previous_num_lines = 0
        
        for extraction in extraction_stream:
            partial_data = extraction.model_dump()
            accumulated_data.update(partial_data)
            
            if verbose:
                # Convert accumulated_data to a pretty JSON string
                output = json.dumps(accumulated_data, indent=2)
                
                # Calculate the number of lines the output occupies
                num_lines = self.get_num_lines(output)
                
                # Clear previous output
                if previous_num_lines > 0:
                    self.clear_lines(previous_num_lines)
                
                # Print the new output
                sys.stdout.write(output + '\n')
                sys.stdout.flush()
                
                # Update previous_num_lines for the next iteration
                previous_num_lines = num_lines
        
        if verbose:
            sys.stdout.write('\n')
        
        return accumulated_data

    def generate_roadmap(self, text):
        system_prompt = """You are a knowledge base system orchestrator module. You are provided a prompt or query, you do not answer the prompt. 
        You are responsible for creating a functional set of steps to retrieve information from the knowledge base to best answer the prompt.
        You respond in json format with the steps to retrieve the information and explain what you are doing. You can query the database for 
        information. Say if the user asks for a comparison between two different concepts, you should make two individual calls to the database.
        Database queries are made via single tag or tag lists that will select matching documents about a subject or concept from the database. Tags are 
        lowercase alphanumeric strings with underscores. This is because sometimes it may take information from more than one specific subject or subjects to answer a problem or synthesize a response.
        Example of a single query plan: What is the population of Bangladesh?
        Response should be: { "steps": [ { "query": "population,bangladesh,bangladesh_population", "explanation": "Gathering information on the population of Bangladesh" } ] }
        Example of a multi query plan: What's the difference between the prime number theorem in and euler's theorem with coprimes?
        Response should be: { "steps": [ { "query": "prime_number_theorem", "explanation": "Gathering information on the prime number theorem" } , { "query": "euler_theorem,coprimes", "explanation": "Gathering information on euler's theorem" } ] }
        Do not mention the example concepts or tags in your response. Generate an answer specifically for your provided prompt as follows."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]

        roadmap = self.get_structured_output(
            messages=messages, response_model=Roadmap, verbose=True
        )

        steps = roadmap["steps"]

        roadmap = [[step["query"], step["explanation"]] for step in steps]
        return roadmap

    def generate_response_with_context(self, conversation_history, context):
        model, _ = self.get_model()

        context_str = "\n".join(context)
        combined_text = "\nRetrieved context:\n" + context_str

        conversation_history.append({"role": "system", "content": combined_text})

        response = model.create_chat_completion(
            messages=conversation_history,
        )

        return response["choices"][0]["message"]["content"]

    def finished_with_subdocs(self, messages, subject_list):

        system_prompt_finished = """Have the generated sub-documents covered all the subjects or concepts in the document? There may be subjects in this list that are redundant or 
        unneccesary. If the sub-documents created so far have covered all subjects listed in the following list, return True. Else, return False. If you believe the entirety of the 
        information present in the source text hasn't been captured in the sub-docs yet, more sub-docs will be created."""

        message_to_send = messages + [
            {"role": "system", "content": system_prompt_finished}
        ]

        choice = self.get_structured_output(
            messages=message_to_send, response_model=Choice, verbose=True
        )

        return choice["choice"] == "yes"

    def break_up_and_summarize_text(self, text):

        print("chunking text")

        system_prompt_subjects = """You are provided with a document. Your task is to identify the major one or more subjects or concepts present in the document.
        List each subject or concept found in the document as a JSON array. Do not explain them. The subjects should be concise and accurately describe topics found in the text.
        Subjects should be as if you had to chunk up the given document into discrete chapters or topics. These subjects will be used to subdivide the text into documents to be 
        placed into a database, so smart chunking is crucial. For example, if a document is a list of 100 rapid-fire topics, one subject should probably be what describes the list as 
        a whole instead of one subject for every entry in the list. Not every single word or term needs to be its own subject.
        Avoid redundancy in similar subjects. If a subject is a subset of another subject, only list the broader subject. Like if 
        the document mentions serverless functions and then goes on to explain AWS Lambda, you would only list serverless functions as a subject, as that topic covers Lambda.
        You're not trying to reach a word count, think more in broad strokes, don't add every single detail or vocab word as a subject."""

        # First, identify the subjects in the text

        subject_response = self.get_structured_output(
            messages=[
                {"role": "user", "content": text},
                {"role": "system", "content": system_prompt_subjects}
            ],
            response_model=SubjectList, verbose=True
        )

        print("subjects:")
        print(subject_response)

        subject_list = subject_response["subjects"]

        subdocs = []

        messages = [{"role": "user", "content": text}]

        for subject_item in subject_list:
            subject = subject_item["subject"]

            system_prompt_subdoc = f"""You are tasked with creating a sub-document for the subject: "{subject}". The sub-doc should mostly quote the original text,
            but you may paraphrase if necessary to abridge or clarify. It should explain the named subject in its entirety. Make sure not to add any external information that isn't found in the original document. 
            Include only the text relevant to the subject. After the sub-document text, list the tags that describe the subject or concept found in the sub-document.
            List only the tags that describe the contents of this sub-document. There may be one or two tags, or very many tags depending on the information conatined in the text
            of the sub-document created. Tags will be used as meta-data for each sub document in a database such that if someone wanted the information in the document, it could be 
            looked up by the tags, so design your tags for that use case. Tags are lowercase alphanumeric strings with underscores. Tags are single words or phrases. If there is a multi-word tag, 
            use underscores "_" as spaces. Avoid tags that are not relevant to the subject but found elsewhere in the text, unless this subject is a subset of a larger subject also defined elsewhere."""

            # Build the messages, including the history
            messages.append({"role": "system", "content": system_prompt_subdoc})


            # Generate sub-document for this subject

            subdoc_response = self.get_structured_output(
                messages=messages, response_model=Subdoc, verbose=True
            )

            tags_possible = subdoc_response["tags"]
            tags_trimmed = []

            for tag in tags_possible:
                if tag in tags_trimmed or tag == "":
                    continue
                tags_trimmed.append(tag)

            subdocs.append(
                {"subdoc_text": subdoc_response["subdoc_text"], "tags": tags_trimmed}
            )

            print("subdoc:")
            print(subdocs[-1])

            print(tags_trimmed)

            # Add this response to the message history for context in the next loop
            messages.append(
                {
                    "role": "assistant",
                    "content": str(subdoc_response),
                }
            )

            if self.finished_with_subdocs(messages, subject_list):
                print("LLM says all subjects already covered. Returning early.")
                break

        return subdocs

    def generate_response(self, conversation_history):
        model = self.get_model()

        # TODO: Figure out how to remove entries from conversation history such that we stay within a token limit
        # tokens = model.tokenize(conversation_history)
        # if len(tokens) > 28000:
        #     truncated_tokens = tokens[-28000:]
        #     conversation_history = self._model.detokenize(truncated_tokens).decode('utf-8')

        no_context_prompt = """Based on the conversation history, you have elected that the user query can be answered without additional context from your database. Respond to the user."""

        response = model.create_chat_completion(
            messages=conversation_history
            + [{"role": "system", "content": no_context_prompt}]
        )

        return response["choices"][0]["message"]["content"]

    def decide_to_respond_or_use_tool(self, conversation_history):

        system_prompt_choice = """Decide if the current conversation history has the specific factual answer to the question being posed in the most recent user 
        message. If it does contain the information, say yes. If not, say no. If you are unsure at all, say no.You are not to base this decision on existing general knowledge.
        You only know information present in the context or database. Your output determines if a database lookup will be performed. If it does 
        not appear necessary to perform the database lookup, say yes. If it does appear necessary, say no."""

        choice_response = self.get_structured_output(
            messages=conversation_history
            + [{"role": "system", "content": system_prompt_choice}],
            response_model=Choice, verbose=True
        )

        return choice_response["choice"] == "yes"
