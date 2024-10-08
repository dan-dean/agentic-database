# agentic-database

A novel implementation of RAG and a data processor utilizing LLM agents at every step.

[See Installation instructions](#installation)

## Introduction

Having built naive systems with vector embedding RAG, I found it dissapointing that the vector database and indexing was a black box, in which the accuracy of the recall system, albeit based on semantic context, was obfuscated and unparseable. Vector databases rely on the query vector of the prompt being within spitting euclidean distance of documents in the database that would answer the question, and this would not always be true. Also, naive document entry systems would chunk up your data without regard for the semantic information contained within, resulting in bad entries with half-formed ideas. The name of the game is to reduce the length of context strings that must be provided to the LLM at prompt time, to reduce costs and time to first token. Hence, smaller chunked data being favorable.

Vector database and document chunking are meant to be quick and dirty approaches to RAG. As with any dataset, the better data you can put in, the better your system functions, and the more useful it becomes. If you want more correctly defined documents for your database, this requires manual intervention and is costly time-wise.

I aim to address a number of these issues with this project, starting with document creation. Indexing documents initially is a time intensive operation, but once they are present in the database, queries happen quickly. An LLM agent pulls out all the major subjects and topics found in a document and forms a list. Then, the agent creates sub-documents from the source document that are largely quotations and mild summarizations of segments. These sub-documents are also tagged with the subjects discussed within them and stored within a SQL database with relations between sub-documents, source documents, and tags. This ensures that the content in each document is complete in its scope.

The metadata tags are stored in a FAISS vector index. In full-fledged vector databases for RAG, the semantic meaning and intent of documents is muddied across the large amount of text contained in each document. Indexing just terse metadata tags results in a greater likelihood of positive hits for a query. The euclidean distance between aws and amazon_web_services in the vector space is a lot more reliably short than "What is amazon web services?" and a 600 page document that may partially describe the answer.

On prompt time, your prompt is handed off to a roadmapping agent that determines what queries are necessary, as many questions may require information from different domains wihtin your database.

For example:
"What is AWS?" is a one-step question, where it needs to query for information about AWS.
"What is the difference between Azure and AWS?" is a two-step question, where it needs to make two calls to the database as the differences and feature-sets of Azure and AWS may not be contained in the same sub-document.

For each search step in this roadmap, the agent creates possible tags to search for based on your prompt. A query is made to the vector database and neighboring real-tags that have matching documents in the SQL database are retrieved. The agent makes a judgement call on what tags are actually relevant, and the highest-matching documents are retrieved from the database.

These documents are appended to your prompt, and the LLM responds.

Because of the terse nature of tags, many (20+) neighboring tags in the vector database can be retrieved for every possible tag the agent asks for without significantly slowing the time to first token.

In my experimentation, this solution has proved more accurate and finely tuned than naive indexing and document chunking. The preservation of actual data and complete ideas and subjects that the agent-based document chunking system maintains is hard to understate the importance of. The classes in this project provide hooks to be able to parse the populated tables of documents and perform searches and retrievals. I believe the SQL tag approach has this as an edge over black-box vector embedding.

A basic CLI implementation has been created to allow for easy interaction with the modules, albeit not with all the database parsing mentioned above. This CLI interacts with the async_agentic_database module which provides a queue-based system for loading in documents and queries.

## What's next

1. The LLM agents in this system are not built with any existing libraries or frameworks, instead by constraining raw output to JSON schemas and grammars defined for the use-cases. Adding pydantic to increase the legibility of the LLM system is next.

2. The agentic document chunking system will be pulled out and further modularized to exist as a stand-alone package so that it is agnostic to the RAG system it is used with.

3. I want to then experiment with the colBERT reranking system instead, while still utilizing a more automated and smart document chunking system as from the library described. ColBERT reranking offers more granular vector embeddings that may alay the fears present in muddied semantic meaning from traditional vector databases that reduce their accuracy. While we then sacrifice parseability present in the tagged-up SQL database, if colBERT retrieval is consistently accurate, then the mission has been accomplished.

## Installation

Prerequisites

    pip install huggingface_hub
    pip install faiss-cpu
    pip install sentence_transformers
    pip install llama_cpp_python

About using GPU acceleration for llama_cpp:

From the [llama_cpp_python](https://github.com/abetlen/llama-cpp-python) repo:
To install with CUDA support, set the GGML_CUDA=on environment variable before installing:

    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

Pre-built Wheel (New)

It is also possible to install a pre-built wheel with CUDA support. As long as your system meets some requirements:

CUDA Version is 12.1, 12.2, 12.3, 12.4 or 12.5
Python Version is 3.10, 3.11 or 3.12
pip install llama-cpp-python \
 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/<cuda-version>
Where <cuda-version> is one of the following:

cu121: CUDA 12.1
cu122: CUDA 12.2
cu123: CUDA 12.3
cu124: CUDA 12.4
cu124: CUDA 12.5
For example, to install the CUDA 12.1 wheel:

    pip install llama-cpp-python \
      --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

clone the repo

Run cli_implementation.py

## Quickstart

    mk_db "example database"
    set_db 1
    add *path-to-text-file*
    add *path-to-text-file*
    ask "What is the difference between Azure and AWS?"
    thread #to start a chat thread where conversation history is maintained

Output of the ask command:

    > Gathering information on Azure's cloud computing capabilities
    found  7  documents with relevant tags
    reading document with tags:  ['cloud_services', 'azure', 'microsoft']
    Gathering information on AWS's cloud computing capabilities
    found  4  documents with relevant tags
    reading document with tags:  ['aws', 'amazon_web_services']
    Response: Based on the retrieved context, the main differences between Azure and AWS are:

    1. **Ownership**: Azure is a cloud platform offered by Microsoft, while AWS is a cloud platform offered by Amazon.
    2. **Services**: Both platforms offer a range of cloud services, including IaaS, PaaS, and SaaS. However, Azure and AWS have different service offerings, with Azure focusing on enterprise-grade services and AWS focusing on a broader range of services, including machine learning and serverless computing.
    3. **Scalability**: Both platforms offer scalable services, but AWS is known for its ability to scale up or down quickly in response to changing workloads.
    4. **Serverless Computing**: AWS Lambda is a serverless computing model that allows developers to run code without provisioning or managing servers. Azure also offers a serverless computing model, Azure Functions, but it is not as mature as AWS Lambda.
    5. **Cost**: The cost of using Azure and AWS can vary depending on the services used and the scale of the deployment. However, AWS is generally considered to be more cost-effective for large-scale deployments.
    6. **Security and compliance**: Azure has robust security and compliance features, including threat detection, encryption, and compliance with global standards such as GDPR and HIPAA.

    It's worth noting that both Azure and AWS have their own strengths and weaknesses, and the choice between them ultimately depends on the specific needs of the organization.

\*_Note that the documents in this shown test run were generated by GPT 4o and may not be accurate. They just had synthetic test data._

Use the **help** command to list the available commands in the CLI.
