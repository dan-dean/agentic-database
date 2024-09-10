from llm_handler import *

# test loading in model

llm_handler = LLMHandler()

cloud_compute_text = """
Cloud computing is a critical component in modern technology infrastructures, enabling businesses to build, deploy, and scale applications efficiently. Three major concepts stand out in cloud computing: Amazon Web Services (AWS), scalable code development, and serverless functions such as AWS Lambda.

Amazon Web Services (AWS) is a cloud platform offered by Amazon, providing a vast range of services including computing power, storage options, and machine learning. AWS is one of the most popular cloud platforms and offers highly flexible, scalable, and secure services. It is often the first choice for companies looking to migrate their applications to the cloud. AWS offers services like EC2 for compute capacity, S3 for storage, and various database services, which allow businesses to scale up or down based on their needs.

Building scalable code is a key requirement in cloud-based systems. Scalability ensures that applications can handle increasing loads by efficiently managing resources. Developers focus on optimizing their code for cloud environments, considering factors such as server allocation, dynamic load balancing, and fault tolerance. Techniques like microservices architecture, horizontal scaling, and distributed databases help create robust, high-performance systems that adapt to user demand without crashing or requiring major code rewrites.

AWS Lambda is an example of a serverless computing model, allowing developers to run code without provisioning or managing servers. Lambda functions are executed in response to events, such as HTTP requests or database updates, and automatically scale based on the workload. This serverless approach reduces operational complexity and cost, making it easier for developers to focus on writing code instead of managing infrastructure. AWS Lambda is especially useful for building event-driven architectures, simplifying the process of developing modern cloud-native applications.
"""

print("token count test")

print(llm_handler.get_token_count(cloud_compute_text))  

print("the big one, testing summary and chunking on small scale")

sub_docs_cloud = llm_handler.break_up_and_summarize_text(cloud_compute_text)

cloud_tags = []

for sub_doc in sub_docs_cloud:
    print(sub_doc["subdoc_text"])
    print("tags:")
    for tag in sub_doc["tags"]:
        print(tag)
        cloud_tags.append(tag)

print("testing getting prospective tags from prompt")

prompt_cloud = "What are the services that are offered by AWS?"

prospective_tags_cloud = llm_handler.generate_tags(prompt_cloud)

print(prospective_tags_cloud)

print("testing selecting relevant tags from real tag candidates")

selected_tags_cloud = llm_handler.return_relevant_tags(prompt_cloud, cloud_tags)

print("Selected tags for prompt: ", prompt_cloud)
print(selected_tags_cloud)

print("testing creating roadmap simple")

roadmap = llm_handler.generate_roadmap(prompt_cloud)

print(roadmap)

print("testing creating roadmap with multiple steps")

prompt_2_step = "How does AWS compare to Azure in terms of services offered?"

roadmap_2_step = llm_handler.generate_roadmap(prompt_2_step)

print(roadmap_2_step)

llm_handler.release_model()


