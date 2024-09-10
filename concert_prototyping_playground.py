from llm_handler import *
from tag_database_handler import *
from doc_database_handler import *
from collections import Counter

cloud_compute_aws_text = '''Cloud computing is a critical component in modern technology infrastructures, enabling businesses to build, deploy, and scale applications efficiently. Three major concepts stand out in cloud computing: Amazon Web Services (AWS), scalable code development, and serverless functions such as AWS Lambda.

Amazon Web Services (AWS) is a cloud platform offered by Amazon, providing a vast range of services including computing power, storage options, and machine learning. AWS is one of the most popular cloud platforms and offers highly flexible, scalable, and secure services. It is often the first choice for companies looking to migrate their applications to the cloud. AWS offers services like EC2 for compute capacity, S3 for storage, and various database services, which allow businesses to scale up or down based on their needs.

Building scalable code is a key requirement in cloud-based systems. Scalability ensures that applications can handle increasing loads by efficiently managing resources. Developers focus on optimizing their code for cloud environments, considering factors such as server allocation, dynamic load balancing, and fault tolerance. Techniques like microservices architecture, horizontal scaling, and distributed databases help create robust, high-performance systems that adapt to user demand without crashing or requiring major code rewrites.

AWS Lambda is an example of a serverless computing model, allowing developers to run code without provisioning or managing servers. Lambda functions are executed in response to events, such as HTTP requests or database updates, and automatically scale based on the workload. This serverless approach reduces operational complexity and cost, making it easier for developers to focus on writing code instead of managing infrastructure. AWS Lambda is especially useful for building event-driven architectures, simplifying the process of developing modern cloud-native applications.
'''

microsoft_products_text = '''Microsoft Services Overview
============================

Office 365
----------

Office 365 is a cloud-based productivity suite offered by Microsoft that includes various applications designed to improve business and personal productivity. Office 365 includes well-known applications such as Word, Excel, PowerPoint, and Outlook, but also integrates cloud-based services such as OneDrive, Microsoft Teams, and SharePoint.

With Office 365, users can access and edit their documents from virtually anywhere with an internet connection. The service supports real-time collaboration, allowing multiple users to work on the same document simultaneously, which is especially useful in modern, distributed work environments. Security and compliance are key features, with Office 365 providing enterprise-grade security to protect data and meet regulatory standards.

OneDrive allows users to store and sync files across devices, ensuring that important documents are always backed up and accessible. Microsoft Teams, on the other hand, facilitates real-time communication and collaboration, with video conferencing, chat, and file sharing integrated into a single platform. SharePoint is used for content management and team collaboration, making it easier for businesses to manage documents and workflows.

Windows Operating Systems
--------------------------

Microsoft's Windows platform offers a variety of operating systems tailored to different user needs, including Windows Home, Pro, and Server editions.

Windows Home is designed for personal users, offering a familiar interface and essential features for everyday tasks such as web browsing, document creation, and media consumption. It comes pre-installed on most consumer PCs and is widely regarded for its ease of use, reliability, and compatibility with a vast array of hardware and software.

Windows Pro is geared towards professionals and small to medium-sized businesses. It includes all the features of Windows Home, along with advanced capabilities such as BitLocker encryption, remote desktop access, and domain join functionality. These features provide enhanced security and remote management, making Windows Pro ideal for IT professionals managing business networks.

Windows Server is an enterprise-grade operating system designed for managing large-scale networks and data centers. It supports virtualization, robust security policies, and scalable infrastructure management. Businesses use Windows Server for hosting applications, managing cloud services, and centralizing data storage. Key features include Active Directory, which manages user permissions and network resources, and Hyper-V, a hypervisor for creating and managing virtual machines. Windows Server is a critical component for enterprise IT infrastructures, offering tools to streamline operations and maintain high availability of services.

Azure Cloud Platform
--------------------

Azure is Microsoft’s cloud computing platform, providing a range of services for building, deploying, and managing applications across a global network of data centers. Like Amazon Web Services (AWS), Azure offers on-demand computing resources, storage, and networking capabilities, but it also integrates deeply with existing Microsoft services such as Windows, SQL Server, and Office 365.

Azure supports various deployment modecls, including Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS), giving businesses the flexibility to choose the right solution for their needs. IaaS enables users to run virtual machines and store data in the cloud, while PaaS simplifies app development by offering managed services for databases, application hosting, and monitoring. SaaS allows users to access cloud-based software like Office 365 without needing to manage the underlying infrastructure.

One of Azure’s standout features is its seamless integration with hybrid cloud environments, enabling businesses to run workloads both in their on-premise data centers and on Azure’s cloud. Azure Arc further extends this capability, allowing organizations to manage multi-cloud environments and on-premises resources through a unified interface.

Azure’s security and compliance features are robust, with built-in threat detection, encryption, and compliance with global standards such as GDPR and HIPAA. Azure also offers tools for DevOps, AI and machine learning, and big data processing, making it a versatile platform for a wide range of use cases.

Conclusion
----------

Microsoft’s suite of services—from Office 365 and Windows operating systems to the Azure cloud platform—offers powerful tools for personal use, businesses, and enterprise IT infrastructures. By integrating productivity tools with cloud-based services and advanced security, Microsoft ensures its products remain at the forefront of modern technology solutions.
'''

llm_handler = LLMHandler()

tag_handler = TagDatabaseHandler()

tag_handler.create_database("test_concert_tags")

doc_db_name = create_database("test_concert_docs")

print("starting tests")

try:

    sub_docs_cloud = llm_handler.break_up_and_summarize_text(cloud_compute_aws_text)

    all_tags = []

    for sub_doc in sub_docs_cloud:
        for tag in sub_doc["tags"]:
            all_tags.append(tag)

    sub_docs_microsoft = llm_handler.break_up_and_summarize_text(microsoft_products_text)

    for sub_doc in sub_docs_microsoft:
        for tag in sub_doc["tags"]:
            all_tags.append(tag)

    llm_handler.release_model()

    print("all tags: ", all_tags)

    tag_handler.add_entry_to_database("test_concert_tags", all_tags)

    print(sub_docs_cloud)
    print(sub_docs_microsoft)

    add_entry_to_database(doc_db_name, cloud_compute_aws_text, sub_docs_cloud)

    add_entry_to_database(doc_db_name, microsoft_products_text, sub_docs_microsoft)

    tag_handler.release_model()

    print("Database creation and population complete.")

    # prompt_2_step = "How does AWS compare to Azure in terms of services offered?"

    while True:

        prompt = input("Enter a prompt or exit: ")

        if prompt == "exit":
            break

        print("creating roadmap")

        roadmap_2_step = llm_handler.generate_roadmap(prompt)

        context = []

        for step in roadmap_2_step:
            print(step[1])
            print("using tags: ", step[0])

            #get real tags from prospective
            real_tags = tag_handler.get_nearest_neighbors("test_concert_tags", step[0], 10)

            print("real tags: ", real_tags)

            #get relevant tags from real

            real_tags_pool = []

            for tag_list in real_tags:
                for tag in tag_list:
                    real_tags_pool.append(tag)

            relevant_tags = llm_handler.return_relevant_tags(step[1], real_tags_pool)

            if len(relevant_tags) == 0:
                print("I'm sorry, I have no knowledge on that subject.")
                continue

            print("relevant tags: ", relevant_tags)

            #get doc uuids from relevant tags

            doc_uuids, doc_tags = get_document_uuid_tags_from_tag(doc_db_name, relevant_tags)

            print("found ", len(doc_uuids), " documents with relevant tags")

            #get top hits
            # Count the occurrences of each UUID
            uuid_counts = Counter(doc_uuids)

            # Find the UUID with the highest occurrence
            most_common_uuid, most_common_count = uuid_counts.most_common(1)[0]

            doc_text = get_document_text_from_uuid(doc_db_name, most_common_uuid)

            context.append(doc_text)
        
        print("context: ", context)
        print("answering...")
        answer = llm_handler.generate_response_with_context(prompt, context)
                    
        print("Answer: ", answer)


finally:
    tag_handler.release_model()
    llm_handler.release_model()

    tag_handler.delete_database("test_concert_tags")
    delete_database(doc_db_name)

    print("Database deletion complete.")
