import time
from doc_database_handler import *

db_name = create_database("test_db_cs")

print(db_name)

databases = get_existing_databases()
first_db = databases[0]
print(first_db['title'] + " " + first_db['last_modified'] + " " + first_db['file']) 

# time.sleep(5)

# update_database_title(first_db['file'], "test_db_cs_modified")
# databases = get_existing_databases()
# first_db = databases[0]
# print(first_db['title'] + " " + first_db['last_modified'] + " " + first_db['file'])

# time.sleep(5)

# update_database_title(first_db['file'], "test_db_cs")
# databases = get_existing_databases()
# first_db = databases[0]
# print(first_db['title'] + " " + first_db['last_modified'] + " " + first_db['file'])

cloud_compute_text = """
Cloud computing is a critical component in modern technology infrastructures, enabling businesses to build, deploy, and scale applications efficiently. Three major concepts stand out in cloud computing: Amazon Web Services (AWS), scalable code development, and serverless functions such as AWS Lambda.

Amazon Web Services (AWS) is a cloud platform offered by Amazon, providing a vast range of services including computing power, storage options, and machine learning. AWS is one of the most popular cloud platforms and offers highly flexible, scalable, and secure services. It is often the first choice for companies looking to migrate their applications to the cloud. AWS offers services like EC2 for compute capacity, S3 for storage, and various database services, which allow businesses to scale up or down based on their needs.

Building scalable code is a key requirement in cloud-based systems. Scalability ensures that applications can handle increasing loads by efficiently managing resources. Developers focus on optimizing their code for cloud environments, considering factors such as server allocation, dynamic load balancing, and fault tolerance. Techniques like microservices architecture, horizontal scaling, and distributed databases help create robust, high-performance systems that adapt to user demand without crashing or requiring major code rewrites.

AWS Lambda is an example of a serverless computing model, allowing developers to run code without provisioning or managing servers. Lambda functions are executed in response to events, such as HTTP requests or database updates, and automatically scale based on the workload. This serverless approach reduces operational complexity and cost, making it easier for developers to focus on writing code instead of managing infrastructure. AWS Lambda is especially useful for building event-driven architectures, simplifying the process of developing modern cloud-native applications.
"""

cloud_compute_subdocs = [
    [
        "Cloud computing allows businesses to build, deploy, and scale applications using various cloud services. It includes key technologies like Amazon Web Services (AWS), scalable code development, and serverless functions such as AWS Lambda. These concepts form the backbone of modern cloud infrastructure.", 
        ["cloud_computing", "aws", "scalable_code", "aws_lambda"]
    ],
    [
        "Amazon Web Services (AWS) is a cloud platform offered by Amazon that provides a range of services such as EC2 for compute capacity, S3 for storage, and various database services. It is highly flexible, scalable, and secure, making it a popular choice for businesses migrating to the cloud.", 
        ["aws", "amazon_web_services", "cloud_platform", "ec2", "s3"]
    ],
    [
        "Building scalable code ensures that cloud applications can handle increasing loads by optimizing resource management. Techniques such as microservices architecture, horizontal scaling, and distributed databases help build systems that can grow dynamically with user demand.", 
        ["scalable_code", "microservices_architecture", "horizontal_scaling", "distributed_databases"]
    ],
    [
        "AWS Lambda is a serverless computing model that allows developers to run code without managing servers. It scales automatically based on workload and simplifies event-driven architectures by responding to events like HTTP requests or database updates.", 
        ["aws_lambda", "serverless_computing", "event_driven_architecture", "lambda_functions"]
    ]
]

academia_text = """
In computer science, theoretical foundations play a crucial role in understanding the limits and capabilities of computation. Three major topics in the field are Turing machines, NP-completeness, and the Church-Turing thesis. These concepts provide a framework for analyzing the computational power of machines, the complexity of problems, and the fundamental limits of what can be computed.

A Turing machine is a theoretical model of computation that defines an abstract machine capable of manipulating symbols on a tape according to a set of rules. The Turing machine is considered one of the simplest and most powerful models of computation, laying the foundation for modern computer science. Turing machines can simulate any algorithm, making them a central concept in the study of computability and complexity.

The concept of NP-completeness is central to computational complexity theory, which classifies problems based on how difficult they are to solve. NP-complete problems are the hardest problems in the class NP (nondeterministic polynomial time), meaning that if one NP-complete problem can be solved efficiently, then all NP problems can be solved efficiently. However, no one has yet proven whether NP-complete problems can be solved in polynomial time, which is one of the greatest unsolved questions in computer science.

The Church-Turing thesis is a hypothesis about the nature of computation, stating that any function that can be computed algorithmically can also be computed by a Turing machine. This thesis forms the foundation of theoretical computer science, bridging the gap between algorithms and mechanical computation, and providing a unified framework for understanding what it means for a function to be computable.
"""

academia_subdocs = [
    [
        "Theoretical computer science focuses on the limits of computation, and three foundational concepts are Turing machines, NP-completeness, and the Church-Turing thesis. These ideas help to define what can be computed and how efficiently problems can be solved.", 
        ["theoretical_computer_science", "turing_machines", "np_complete", "church_turing_thesis"]
    ],
    [
        "A Turing machine is a theoretical construct that manipulates symbols on a tape according to predefined rules. It is capable of simulating any algorithm and is a foundational concept in the theory of computation.", 
        ["turing_machines", "computability_theory", "theoretical_computation", "tape_symbol_manipulation"]
    ],
    [
        "NP-complete problems are the hardest problems in NP, a class of problems that can be verified by a nondeterministic polynomial-time machine. The unresolved question of whether NP-complete problems can be solved in polynomial time is a major challenge in computer science.", 
        ["np_complete", "complexity_theory", "polynomial_time", "computational_complexity"]
    ],
    [
        "The Church-Turing thesis posits that any function that can be computed algorithmically can be computed by a Turing machine. This thesis is a key concept in theoretical computer science, linking algorithms to mechanical computation.", 
        ["church_turing_thesis", "computability", "algorithmic_computation", "turing_machines"]
    ]
]

software_engineering_text = """
Software engineering is a discipline that applies engineering principles to the development and maintenance of software systems. Three important topics in modern software engineering are continuous deployment, refactoring techniques, and the separation of development and production environments. Each of these concepts plays a crucial role in the efficiency, quality, and stability of software development workflows.

Continuous deployment is a software release practice that enables teams to automatically deploy new changes to production as soon as they pass automated testing. This approach allows for rapid iteration, reduces the risk of manual errors, and ensures that the latest features and fixes are delivered to users quickly. Continuous deployment is often integrated into CI/CD (Continuous Integration/Continuous Deployment) pipelines to maintain high-quality code and minimize downtime.

Refactoring refers to the process of restructuring existing code without changing its external behavior. Refactoring techniques improve the internal structure of the code, making it cleaner, more efficient, and easier to maintain. Common refactoring methods include renaming variables for clarity, breaking down large functions into smaller ones, and simplifying complex logic. Effective refactoring reduces technical debt and improves the long-term sustainability of the codebase.

The separation between development and production environments is a critical concept in software engineering. The development space is where new features, updates, and experiments are implemented and tested. The production space is where the live, stable version of the software runs, serving end-users. Maintaining a clear distinction between these environments ensures that changes can be thoroughly tested in a controlled environment before being deployed to production, minimizing the risk of introducing bugs to live users.
"""

software_engineering_subdocs = [
    [
        "In software engineering, key concepts like continuous deployment, refactoring techniques, and the separation of development and production spaces are essential for maintaining a high-quality and efficient development process.", 
        ["software_engineering", "continuous_deployment", "refactoring_techniques", "development_space", "production_space"]
    ],
    [
        "Continuous deployment allows for the automatic release of software updates to production as soon as they pass automated tests. This method integrates closely with CI/CD pipelines and ensures rapid delivery of new features and bug fixes to users.", 
        ["continuous_deployment", "ci_cd", "automated_testing", "software_release", "rapid_iteration"]
    ],
    [
        "Refactoring techniques improve the structure of existing code without altering its external behavior. Techniques like renaming variables, breaking down large functions, and simplifying logic help make code more maintainable and reduce technical debt.", 
        ["refactoring_techniques", "code_maintenance", "technical_debt", "code_cleanup", "simplification"]
    ],
    [
        "Development and production spaces are distinct environments in the software lifecycle. The development space is used for building and testing, while the production space is where the live application runs, serving real users. Separating these environments reduces the risk of introducing bugs into production.", 
        ["development_space", "production_space", "software_environments", "testing", "live_application"]
    ]
]

add_entry_to_database(first_db['file'], text=cloud_compute_text, sub_docs=cloud_compute_subdocs, document_type="text")
add_entry_to_database(first_db['file'], text=academia_text, sub_docs=academia_subdocs, document_type="text")
add_entry_to_database(first_db['file'], text=software_engineering_text, sub_docs=software_engineering_subdocs, document_type="text")

db_tags = get_all_tags(first_db['file'])

print(db_tags)

print("document uuid tags from tag")
uuids, tags = get_document_uuid_tags_from_tag(first_db['file'], "live_application")
print(uuids)
document_text = get_document_text_from_uuid(first_db['file'], uuids[0])
print(document_text)

print("multiple matches test")
uuids, tags = get_document_uuid_tags_from_tag(first_db['file'], "development_space")
print(uuids)
for uuid in uuids:
    document_text = get_document_text_from_uuid(first_db['file'], uuid)
    print(document_text)

print("multiple tags multiple matches test")
uuids, tags = get_document_uuid_tags_from_tag(first_db['file'], ["development_space", "software_environments"])
print(uuids)
print(tags)
for uuid in uuids:
    document_text = get_document_text_from_uuid(first_db['file'], uuid)
    print(document_text)

print("get original text test")
original_document = get_original_document_from_document_uuid(first_db['file'], uuids[0])
original_text = original_document[0]
print(original_text)

print("get original text from search test")
original_documents = get_original_documents_from_textual_match(first_db['file'], "discipline that applies")
print(original_documents[0][1])

print("get document uuids from original document test")
sub_doc_uuids = get_documents_uuids_from_original_document(first_db['file'], original_documents[0][0])
print(sub_doc_uuids)
sub_doc_text = get_document_text_from_uuid(first_db['file'], sub_doc_uuids[0])
print(sub_doc_text)

#the big one
print("test removal of a original document")
#should remove software engineering document
remove_original_document(first_db['file'], original_documents[0][0])
#should also have removed subdocs and tags
original_documents = get_original_documents_from_textual_match(first_db['file'], "discipline that applies")
#should print nothing
print(original_documents)
print("tag removal test")
print(get_all_tags(first_db['file']))

print("clearing database test")
delete_database(first_db['file'])

# show no databases exist
databases = get_existing_databases()
print("database count: " + str(len(databases)))

# test tag stacking
# First original document about DevOps pipeline and automated testing
devops_1_text = """
The DevOps pipeline is a crucial aspect of modern software development, streamlining the process from code commit to deployment. It enables continuous integration and delivery, ensuring that software updates are tested and deployed efficiently. By automating many of the steps in the development lifecycle, the DevOps pipeline helps reduce human error and speeds up the release of new features and fixes.

Automated testing is an essential part of modern software development. It involves running tests automatically after each code change to ensure that the software remains functional and free from bugs. Automated testing reduces the time required for manual testing and provides fast feedback to developers, allowing issues to be addressed before they reach production.
"""

# First document's sub-docs
devops_1_subdocs = [
    [
        "The DevOps pipeline is a streamlined approach to software development, focusing on continuous integration and delivery. It automates many steps, reducing human error and ensuring faster deployment.", 
        ["devops_pipeline", "continuous_integration", "continuous_delivery", "automation"]
    ],
    [
        "Automated testing is a critical practice in software engineering, ensuring that code changes are tested quickly and efficiently. This reduces manual testing time and helps identify issues early in the development cycle.", 
        ["automated_testing", "software_testing", "testing_automation", "fast_feedback"]
    ]
]

# Second original document about DevOps pipeline and infrastructure as code
devops_2_text = """
The DevOps pipeline integrates tools and practices that automate the flow of software from development to production. With continuous integration and continuous delivery, the pipeline ensures that code is always in a deployable state. This automation enhances collaboration between development and operations teams and speeds up the deployment process, reducing downtime and improving system reliability.

Infrastructure as code (IaC) is another key element of modern DevOps practices. IaC involves managing and provisioning computing resources through machine-readable scripts rather than physical hardware configurations. By codifying infrastructure, teams can automate the setup of environments, ensuring consistency across development, testing, and production stages.
"""

# Second document's sub-docs
devops_2_subdocs = [
    [
        "The DevOps pipeline is a key practice that automates the process of moving code from development to production, enabling continuous integration and delivery. This improves collaboration between development and operations teams.", 
        ["devops_pipeline", "continuous_integration", "continuous_delivery", "collaboration"]
    ],
    [
        "Infrastructure as code (IaC) allows teams to manage infrastructure through code, enabling automated environment setup. This ensures consistent and repeatable deployments across different environments.", 
        ["infrastructure_as_code", "iac", "automated_infrastructure", "consistent_deployment"]
    ]
]

db_name = create_database("test_db_devops")



add_entry_to_database(db_name, text=devops_1_text, sub_docs=devops_1_subdocs, document_type="text")
add_entry_to_database(db_name, text=devops_2_text, sub_docs=devops_2_subdocs, document_type="text")

# Check if the tags are present
db_tags = get_all_tags(db_name)
print(db_tags)

#show doc counts
print("# of original documents: " + str(get_number_of_original_documents(db_name)))
print("# of sub-documents: " + str(get_number_of_documents(db_name)))

print("deleting one original document")
original_documents = get_original_documents_from_textual_match(db_name, "The DevOps pipeline is a crucial aspect of modern software development")

remove_original_document(db_name, original_documents[0][0])

print("# of original documents: " + str(get_number_of_original_documents(db_name)))
print("# of sub-documents: " + str(get_number_of_documents(db_name)))

print("tags after deletion, devops_pipeline should still be present")
print(get_all_tags(db_name))
print("deleting the other original document")
original_documents = get_original_documents_from_textual_match(db_name, "The DevOps pipeline integrates tools and practices that automate the")

remove_original_document(db_name, original_documents[0][0])

print("# of original documents: " + str(get_number_of_original_documents(db_name)))
print("# of sub-documents: " + str(get_number_of_documents(db_name)))

print("tags after deletion, devops_pipeline should be gone")
print(get_all_tags(db_name))

print("adding files back, testing tag retrieval")
add_entry_to_database(db_name, text=devops_1_text, sub_docs=devops_1_subdocs, document_type="text")
add_entry_to_database(db_name, text=devops_2_text, sub_docs=devops_2_subdocs, document_type="text")

tagged_docs = get_all_document_uuids_from_tag(db_name, "devops_pipeline")
 
print("tagged docs devops_pipeline")
print(tagged_docs)

tagged_docs = get_all_document_uuids_from_tag(db_name, "infrastructure_as_code")

print("tagged docs infrastructure_as_code")
print(tagged_docs)

print("testing tags from uuid retrieval")

tags = get_tags_from_document_uuid(db_name, tagged_docs[0])

print("tags from uuid")
print(tags)




print("clearing database...")
delete_database(db_name)

