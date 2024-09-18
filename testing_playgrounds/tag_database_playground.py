from tag_database_handler import *

# see if model loading and caching works
tag_db = TagDatabaseHandler()

# create a database
print(tag_db.create_database("test_db_cs"))

# # add an entry to the database

to_add_ml = [
    "supervised_learning", "unsupervised_learning", "reinforcement_learning", "deep_neural_networks",
    "decision_trees", "random_forest", "support_vector_machines", "k_nearest_neighbors", "k_means_clustering",
    "gradient_boosting", "convolutional_neural_network", "recurrent_neural_network", "natural_language_processing",
    "generative_adversarial_networks", "bayesian_networks", "feature_extraction", "dimensionality_reduction",
    "overfitting_prevention", "hyperparameter_optimization", "cross_validation"
]

to_add_se = [
    "continuous_integration", "continuous_delivery", "version_control_systems", "unit_testing", "agile_methodologies",
    "scrum_framework", "software_development_lifecycle", "object_oriented_programming", "functional_programming",
    "microservices_architecture", "monolithic_architecture", "design_patterns", "dependency_injection",
    "test_driven_development", "refactoring_techniques", "software_maintenance", "api_development",
    "production_branch", "devops_pipeline", "software_testing_automation"
]

to_add_cloud = [
    "amazon_web_services", "microsoft_azure", "google_cloud_platform", "cloud_storage", "serverless_architecture",
    "infrastructure_as_code", "containerization", "kubernetes_clusters", "virtual_machines", "multi_cloud_strategy",
    "cloud_security", "hybrid_cloud", "edge_computing", "cloud_scalability", "distributed_computing",
    "cloud_orchestration", "load_balancing", "cloud_cost_optimization", "platform_as_a_service", "software_as_a_service"
]

to_add_academia = [
    "computational_complexity", "turing_machines", "lambda_calculus", "quantum_computing",
    "post_quantum_cryptography", "graph_theory", "automata_theory", "computability_theory", "formal_languages",
    "cryptographic_algorithms", "parallel_processing", "computer_architecture", "information_theory",
    "distributed_systems", "data_structures", "algorithm_analysis", "software_verification", "concurrency_control",
    "data_encryption_algorithms", "symbolic_computation"
]

tag_db.add_entry_to_database("test_db_cs", to_add_ml)
tag_db.add_entry_to_database("test_db_cs", to_add_se)
tag_db.add_entry_to_database("test_db_cs", to_add_cloud)
tag_db.add_entry_to_database("test_db_cs", to_add_academia)


# get nearest neighbors
neighbors = tag_db.get_nearest_neighbors("test_db_cs", "cloud_compute")

print(neighbors)

neighbors = tag_db.get_nearest_neighbors("test_db_cs", "software_engineering")

print(neighbors)

neighbors = tag_db.get_nearest_neighbors("test_db_cs", "academia")

print(neighbors)

neighbors = tag_db.get_nearest_neighbors("test_db_cs", "machine_learning")

print(neighbors)

# delete an entry from the database
