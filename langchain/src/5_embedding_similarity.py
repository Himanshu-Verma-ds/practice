from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()


model= HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

docs= [
    "What are the best practices for increasing model performance in machine learning?",
    "Explain how to clean data for better machine learning results.",
    "What is the most efficient algorithm for training a classification model?",
    "How can hyperparameter tuning impact the accuracy of a model?",
    "Describe ways to handle imbalanced datasets in ML.",
    "What tools can be used to visualize training performance?",
    "How does overfitting affect model accuracy?",
    "Is feature selection important for model performance?",
    "What metrics are useful to evaluate classification models?",
    "List common issues that reduce ML model accuracy."
]

query = "How can I improve the accuracy of my machine learning model?"

docs_embedding= model.embed_documents(docs)
query_embedding= model.embed_query(query)

scores= cosine_similarity([query_embedding], docs_embedding)
scores= scores[0]

idx, score= sorted(list(enumerate(scores)), key= lambda x:x[1], reverse= True)[0]

print('Query: ', query)
print('Index of the embedding having highest similarity score: ', idx)
print('Best matching document: ', docs[idx])
print('Score: ', score)