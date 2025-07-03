from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

model= HuggingFaceEmbeddings(model_name= "sentence-transformers/all-MiniLM-L6-v2")

docs= [
    "How many states are there in India?",
    "How is the weather here in India?",
    "Is India a union of states?"
]

result= model.embed_documents(docs)
print(result)


if len(result) > 1:
    print('Number of sentences embedded: ', len(result))
    print('Dimension of each embedding: ', len(result[0]))
    
else:
    print('Dimension of embedding: ', len(result))
