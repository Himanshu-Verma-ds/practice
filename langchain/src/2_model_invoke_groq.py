from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


model= ChatGroq(model= "qwen/qwen3-32b")
question= "What are the number of states in India?"
result= model.invoke(question)
print(result.content)