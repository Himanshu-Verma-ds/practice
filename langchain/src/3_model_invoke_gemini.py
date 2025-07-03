from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()


model= ChatGoogleGenerativeAI(model="gemini-2.0-flash")

question= "What are the number of states in India ?"
result= model.invoke(question)
print(result.content)