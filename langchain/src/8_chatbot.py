from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()


model= GoogleGenerativeAI(model= "gemini-2.0-flash")

messsages= [
    SystemMessage("You are a helpfull asisstant, expert in your domain")
]

while True:
    user= input(HumanMessage())