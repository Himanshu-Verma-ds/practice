from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()


model= GoogleGenerativeAI(model= "gemini-2.0-flash")

messages= [
    SystemMessage("You are a helpfull asisstant, expert in your domain")
]

while True:
    user_input= input("You: ")
    messages.append(HumanMessage(user_input))
    if user_input == 'exit':
        break 
    
    result= model.invoke(messages)
    ai_message= result
    messages.append(AIMessage(ai_message))
    print('AI: ', ai_message)

print('Entire chat history: ', messages)