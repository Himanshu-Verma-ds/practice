from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model= "gemini-2.0-flash")

chat_template= ChatPromptTemplate([
    ('system', 'You are a helpful customer chat support agent'),
    MessagesPlaceholder(variable_name= 'chat_history'),
    ('human', '{query}')
])


message_history= []
with open('chat_history.txt', 'r') as f:
    message_history.extend(f.readlines())
    

query= 'Where is my order ?'

prompt= chat_template.invoke({
    'chat_history': message_history,
    'query': query
})

print('Prompt: \n', prompt)

result= model.invoke(prompt)
print('\nResult: \n',result.content)