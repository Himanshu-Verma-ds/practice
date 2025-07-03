from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model= "gemini-2.0-flash")

chat_prompt_template= ChatPromptTemplate([
    ('system', 'You are a helpfull assistant expertised in {domain}'),
    ('human', 'Explain {topic} in simple terms')
])

prompt= chat_prompt_template.invoke({
                                    'domain': 'machine learning',
                                    'topic': 'Decision trees'
                                })

print('Prompt: \n', prompt)

response= model.invoke(prompt)
print('\nResult: \n')
print(response.content)
