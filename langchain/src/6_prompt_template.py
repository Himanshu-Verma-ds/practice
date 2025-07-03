from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
load_dotenv()


model= HuggingFaceEndpoint(repo_id= "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                           task= "text-generation")

model= ChatGoogleGenerativeAI(model= "gemini-2.0-flash")


# loading the prompt instead of defining it every time
prompt = load_prompt('prompt_template.json')


subject= input("Mention the subject: ")
audience= input("Mention the audience: ")
tone= input("Mention the tone: ")

prompt= prompt.invoke({
    'subject': subject,
    'audience': audience,
    'tone': tone
})

print()
result= model.invoke(prompt)
print("Result: \n", result.content)