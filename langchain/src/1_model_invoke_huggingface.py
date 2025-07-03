from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

llm= HuggingFaceEndpoint(repo_id= "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                         task= "text-generation")

model= ChatHuggingFace(llm=llm)
question= "What are the number of states in India?"
result= model.invoke(question)
print(result.content)