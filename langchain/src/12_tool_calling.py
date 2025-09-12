from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from dotenv import load_dotenv
load_dotenv()

model= ChatGoogleGenerativeAI(model= 'gemini-2.0-flash')
# result= model.invoke('What is the capital of France?')
# print(result.content)


@tool
def multiply(a: float,
             b: float) -> float:
    """
    Returns the result after multiplication given 2 numbers
    
    Parameters:
    a (float): First number to multiply
    b (float): Second number to multiply 
    
    Returns:
    result (float): Result of multiplication of 2 numbers
    """
    result= a*b
    return result 

# fetching information of tool defined
print('Name of the tool: ', multiply.name)
print('Description of the tool: ', multiply.description)
print('Schema of the tool: ', multiply.args)


# Tool binding
model_with_tools= model.bind_tools([multiply])
print('model with tool: ', model_with_tools)


# Tool calling
# [NOTE: LLM does not actually executes the tool/function it just suggests what tool to call and with what arguments]
print()
result= model_with_tools.invoke("Can you multiply 10 with 20?")
print('Multiplication question: ', result)

print()
print('Only the tools called: ', result.tool_calls)


# We are now executing the tool: Passing the result of tool_call given by llm to the actual function/tool that we defined
print('Executing the tool ........')
print(multiply.invoke(result.tool_calls[0]))