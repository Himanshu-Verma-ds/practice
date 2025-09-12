from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def multiply(a: float,
             b: float) -> float:
    
    """
    Returns the multiplication result of 2 numbers
    
    Parameters:
    a (float): First number to be multiplied
    b (float): Second number to be multipled
    
    Returns: 
    result (float): Result of the multiplication done
    """
    result= a*b
    return result 

query= HumanMessage("Can you provide multiplication of 3 and 100 ?")
messages= [query]

# binding the tool with llm
model= ChatGoogleGenerativeAI(model= 'gemini-2.0-flash')
model_with_tools= model.bind_tools([multiply])

# prompting the llm with tools with the appended messages list
result= model_with_tools.invoke(messages)
messages.append(result)

# invoking the tool with the output of the llm with tool, returns a ToolMessage
tool_result= multiply.invoke(result.tool_calls[0])
messages.append(tool_result)

# final result is prompting the llm with tool with the message that contains the output of the result of the tool (ToolMessage)
final_result= model_with_tools.invoke(messages)
print(final_result)
