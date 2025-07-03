from typing import TypedDict, Optional, List, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

class Review(BaseModel):
    key_themes: list[str]= Field(description= "Write all the key themes discussed in the review in a list")
    summary: str= Field(description= "Provide a short summary of the review")
    sentiment: Literal["positive", "negative", "neutral"]= Field(description= "The final sentiment of the review provided either positive, negative or neutral")
    pros: Optional[List[str]]= Field(description= "Provide a list of pros for the given review")
    cons: Optional[List[str]]= Field(description= "Provide a list of cons for the given review")
    
    
review= Review()
print(review)

model= ChatGoogleGenerativeAI(model= "gemini-2.0-flash")
structured_output= model.with_structured_output(Review)
result= structured_output.invoke("'Zindagi Na Milegi Dobara' is a refreshing take on friendship, self-discovery, and the joy of living in the moment. The film beautifully captures the essence of travel and the healing power it holds. Hrithik Roshan, Farhan Akhtar, and Abhay Deol share genuine chemistry that feels real and relatable. Director Zoya Akhtar masterfully balances humor, emotion, and introspection. The Spanish locales are stunning and add to the visual appeal. Dialogues are crisp and meaningful, often leaving a lasting impression. Shankar-Ehsaan-Loy’s music perfectly complements the narrative, especially “Senorita” and “Der Lagi.” The poetry interludes by Javed Akhtar add a soulful depth. It’s not just a road trip movie — it’s a reminder to live fearlessly. A must-watch for those seeking a feel-good film with heart and substance.")
print(result.content)