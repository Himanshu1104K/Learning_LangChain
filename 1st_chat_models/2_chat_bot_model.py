"""
Types of Messages in LangChain

1. SystemMessage : Define the AI's role and sets the context for the Conversation.
2. HumanMessage : Represents the User's input or query.
3. AIMessage : The AI's response to the HumanMessage.

"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo")

messages = [
    SystemMessage("You are an expert in the social media content strategy"),
    HumanMessage("Give a short tip to create engagin posts on Instagram?"),
]

result = llm.invoke(messages)

print(result.content)
