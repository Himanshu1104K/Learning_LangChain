from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


messages = [
    SystemMessage("You are an expert in the social media content strategy"),
    HumanMessage("Give a short tip to create engagin posts on Instagram?"),
]


llm = ChatOpenAI(model="gpt-3.5-turbo")
model = ChatAnthropic(model="claude-3-opus-20240229")
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# result = llm.invoke(messages)
result = model.invoke(messages)

print(result.content)
