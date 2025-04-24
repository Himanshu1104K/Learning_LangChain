from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about the {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"animal": "cat", "fact_count": 2})

print(result)
