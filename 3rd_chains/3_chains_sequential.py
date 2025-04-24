from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about the {animal}."),
        ("human", "Tell me {fact_count} facts."),
    ]
)

Translation_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a translator and convert the provided text into {language}.",
        ),
        ("human", "Translate the following text to {language} : {text}"),
    ]
)

count_words = RunnableLambda(lambda x: f"Word Count : {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(
    lambda output: {"text": output, "language": "hinglish"}
)

chain = (
    animal_facts_template
    | llm
    | StrOutputParser()
    | prepare_for_translation
    | Translation_template
    | llm
    | StrOutputParser()
)
result = chain.invoke({"animal": "cat", "fact_count": 2})

print(result)
