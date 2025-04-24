from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableBranch
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback : {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a response addressing the following negative feedback: {feedback}.",
        ),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback : {feedback}.",
        ),
    ]
)

excalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate the following feedback to human agent : {feedback}.",
        ),
    ]
)

Classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Classify the following feedback into one of the following categories : positive, negative, neutral, excalate : {feedback}.",
        ),
    ]
)

classification_chain = Classification_template | llm | StrOutputParser()

branches = RunnableBranch(
    (lambda x: "positive" in x, positive_feedback_template | llm | StrOutputParser()),
    (lambda x: "negative" in x, negative_feedback_template | llm | StrOutputParser()),
    (lambda x: "neutral" in x, neutral_feedback_template | llm | StrOutputParser()),
    excalate_feedback_template | llm | StrOutputParser(),
)

chain = classification_chain | branches

review = "The product is very terrible and I hate it. it broke after 2 days of use"

result = chain.invoke({"feedback": review})

print(result)
