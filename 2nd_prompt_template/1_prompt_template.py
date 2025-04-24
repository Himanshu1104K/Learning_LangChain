from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatAnthropic(model="claude-3-opus-20240229")

# template = "Write a {tone} email to {company} expressing interest in the {position}, mentioning {skill} as a key strength. Keep it to 4 lines max."

# prompt_template = ChatPromptTemplate.from_template(template)

# prompt = prompt_template.invoke(
#     {
#         "tone": "energetic",
#         "company": "Samsung",
#         "position": "AI Engineer",
#         "skill": "AI",
#     }
# )

message = [
    ("system", "You are comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt = ChatPromptTemplate.from_messages(message)

prompt = prompt.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)

print(prompt)
print(result.content)
