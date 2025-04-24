from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv


load_dotenv()

model = ChatAnthropic(model="claude-3-opus-20240229")

chat_history = []

system_message = SystemMessage(content="You are a helpful assistant.")
chat_history.append(system_message)

# Chat loop
while True:
    query = input("You : ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f"Assistant : {response}")


print("-------Message History-------")
print(chat_history)
