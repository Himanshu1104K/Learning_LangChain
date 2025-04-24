from firebase_admin import firestore, credentials
import firebase_admin
from dotenv import load_dotenv
from langchain_google_firestore import FirestoreChatMessageHistory
from langchain_anthropic import ChatAnthropic

load_dotenv()

cred = credentials.Certificate("../serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


print("Initializing Firestore chat Message history...")

SESSION_ID = "user_session_new"
COLLECTION_NAME = "chat_history"

chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID, collection=COLLECTION_NAME, client=db
)

print("Chat history initialized successfully.")
print(f"Current chat history: {chat_history.messages}")


model = ChatAnthropic(model="claude-3-opus-20240229")
print("Start Chat with AI. Type 'exit to quit.")

while True:
    query = input("User : ")
    if query.lower() == "exit":
        break

    chat_history.add_user_message(query)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI : {ai_response.content}")


print("Chat ended.")
