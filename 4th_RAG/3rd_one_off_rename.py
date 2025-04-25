import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_anthropic import ChatAnthropic


current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db")


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

query = "Where does Gandalf meet Frodo?"

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
relevent_docs = retriever.invoke(query)

# print("\n--- Relevant Documents ---\n")
# for i, doc in enumerate(relevent_docs, 1):
#     print(f"\nDocument {i}:")
#     print(f"Content: {doc.page_content}...\n")

#     if doc.metadata:
#         print(f"Source : {doc.metadata.get('source','Unknown')}\n")

combined_input = (
    "Here are some documents that might help you answer the question: "
    + query
    + "\n\n".join([doc.page_content for doc in relevent_docs])
    + "\n\nPlease provide a rough answer to the question based only on the documents provided. If the answer is not found in the documents, please respond with i'm not sure."
)

llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

message = [("system", "You are a helpful assistant."), ("human", combined_input)]

result = llm.invoke(message)

print(result.content)
