import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, "documents")
persistent_dir = os.path.join(current_dir, "db", "chroma_db_with_metadata")

print(f"Book directory: {books_dir}")
print(f"Persistent directory: {persistent_dir}")

if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist, Initializing vector store...")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"File not found: {books_dir}")

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path, encoding="utf-8")
        book_docs = loader.load()

        for doc in book_docs:
            doc.metadata = {"source": book_file}
            documents.append(doc)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunk Information ---\n")
    print(f"Total Chunks: {len(docs)}")
    print(f"First Chunk: \n{docs[0].page_content}\n")

    # Creating Embeddings
    print("\n--- Creating embeddings ---\n")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("\n--- Finished Creating embeddings ---\n")

    print("\n--- Initializing Vector Store ---\n")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_dir)
    print("\n--- Vector Store Initialized ---\n")

else:
    print("Vector Store already exists, Loading from persistent directory...")
