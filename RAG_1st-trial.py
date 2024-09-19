from langchain_together import ChatTogether
from langchain_together import TogetherEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# LLM
llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="a3f5c60b10317ed3745810a435037f2e2ace916529db7f5585696f709d628f94",
)

# EMBEDDER
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key="a3f5c60b10317ed3745810a435037f2e2ace916529db7f5585696f709d628f94",
)

# LOADING
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Badminton")
docs = loader.load()

# SPLITTING
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

# EMBEDDING
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
# print(vectorstore._collection.get(include=["metadatas", "embeddings", "documents"]))

# RETRIEVING
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retrieved_docs = retriever.invoke("Tell me something about badminton?")
# print(retrieved_docs[0].page_content)

# GENERATION
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
response = rag_chain.invoke({"input": "What is RAG"})
print(response["answer"])