from langchain_together import ChatTogether,TogetherEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List

# LLM
llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="a3f5c60b10317ed3745810a435037f2e2ace916529db7f5585696f709d628f94",
)

# LOADING
loader = WebBaseLoader("https://www.promptingguide.ai/techniques/rag")
docs = loader.load()

# SPLITTING
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
all_splits = text_splitter.split_documents(docs)

# EMBEDDING
embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    api_key="a3f5c60b10317ed3745810a435037f2e2ace916529db7f5585696f709d628f94",
)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
# print(vectorstore._collection.get(include=["metadatas", "embeddings", "documents"]))

# RETRIEVING
class LineListOutputParser(BaseOutputParser[List[str]]):
    """Output parser for a list of lines."""
    def parse(self, text: str) -> List[str]:
        lines = text.strip().split("\n")
        return list(filter(None, lines))  # Remove empty lines
output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate five 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Chain
llm_chain = QUERY_PROMPT | llm | output_parser

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6}), 
    llm_chain=llm_chain, 
    parser_key="lines"
) 

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