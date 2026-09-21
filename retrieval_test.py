from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


query = "招聘网站 找工作 招聘平台 职位搜索 岗位搜索"

results = vectorstore.similarity_search_with_score(
    query=query,
    k=8,
)


print("Query:", query)

for rank, (document, score) in enumerate(results):
    print(f"\n===== Rank {rank} =====")
    print("score:", score)
    print("metadata:", document.metadata)
    print(document.page_content)
