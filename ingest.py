import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

file_paths = [
    "data/engineering-agent.md",
    "data/job-agent.md",
]

documents = []

for file_path in file_paths:
    loader = TextLoader(
        file_path,
        encoding="utf-8",
    )

    loaded_documents = loader.load()
    documents.extend(loaded_documents)

print("Document 数量:", len(documents))

for i, document in enumerate(documents):
    print(f"\n===== Document {i} =====")
    print("metadata:", document.metadata)
    print(document.page_content[:100])

headers_to_split_on = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on,
    strip_headers=False,
)

markdown_sections = []

for document in documents:
    sections = markdown_splitter.split_text(
        document.page_content
    )

    for section in sections:
        section.metadata = {
            **document.metadata,
            **section.metadata,
        }

        header_1 = section.metadata.get("header_1")

        if header_1 and not section.page_content.startswith(
            f"# {header_1}"
        ):
            section.page_content = (
                f"# {header_1}\n"
                f"{section.page_content}"
            )

    markdown_sections.extend(sections)


print("Markdown Section 数量:", len(markdown_sections))
# 
for i, section in enumerate(markdown_sections):
    print(f"\n===== Section {i} =====")
    print("metadata:", section.metadata)
    print(section.page_content)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunks = splitter.split_documents(markdown_sections)


print("\n最终 Chunk 数量:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n===== Chunk {i} =====")
    print("metadata:", chunk.metadata)
    print(chunk.page_content)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)


PERSIST_DIRECTORY = "./chroma_db"

if os.path.exists(PERSIST_DIRECTORY):
    shutil.rmtree(PERSIST_DIRECTORY)


vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)


print("向量数据库创建完成")
print(f"Chunk数量: {len(chunks)}")
