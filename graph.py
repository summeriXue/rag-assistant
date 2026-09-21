from typing import TypedDict

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field


class RAGState(TypedDict):
    original_question: str
    search_query: str
    documents: list[Document]

    answer: str

    documents_relevant: bool
    answer_supported: bool
    verification_feedback: str

    retrieval_retry_count: int
    generation_retry_count: int


class DocumentGrade(BaseModel):
    relevant: bool = Field(
        description="检索到的资料是否足以回答用户问题"
    )


class RerankScore(BaseModel):
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="该候选资料对回答用户问题的相关程度，0 表示完全无关，1 表示高度相关且直接有助于回答"
    )


class AnswerVerification(BaseModel):
    supported: bool = Field(
        description="回答中的事实性内容是否完全受到资料支持"
    )
    feedback: str = Field(
        description="验证结果的原因；如果不受支持，指出具体哪些内容缺少资料依据"
    )


load_dotenv()


RETRIEVAL_CANDIDATE_K = 4
RERANK_TOP_N = 2

MAX_RETRIEVAL_RETRIES = 2
MAX_GENERATION_RETRIES = 2


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
)

document_grader = llm.with_structured_output(DocumentGrade)
document_reranker = llm.with_structured_output(RerankScore)
answer_verifier = llm.with_structured_output(AnswerVerification)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": RETRIEVAL_CANDIDATE_K
    }
)


def transform_query(state: RAGState):
    original_question = state["original_question"]

    prompt = f"""
你是一个 RAG 检索查询优化器。

你的任务不是回答用户问题，而是把用户问题转换成更适合语义检索的查询。

优化时重点执行以下操作：

1. 保留用户原始问题的真实意图。
2. 删除“我想让它”“有没有这个能力”“怎么办”等对检索帮助较小的口语表达。
3. 对用户问题中的核心概念补充常见同义表达或近义表达。
4. 优先使用描述“对象、动作、问题类型”的关键词。
5. 可以保留多个具有相同意图的表达，以提高语义检索召回率。
6. 不要回答问题。
7. 不要猜测具体工具名、函数名、类名或知识库中特有的专有名称。
8. 不要添加用户问题无法合理推断出的新事实。
9. 只输出优化后的检索查询，不要解释。

示例：

用户问题：
我想让它帮我从招聘网站上找工作，它有这个能力吗？

检索查询：
招聘网站 找工作 招聘平台 职位搜索 岗位搜索

用户问题：

{original_question}

检索查询：
"""

    response = llm.invoke(prompt)
    transformed_query = response.content.strip()

    print("\n[transform_query]")
    print("original_question:", original_question)
    print("transformed_query:", transformed_query)

    return {
        "search_query": transformed_query
    }


def retrieve(state: RAGState):
    search_query = state["search_query"]

    print("\n[retrieve]")
    print("search_query:", search_query)

    docs = retriever.invoke(search_query)

    print("documents:", len(docs))

    for i, doc in enumerate(docs):
        print(f"\n--- Retrieved {i} ---")
        print("metadata:", doc.metadata)
        print(doc.page_content)

    return {
        "documents": docs
    }


def rerank_documents(state: RAGState):
    original_question = state["original_question"]
    documents = state["documents"]

    scored_documents = []

    print("\n[rerank_documents]")

    for i, document in enumerate(documents):
        prompt = f"""
你是一个 RAG 文档重排序器。

请判断下面这份候选资料对于回答用户原始问题有多大帮助。

评分范围为 0.0 到 1.0：
- 1.0 表示资料直接包含回答问题所需的信息
- 0.5 表示资料与问题相关，但不能直接回答问题
- 0.0 表示资料与问题无关

用户原始问题：

{original_question}

候选资料：

{document.page_content}
"""

        result = document_reranker.invoke(prompt)
        score = result.relevance_score

        print(f"document {i} score:", score)
        print("metadata:", document.metadata)

        scored_documents.append(
            (document, score)
        )

    scored_documents = sorted(
        scored_documents,
        key=lambda item: item[1],
        reverse=True,
    )

    top_documents = [
        document
        for document, score in scored_documents[:RERANK_TOP_N]
    ]

    print(f"\nReranked Top {RERANK_TOP_N}:")

    for i, (document, score) in enumerate(
        scored_documents[:RERANK_TOP_N]
    ):
        print(f"\n--- Reranked {i} ---")
        print("score:", score)
        print("metadata:", document.metadata)
        print(document.page_content)

    return {
        "documents": top_documents
    }



def grade_documents(state: RAGState):
    original_question = state["original_question"]
    documents = state["documents"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
你是一个文档相关性判断器。

请判断下面提供的资料是否包含足够的信息来回答用户问题。

资料：

{context}

用户原始问题：

{original_question}
"""

    result = document_grader.invoke(prompt)

    is_relevant = result.relevant

    print("\n[grade_documents]")
    print("documents_relevant:", is_relevant)

    return {
        "documents_relevant": is_relevant
    }


def rewrite_query(state: RAGState):
    original_question = state["original_question"]
    search_query = state["search_query"]
    retrieval_retry_count = state["retrieval_retry_count"]

    prompt = f"""
你是一个 RAG 查询优化器。

当前查询没有检索到足以回答用户原始问题的资料。

请重写当前检索查询，使其更适合在技术知识库中进行语义检索。

要求：
1. 保留用户原始问题的真实意图。
2. 不要自己回答问题。
3. 不要添加原始问题中不存在的事实。
4. 只输出重写后的检索查询。

用户原始问题：

{original_question}

当前检索查询：

{search_query}
"""

    response = llm.invoke(prompt)

    new_query = response.content.strip()

    print("\n[rewrite_query]")
    print("old_query:", search_query)
    print("new_query:", new_query)
    print("retrieval_retry_count:", retrieval_retry_count + 1)

    return {
        "search_query": new_query,
        "retrieval_retry_count": retrieval_retry_count + 1,
    }


def generate(state: RAGState):
    original_question = state["original_question"]
    documents = state["documents"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
你是一个技术文档助手。

请严格根据下面提供的资料回答用户问题。
如果资料中没有答案，请明确说明不知道。

资料：

{context}

用户问题：

{original_question}
"""

    response = llm.invoke(prompt)

    print("\n[generate]")

    return {
        "answer": response.content
    }


def verify_answer(state: RAGState):
    documents = state["documents"]
    answer = state["answer"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
你是一个答案事实一致性检查器。

请判断下面的回答是否完全受到资料支持。

如果回答中的事实性内容都能从资料中得到支持，
则 supported 为 true。

如果回答包含资料中没有提供、无法确认或与资料冲突的事实，
则 supported 为 false，并在 feedback 中指出具体原因。

资料：

{context}

回答：

{answer}
"""

    result = answer_verifier.invoke(prompt)

    print("\n[verify_answer]")
    print("answer_supported:", result.supported)
    print("feedback:", result.feedback)

    return {
        "answer_supported": result.supported,
        "verification_feedback": result.feedback,
    }


def regenerate(state: RAGState):
    original_question = state["original_question"]
    documents = state["documents"]
    previous_answer = state["answer"]
    verification_feedback = state["verification_feedback"]
    generation_retry_count = state["generation_retry_count"]

    context = "\n\n".join(
        doc.page_content
        for doc in documents
    )

    prompt = f"""
你是一个技术文档助手。

上一次回答没有通过事实一致性检查。

请根据资料和验证反馈修正回答。

要求：
1. 严格根据资料回答用户问题。
2. 删除资料无法支持的事实。
3. 不要添加资料中不存在的信息。
4. 保留原回答中有资料依据的内容。

资料：

{context}

用户问题：

{original_question}

上一次回答：

{previous_answer}

验证反馈：

{verification_feedback}
"""

    response = llm.invoke(prompt)

    print("\n[regenerate]")
    print("generation_retry_count:", generation_retry_count + 1)

    return {
        "answer": response.content,
        "generation_retry_count": generation_retry_count + 1,
    }


def no_answer(state: RAGState):
    print("\n[no_answer]")

    return {
        "answer": "知识库中没有足够的信息回答这个问题。"
    }


def generation_failed(state: RAGState):
    print("\n[generation_failed]")

    return {
        "answer": "资料中存在相关信息，但当前未能生成通过事实一致性验证的回答。"
    }


def format_sources(documents: list[Document]):
    sources = []

    for document in documents:
        metadata = document.metadata

        source = metadata.get(
            "source",
            "unknown source",
        )

        header_1 = metadata.get("header_1")
        header_2 = metadata.get("header_2")

        parts = [source]

        if header_1:
            parts.append(header_1)

        if header_2:
            parts.append(header_2)

        source_text = " → ".join(parts)

        if source_text not in sources:
            sources.append(source_text)

    return sources


def route_after_grading(state: RAGState):
    if state["documents_relevant"]:
        return "generate"

    if state["retrieval_retry_count"] < MAX_RETRIEVAL_RETRIES:
        return "rewrite_query"

    return "no_answer"


def route_after_verification(state: RAGState):
    if state["answer_supported"]:
        return "end"

    if state["generation_retry_count"] < MAX_GENERATION_RETRIES:
        return "regenerate"

    return "generation_failed"


builder = StateGraph(RAGState)

builder.add_node("transform_query", transform_query)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank_documents", rerank_documents)
builder.add_node("grade_documents", grade_documents)
builder.add_node("rewrite_query", rewrite_query)
builder.add_node("generate", generate)
builder.add_node("verify_answer", verify_answer)
builder.add_node("regenerate", regenerate)
builder.add_node("no_answer", no_answer)
builder.add_node("generation_failed", generation_failed)

builder.add_edge(START, "transform_query")
builder.add_edge("transform_query", "retrieve")
builder.add_edge("retrieve", "rerank_documents")
builder.add_edge("rerank_documents", "grade_documents")

builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query",
        "no_answer": "no_answer",
    },
)

builder.add_edge("rewrite_query", "retrieve")
builder.add_edge("generate", "verify_answer")
builder.add_conditional_edges(
    "verify_answer",
    route_after_verification,
    {
        "end": END,
        "regenerate": "regenerate",
        "generation_failed": "generation_failed",
    },
)
builder.add_edge("regenerate", "verify_answer")
builder.add_edge("no_answer", END)
builder.add_edge("generation_failed", END)

graph = builder.compile()
