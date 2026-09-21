from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from pydantic import BaseModel, Field

class RerankScore(BaseModel):
    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
        description="该候选资料对回答用户问题的相关程度",
    )

load_dotenv()

llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
)

document_reranker = llm.with_structured_output(RerankScore)

eval_cases = [
    # Engineering Agent / Project File Tools
    {
        "question": "Engineering Agent 怎么读取项目里的文件？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Project File Tools",
    },
    {
        "question": "如果我让它看看项目里某个 Python 文件写了什么，它靠什么做到？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Project File Tools",
    },

    # Engineering Agent / Validation
    {
        "question": "Engineering Agent 修改完代码以后怎么验证？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Validation",
    },
    {
        "question": "它把代码改完以后，怎么确认没有把项目改坏？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Validation",
    },

    # Engineering Agent / Conversation Continuation
    {
        "question": "页面刷新以后，Engineering Agent 没执行完的任务怎么办？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Conversation Continuation",
    },
    {
        "question": "干活干到一半我不小心刷新网页了，之前那个任务还能接着跑吗？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Conversation Continuation",
    },

    # Job Agent / Job Search
    {
        "question": "Job Agent 怎么搜索职位？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Job Search",
    },
    {
        "question": "我想让它帮我从招聘网站上找工作，它有这个能力吗？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Job Search",
    },

    # Job Agent / Validation
    {
        "question": "Job Agent 怎么检查职位记录是否完整？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Validation",
    },
    {
        "question": "搜到一个岗位以后，怎么检查公司名和岗位名这些信息有没有缺？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Validation",
    },

    # Job Agent / Conversation Continuation
    {
        "question": "Job Agent 搜职位中断以后应该怎么办？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Conversation Continuation",
    },
    {
        "question": "一个招聘平台搜不下去了，能不能换其他招聘来源继续找？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Conversation Continuation",
    },
    {
        "question": "这个 Agent 能帮我找招聘岗位吗？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Job Search",
    },
    {
        "question": "找工作时能不能让它去不同的平台搜职位？",
        "expected_source": "data/job-agent.md",
        "expected_header_2": "Job Search",
    },
    {
        "question": "代码改完了，我怎么知道这次修改是正常的？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Validation",
    },
    {
        "question": "网页关掉再打开以后，之前没有完成的执行还能恢复吗？",
        "expected_source": "data/engineering-agent.md",
        "expected_header_2": "Conversation Continuation",
    },
]

def is_expected_document(document, case):
    metadata = document.metadata

    return (
        metadata.get("source") == case["expected_source"]
        and metadata.get("header_2") == case["expected_header_2"]
    )

def find_expected_rank(documents, case):
    for rank, document in enumerate(documents):
        if is_expected_document(document, case):
            return rank

    return None

def transform_query(question):
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

{question}

检索查询：
"""

    response = llm.invoke(prompt)

    return response.content.strip()

def rerank_documents(question, documents, top_n=2):
    scored_documents = []

    for document in documents:
        prompt = f"""
你是一个 RAG 文档重排序器。

请判断下面这份候选资料对于回答用户问题有多大帮助。

评分范围为 0.0 到 1.0：
- 1.0 表示资料直接包含回答问题所需的信息
- 0.5 表示资料与问题相关，但不能直接回答问题
- 0.0 表示资料与问题无关

用户问题：

{question}

候选资料：

{document.page_content}
"""

        result = document_reranker.invoke(prompt)

        scored_documents.append(
            (document, result.relevance_score)
        )

    scored_documents = sorted(
        scored_documents,
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        document
        for document, score in scored_documents[:top_n]
    ]

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

K = 2

hits = 0

for i, case in enumerate(eval_cases):
    question = case["question"]

    documents = vectorstore.similarity_search(
        query=question,
        k=K,
    )

    hit = any(
        is_expected_document(document, case)
        for document in documents
    )

    if hit:
        hits += 1

    print(f"\n===== Case {i + 1} =====")
    print("Question:", question)
    print("Expected:", case["expected_header_2"])
    print("Hit:", hit)

    for rank, document in enumerate(documents):
        print(
            f"Rank {rank}:",
            document.metadata.get("header_1"),
            "/",
            document.metadata.get("header_2"),
        )

hit_rate = hits / len(eval_cases)

print("\n===== Evaluation Result =====")
print(f"Hit@{K}: {hits}/{len(eval_cases)} = {hit_rate:.1%}")

# RERANK_CANDIDATE_K = 4
# RERANK_TOP_N = 2

# rerank_hits = 0

# for i, case in enumerate(eval_cases):
#     question = case["question"]

#     candidates = vectorstore.similarity_search(
#         query=question,
#         k=RERANK_CANDIDATE_K,
#     )

#     candidate_rank = find_expected_rank(
#         candidates,
#         case,
#     )

#     documents = rerank_documents(
#         question,
#         candidates,
#         top_n=RERANK_TOP_N,
#     )

#     rerank_rank = find_expected_rank(
#         documents,
#         case,
#     )

#     if candidate_rank is None:
#         diagnosis = "Recall Failure"
#     elif rerank_rank is None:
#         diagnosis = "Reranking Failure"
#     else:
#         diagnosis = "Success"

#     hit = rerank_rank is not None

#     if hit:
#         rerank_hits += 1

#     print(f"\n===== Rerank Case {i + 1} =====")
#     print("Question:", question)
#     print("Expected:", case["expected_header_2"])
#     print("Candidate Rank:", candidate_rank)
#     print("Rerank Rank:", rerank_rank)
#     print("Diagnosis:", diagnosis)
#     print("Hit:", hit)

#     for rank, document in enumerate(documents):
#         print(
#             f"Rank {rank}:",
#             document.metadata.get("header_1"),
#             "/",
#             document.metadata.get("header_2"),
#         )

# rerank_hit_rate = rerank_hits / len(eval_cases)

# print("\n===== Comparison =====")
# print(
#     f"Vector Hit@{K}: "
#     f"{hits}/{len(eval_cases)} = {hit_rate:.1%}"
# )
# print(
#     f"Vector Top {RERANK_CANDIDATE_K} "
#     f"+ Reranker Hit@{RERANK_TOP_N}: "
#     f"{rerank_hits}/{len(eval_cases)} = "
#     f"{rerank_hit_rate:.1%}"
# )

# ==============================
# Query Transformation Evaluation
# ==============================

# TRANSFORM_EVAL_K = 8

# transform_improved = 0
# transform_same = 0
# transform_degraded = 0

# rank_deltas = []

# for i, case in enumerate(eval_cases):
#     question = case["question"]

#     print(f"\n===== Transform Case {i + 1} =====")
#     print("Original Question:", question)

#     # 1. 原始 Query 的检索结果
#     original_documents = vectorstore.similarity_search(
#         query=question,
#         k=TRANSFORM_EVAL_K,
#     )

#     original_rank = find_expected_rank(
#         original_documents,
#         case,
#     )

#     # 2. 生成更适合检索的 Query
#     transformed_query = transform_query(question)

#     print("Transformed Query:", transformed_query)

#     # 3. Transformation 后重新检索
#     transformed_documents = vectorstore.similarity_search(
#         query=transformed_query,
#         k=TRANSFORM_EVAL_K,
#     )

#     transformed_rank = find_expected_rank(
#         transformed_documents,
#         case,
#     )

#     print("Original Rank:", original_rank)
#     print("Transformed Rank:", transformed_rank)

#     if original_rank is not None and transformed_rank is not None:
#         rank_delta = original_rank - transformed_rank
#         print("Rank Delta:", rank_delta)
#     else:
#         rank_delta = None
#         print("Rank Delta:", rank_delta)

#     if rank_delta is not None:
#         rank_deltas.append(rank_delta)

#     # 4. 比较排名变化
#     if original_rank is None and transformed_rank is not None:
#         result = "Improved"
#         transform_improved += 1

#     elif original_rank is not None and transformed_rank is None:
#         result = "Degraded"
#         transform_degraded += 1

#     elif original_rank is None and transformed_rank is None:
#         result = "Same"
#         transform_same += 1

#     elif transformed_rank < original_rank:
#         result = "Improved"
#         transform_improved += 1

#     elif transformed_rank > original_rank:
#         result = "Degraded"
#         transform_degraded += 1

#     else:
#         result = "Same"
#         transform_same += 1

#     print("Result:", result)

# print("\n===== Query Transformation Result =====")
# print("Improved:", transform_improved)
# print("Same:", transform_same)
# print("Degraded:", transform_degraded)

# if rank_deltas:
#     average_rank_delta = sum(rank_deltas) / len(rank_deltas)
#     print("Average Rank Delta:", average_rank_delta)

RERANK_CANDIDATE_K = 4
RERANK_TOP_N = 2

transformed_rerank_hits = 0

for i, case in enumerate(eval_cases):
    question = case["question"]

    # 1. Query Transformation
    transformed_query = transform_query(question)

    # 2. 用改写后的 Query 做 Vector Retrieval
    candidates = vectorstore.similarity_search(
        query=transformed_query,
        k=RERANK_CANDIDATE_K,
    )

    candidate_rank = find_expected_rank(
        candidates,
        case,
    )

    # 3. Reranker 仍然根据原始问题判断候选资料
    documents = rerank_documents(
        question,
        candidates,
        top_n=RERANK_TOP_N,
    )

    rerank_rank = find_expected_rank(
        documents,
        case,
    )

    # 4. 诊断失败发生在哪一层
    if candidate_rank is None:
        diagnosis = "Recall Failure"
    elif rerank_rank is None:
        diagnosis = "Reranking Failure"
    else:
        diagnosis = "Success"

    hit = rerank_rank is not None

    if hit:
        transformed_rerank_hits += 1

    print(f"\n===== Transform + Rerank Case {i + 1} =====")
    print("Original Question:", question)
    print("Transformed Query:", transformed_query)
    print("Expected:", case["expected_header_2"])
    print("Candidate Rank:", candidate_rank)
    print("Rerank Rank:", rerank_rank)
    print("Diagnosis:", diagnosis)
    print("Hit:", hit)

transformed_rerank_hit_rate = (
    transformed_rerank_hits / len(eval_cases)
)

print("\n===== Transform + Reranker Result =====")
print(
    f"Transform + Vector Top {RERANK_CANDIDATE_K} "
    f"+ Reranker Hit@{RERANK_TOP_N}: "
    f"{transformed_rerank_hits}/{len(eval_cases)} "
    f"= {transformed_rerank_hit_rate:.1%}"
)
