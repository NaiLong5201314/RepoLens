
#//LangGraph最小工作流//

# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END

# class State(TypedDict):
#     question: str
#     answer: str

# def process_question(state: State):
#     question = state["question"]

#     return {
#         "answer": f"I received your question: {question}"
#     }

# builder=StateGraph(State)

# builder.add_node("process_question", process_question)

# builder.add_edge(START,"process_question")
# builder.add_edge("process_question",END)

# graph= builder.compile()

# result=graph.invoke({
#     "question": "How does RepLens work?",
#     "answer": ""
# })

# print(result["answer"])

#//自动分类并选择路线Graph//
# from typing import Literal
# from typing_extensions import TypedDict
# from langgraph.graph import StateGraph, START, END

# class State(TypedDict):
#     question: str
#     question_type: str
#     answer: str

# def classify_question(state: State):
#     question= state["question"].lower()

#     code_keywords= [
#         "python",
#         "code",
#         "function",
#         "github",
#         "repository",
#         "langgraph"
#     ]

#     if any(keyword in question for keyword in code_keywords):
#         return{"question_type": "code"}

#     return {"question_type": "general"}

# def route_question(
#         state: State
# )-> Literal["code_answer","general_answer"]:

#     if state["question_type"] == "code":
#         return "code_answer"
#     return "general_answer"

# def code_answer(state: State):
#     return {
#         "answer": (
#             f"This is a code-related question:"
#             f"{state['question']}"
#         )
#     }

# def general_answer(state: State):
#     return {
#         "answer": (
#             f"This is a general question:"
#             f"{state['question']}"
#         )
#     }

# builder = StateGraph(State)

# builder.add_node("classify_question", classify_question)

# builder.add_node("code_answer",code_answer)
# builder.add_node("general_answer", general_answer)

# builder.add_edge(START, "classify_question")

# builder.add_conditional_edges(
#     "classify_question",
#     route_question
# )
# builder.add_edge("code_answer",END)
# builder.add_edge("general_answer", END)

# graph= builder.compile()

# questions = [
#     "How does a Python function work?",
#     "What is the weather today?"
# ]

# for question in questions:
#     result = graph.invoke({
#         "question": question,
#         "question_type": "",
#         "answer": ""
#     })

#     print(f"Question: {question}")
#     print(f"Type: {result['question_type']}")
#     print(f"Answer: {result['answer']}")
#     print()

#接入LLM去判断问题类型
from langgraph.checkpoint.memory import InMemorySaver

import re

import hashlib

import subprocess

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain.tools import tool

from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class State(MessagesState):
    question: str
    question_type: str
    classification_reason: str
    answer: str

class QuestionClassification(BaseModel):
    question_type: Literal['code',"general"] = Field(
        description=(
            "Use 'code' when answering requires inspecting the "
            "current project, repository, files, functions, classes, "
            "configuration, implementation, or architecture. "
            "Use 'general' only when project files are unnecessary."
        )
    )

    reason: str = Field(
            description="A short reason for the classification."
        )

APP_ROOT = Path(__file__).resolve().parent
REPOSITORIES_DIRECTORY = APP_ROOT/".repolens_repositories"
CACHE_DIRECTORY = APP_ROOT/".repolens_cache"

PROJECT_ROOT = APP_ROOT
INDEX_DIRECTORY = CACHE_DIRECTORY / "default"
VECTOR_INDEX_PATH = INDEX_DIRECTORY / "vectors.json"
FINGERPRINT_PATH = INDEX_DIRECTORY / "fingerprint.txt"

IGNORED_DIRECTORIES = {
    ".venv",
    ".git",
    "__pycache__",
    "node_modules",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
    ".repolens_index",
    ".repolens_cache",
    ".repolens_repositories",
}

ALLOWED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".toml",
    ".yaml",
    ".yml"
}

@tool
def list_project_files() -> str:
     """List the readable source files in the current project."""

     print("[Tool] Listing project files")

     files = []

     for path in PROJECT_ROOT.rglob("*"):
         if not path.is_file():
             continue

         relative_path = path.relative_to(PROJECT_ROOT)

         if any(
             part in IGNORED_DIRECTORIES
             for part in relative_path.parts
         ):
             continue

         if path.name.startswith(".env"):
             continue

         if path.suffix not in ALLOWED_EXTENSIONS:
             continue

         files.append(str(relative_path))

     if not files:
         return "No readable source files were found."

     return "\n".join(sorted(files)[:200])

@tool
def read_project_file(
    relative_path: str,
    start_line: int = 1,
    end_line: int = 200,
) -> str:
    """
    Read a specific line range from a project file.

    Search for the relevant symbol first, then use its line number
    to read the surrounding code. Do not read the same range twice.
    """
    print(
        f"[Tool] Reading file: {relative_path}, "
        f"lines {start_line}-{end_line}"
    )

    file_path = (
        PROJECT_ROOT / relative_path
    ).resolve()

    project_root = PROJECT_ROOT.resolve()

    if (
        file_path != project_root
        and project_root not in file_path.parents
    ):
        return "Error: File is outside the project directory."

    if not file_path.exists():
        return f"Error: File not found: {relative_path}"

    if not file_path.is_file():
        return f"Error: Not a file: {relative_path}"

    content = file_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    lines = content.splitlines()

    start_line = max(1, start_line)
    end_line = min(
        len(lines),
        end_line,
        start_line + 199,
    )

    if start_line > len(lines):
        return (
            f"Error: start_line {start_line} exceeds "
            f"file length {len(lines)}."
        )

    selected_lines = lines[
        start_line - 1:end_line
    ]

    numbered_content = "\n".join(
        f"{line_number:04d}: {line}"
        for line_number, line in enumerate(
            selected_lines,
            start=start_line,
        )
    )

    return (
        f"File: {relative_path}\n"
        f"Lines: {start_line}-{end_line}\n\n"
        f"{numbered_content}"
    )

def get_python_files() -> list[Path]:
    files = []

    for path in PROJECT_ROOT.rglob("*.py"):
        relative_path = path.relative_to(PROJECT_ROOT)

        if any (
            part in IGNORED_DIRECTORIES
            for part in relative_path.parts
        ):
            continue

        files.append(path)

    return sorted(files)

def create_code_fingerprint(files: list[Path]) -> str:
    """Create a hash representing the current codebase."""

    digest = hashlib.sha256()

    for path in files:
        relative_path = path.relative_to(PROJECT_ROOT)

        digest.update(str(relative_path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


@lru_cache(maxsize=1)
def get_code_retriever():
    python_files = get_python_files()

    if not python_files:
        raise ValueError("No Python files were found.")

    current_fingerprint = create_code_fingerprint(
        python_files
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    
    if (
        VECTOR_INDEX_PATH.exists()
        and FINGERPRINT_PATH.exists()
        and FINGERPRINT_PATH.read_text(
            encoding="utf-8"
        ) == current_fingerprint
    ):
        print("[RAG] Loading saved vector index...")

        vector_store = InMemoryVectorStore.load(
            str(VECTOR_INDEX_PATH),
            embeddings
        )

        return vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

    print("[RAG] Code changed. Building new vector index...")

    source_documents = []

    for path in python_files:
        relative_path = path.relative_to(PROJECT_ROOT)

        content = path.read_text(
            encoding="utf-8",
            errors="replace"
        )

        source_documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(relative_path)
                }
            )
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
        separators=[
            "\nclass ",
            "\ndef ",
            "\n\n",
            "\n",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(
        source_documents
    )

    source_contents = {
        document.metadata["source"]:
        document.page_content
        for document in source_documents
    }

    for chunk in chunks:
        source = chunk.metadata["source"]
        start_index = chunk.metadata.get(
            "start_index",
            0
        )

        chunk.metadata["start_line"] = (
            source_contents[source][:start_index]
            .count("\n") + 1
        )

    vector_store = InMemoryVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    
    vector_store.dump(
        str(VECTOR_INDEX_PATH)
    )

    FINGERPRINT_PATH.write_text(
        current_fingerprint,
        encoding="utf-8"
    )

    print(
        f"[RAG] Indexed and saved "
        f"{len(chunks)} code chunks."
    )

    return vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

@tool
def semantic_code_search(query: str) -> str:
    """Search the codebase for code that is semantically related to a question."""

    print(f"[Tool] Semantic code search: {query}")

    retriever = get_code_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return "No relevant code was found."

    results=[ ]

    for document in documents:
        source = document.metadata.get("source", "unknown")
        start_line = document.metadata.get("start_line", "?")

        if isinstance(start_line,int):
            end_line = (
                start_line
                + document.page_content.count("\n")
            )

            source_header = (
                f"Source: {source},"
                f"lines {start_line}--{end_line}"
            )
        else:
            source_header = f"Source: {source}"

        results.append(
            f"{source_header}\n"
            f"{document.page_content}"
        )
    return "\n\n---\n\n".join(results)

KEYWORD_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "this",
    "that",
    "where",
    "what",
    "which",
    "how",
    "does",
    "do",
    "explain",
    "project"
}

@tool
def keyword_code_search(query: str) -> str:
    """Search source code using exact words, identifiers, or error names."""

    print(f"[Tool] Keyword code search: {query}")

    raw_keywords = re.findall(
        r"[A-Za-z_][A-Za-z0-9_]*",
        query.lower()
    )

    keywords = [
        keyword
        for keyword in raw_keywords
        if len(keyword) > 2
        and keyword not in KEYWORD_STOP_WORDS
    ]

    if not keywords:
        return "No useful keywords were found in the query."

    matches = []

    for path in get_python_files():
        relative_path = path.relative_to(PROJECT_ROOT)

        lines = path.read_text(
            encoding="utf-8",
            errors="replace"
        ).splitlines()

        for index, line in enumerate(lines):
            lower_line = line.lower()

            score = sum(
                keyword in lower_line
                for keyword in keywords
            )

            if score == 0:
                continue

            start = max(0, index - 2)
            end = min(len(lines), index + 3)

            snippet = "\n".join(
                f"{line_number + 1:04d}: "
                f"{lines[line_number]}"
                for line_number in range(start, end)
            )

            matches.append(
                {
                    "score": score,
                    "source": str(relative_path),
                    "line": index + 1,
                    "snippet": snippet,
                }
            )

    if not matches:
        return "No exact keyword matches were found."

    matches.sort(
        key=lambda match: (
            -match["score"],
            match["source"],
            match["line"],
        )
    )

    formatted_results = []

    for match in matches[:8]:
        formatted_results.append(
            f"Source: {match['source']}, "
            f"around line {match['line']}\n"
            f"{match['snippet']}"
        )

    return "\n\n---\n\n".join(formatted_results)




model = init_chat_model(
    "openai:gpt-5-nano"
)


classification_model = model.with_structured_output(
    QuestionClassification
)

tools = [
    semantic_code_search,
    keyword_code_search,
    list_project_files,
    read_project_file
]

code_model = model.bind_tools(tools)

CODEBASE_MARKERS = [
    "this project",
    "this codebase",
    "this repository",
    "in the project",
    "in the codebase",
    "in the repository",
    "which file",
    "what file"
]

CODE_FOLLOWUP_PATTERN = re.compile(
    r"\b(it|this|that|they|them|its)\b"
    r"|\b(the method|the function|the class|the file)\b"
    r"|what happens next"
    r"|before that"
    r"|after that"
)

def get_recent_conversation(state: State) -> str:
     """Return recent user and assistant messages."""

     conversation = []

     for message in state["messages"]:
         if isinstance(message,HumanMessage):
             role = "User"
         elif isinstance(message,AIMessage):
             role = "Assistant"
         else:
             continue

         content = message.content

         if not isinstance(content,str):
             continue

         if not content.strip():
             continue

         conversation.append(
             f"{role}: {content}"
         )

         return "\n".join(conversation[-6:])
         


def classify_question(state: State):
    question = state["question"]
    lower_question = question.lower()
    messages = state.get("messages", [])
    conversation_context = get_recent_conversation(state)

    recent_previous_messages= messages[-10:-1]

    has_previous_code_context = any(
        isinstance(message, ToolMessage)
        for message in recent_previous_messages
    )

    if(
        has_previous_code_context
        and CODE_FOLLOWUP_PATTERN.search(lower_question)
    ):
        return{
            "question_type": "code",
            "classification_reason": (
                "The question explicitly refers "
                "to the current project."
            ),
        }

    if any(
        maker in lower_question
        for maker in CODEBASE_MARKERS
    ):
        return {
            "question_type":"code",
            "classification_reason":(
                "The question explicitly refers to the current project."
            )
        }
    prompt = f"""

If the recent conversation discussed project code and the
current question uses words such as "it", "this", "that",
"the method", or "the class", classify it as "code".

A follow-up question about previously inspected code is still
a code question, even if it could also be answered generally.

Classify whether answering the question requires inspecting
the current project's files.

Choose "code" if the answer depends on this project's files,
functions, classes, modules, configuration, implementation,
architecture, or file locations.

Choose "general" if the question can be answered without
reading the current project.

Examples:

Question: Where is user authentication implemented?
Classification: code

Question: How does the database class work?
Classification: code

Question: Where in this project is the user question classified?
Classification: code

Question: What is a Python class?
Classification: general

Question: Hello, how are you?
Classification: general

Question: What should I eat for dinner?
Classification: general

When uncertain, choose "code"

Recent conversation:
{conversation_context}

Current question:
{question}

"""

    result = classification_model.invoke(prompt)

    return {
        "question_type": result.question_type,
        "classification_reason": result.reason
    }

def route_question(
    state: State
) -> Literal["code", "general"]:

    return state["question_type"]

SOURCE_PATTERN = re.compile(
    r"Source:\s*(?P<path>[^\n,]+)"
    r"(?:,\s*(?P<location>"
    r"(?:around line|lines)\s+"
    r"[0-9?]+(?:-[0-9?]+)?"
    r"))?"
)

def collect_sources(messages) -> list[str]:
    """Collect sources from tool results."""

    sources = []
    seen = set()

    for message in messages:
        if not isinstance(message, ToolMessage):
            continue

        content = message.content

        if not isinstance(content,str):
            content = str(content)

        for match in SOURCE_PATTERN.finditer(content):
            path = match.group("path").strip()
            location = match.group("location")

            if location:
                source = f"{path} ({location})"
            else:
                source = path

            if source not in seen:
                seen.add(source)
                sources.append(source)
    return sources

CODE_AGENT_PROMPT = """
You are RepoLens, an assistant that answers questions about
the current codebase.

Available search strategies:

1. Use keyword_code_search when the question contains an
exact function name, class name, variable, file name, error,
or other identifier.

2. Use semantic_code_search when the question describes a
concept, behavior, workflow, or architecture without knowing
the exact identifier.

3. Use both search tools when one search does not provide
enough evidence.

4. Use read_project_file when complete file context is needed.

5. Use list_project_files when project structure is needed.

Do not guess about code that you have not inspected.

The final answer must:
- answer the question directly;
- mention relevant file paths;
- cite approximate line numbers;
- use the format [source: path, lines X-Y];
- only cite files and line numbers returned by tools;
- never invent a source;
- clearly say when the evidence is insufficient;
- call another tool if the first search is not enough.

Rules:
- Do not read the same file more than once for one question.
- Use no more than 5 tool calls.
- Answer immediately once enough evidence is found.
- If the requested feature cannot be found after searching, clearly state that it is not present.
"""

def code_agent(state: State):
    response = code_model.invoke(
        [SystemMessage(content=CODE_AGENT_PROMPT),
         *state["messages"]
         ]
    )

    update = {
        "messages": [response]
    }

    if not response.tool_calls:
        update["answer"] = response.content
    return update

def general_answer(state: State):
    return{
        "answer": (
            "This question does not require "
            "codebase retrieval."
        )
    }

GITHUB_URL_PATTERN = re.compile(
    r"^https://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repository>[A-Za-z0-9_.-]+)/?$"
)

def configure_project_root(project_path: Path) -> None:
    """
    Switch RepoLens to another repository and give it
    a separate vector index.
    """

    global PROJECT_ROOT
    global INDEX_DIRECTORY
    global VECTOR_INDEX_PATH
    global FINGERPRINT_PATH

    PROJECT_ROOT = project_path.resolve()

    project_id = hashlib.sha256(
        str(PROJECT_ROOT).encode("utf-8")
    ).hexdigest()[:12]

    INDEX_DIRECTORY = CACHE_DIRECTORY / project_id
    VECTOR_INDEX_PATH = INDEX_DIRECTORY / "vectors.json"
    FINGERPRINT_PATH = INDEX_DIRECTORY / "fingerprint.txt"

    INDEX_DIRECTORY.mkdir(parents=True, exist_ok=True)

    get_code_retriever.cache_clear()

def prepare_project(source: str) -> Path:
    """
    Accept either:
    1. An existing local directory
    2. A public GitHub repository URL
    """

    source = source.strip()
    local_path = Path(source).expanduser()

    
    if local_path.exists():
        if not local_path.is_dir():
            raise ValueError(
                "The local path must be a directory."
            )

        project_path = local_path.resolve()
        configure_project_root(project_path)

        print(
            f"[Project] Loaded local repository: "
            f"{project_path}"
        )

        return project_path

    
    match = GITHUB_URL_PATTERN.fullmatch(source)

    if not match:
        raise ValueError(
            "Please enter an existing local directory "
            "or a public GitHub URL.\n"
            "Example: https://github.com/owner/repository"
        )

    owner = match.group("owner")
    repository = (
        match.group("repository")
        .removesuffix(".git")
    )

    clone_url = (
        f"https://github.com/"
        f"{owner}/{repository}.git"
    )

    REPOSITORIES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination = (
        REPOSITORIES_DIRECTORY
        / f"{owner}__{repository}"
    )

    if destination.exists():
        if not (destination / ".git").exists():
            raise RuntimeError(
                "The repository directory is incomplete: "
                f"{destination}"
            )

        print(
            f"[Project] Using saved repository: "
            f"{owner}/{repository}"
        )

    else:
        print(
            f"[Project] Cloning "
            f"{owner}/{repository}..."
        )

        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    clone_url,
                    str(destination),
                ],
                check=True,
            )

        except FileNotFoundError as error:
            raise RuntimeError(
                "Git is not installed or "
                "is not available in PATH."
            ) from error

        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                "Failed to clone the GitHub repository. "
                "Check the URL and confirm that it is public."
            ) from error

    
    configure_project_root(destination)

    print(
        f"[Project] Active repository: "
        f"{destination}"
    )

    return destination

             



builder = StateGraph(State)

builder.add_node("classify_question", classify_question)
builder.add_node("code_agent", code_agent)
builder.add_node("tools", ToolNode(tools))
builder.add_node("general_answer", general_answer)

builder.add_edge(START, "classify_question")
builder.add_conditional_edges(
    "classify_question",
    route_question,
    {
        "code": "code_agent",
        "general": "general_answer"
    }
)

builder.add_conditional_edges("code_agent", tools_condition)
builder.add_edge("tools", "code_agent")
builder.add_edge("general_answer", END)

checkpointer = InMemorySaver()

graph = builder.compile(
    checkpointer=checkpointer
)

if __name__ =="__main__":
    source = input(
        "Local repository path or public GitHub URL:"
    ).strip()

    try:
        prepare_project(source)
        print(f"[Debug] Active root: {PROJECT_ROOT}")

        print(
            "[Debug] Python files:",
            [
                str(path.relative_to(PROJECT_ROOT))
                for path in get_python_files()[:10]
            ]
        )
    except (ValueError,RuntimeError) as error:
        print(f"[Error] {error}")
        raise SystemExit(1)

    project_id = hashlib.sha256(
        str(PROJECT_ROOT).encode("utf-8")
    ).hexdigest()[:12]

    conversation_config = {
        "configurable" : {
            "thread_id": f"repolens-{project_id}"
        }
    }

    print("\nRepoLens is ready.")
    print("Enter 'exit' or 'quit' to stop." )

    while True:
        question = input("\nAsk RepoLens: ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

   


        result = graph.invoke({
            "question": question,
            "question_type": "",
            "classification_reason": "",
            "answer": "",
            "messages": [
                HumanMessage(content=question)
            ],
            
        },
        config = conversation_config,
                    
        )

        print("=" * 50)
        print(f"Question: {question}")
        print(f"Type: {result['question_type']}")
        print(f"Reason: {result['classification_reason']}")
        print(f"Answer: {result['answer']}")

        sources = collect_sources(result["messages"])

        if sources:
            print("\nSources inspected:")

            for source in sources[-8:]:
                print(f"-{source}")
        else:
            print("\nSources inspected: None")

