import uuid
import streamlit as st
from langchain_core.messages import HumanMessage

import main as backend

st.set_page_config(
    page_title="RepoLens",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 RepoLens")
st.caption("A RAG-powered codebase understanding agent "
    "built with LangGraph.")

if "project_loaded" not in st.session_state:
    st.session_state.project_loaded = False

if "project_path" not in st.session_state:
    st.session_state.project_path = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Repository")

    repository_source = st.text_input(
         "Local path or public GitHub URL",
        placeholder=(
            "https://github.com/"
            "karpathy/micrograd"
        ),
    )

    load_repository = st. button(
        "Load repository",
        use_container_width=True,
        type="primary",
    )

    if load_repository:
        if not repository_source.strip():
            st.warning(
                 "Please enter a repository path or URL."
            )
        else:
            try:
                with st.spinner(
                     "Preparing repository..."
                ):
                    project_path = (
                        backend.prepare_project(
                            repository_source
                        )
                    )

                st.session_state.project_loaded = True
                st.session_state.project_path = str(
                    project_path
                )

                st.session_state.thread_id = str(
                    uuid.uuid4()
                )

                st.session_state.chat_history = []

                st.success("Repository loaded!")

            except (ValueError,RuntimeError) as error:
                st.error(str(error))

        if st.session_state.project_loaded:
            st.divider()

            st.success("Active repository")

            st.code(
                st.session_state.project_path,
                language=None,
            )

            if st.button(
                "Start new conversation",
                use_container_width=True,
            ):
                st.session_state.thread_id = str(
                      uuid.uuid4()
                )
                st.session_state.chat_history = []

                st.rerun()


if not st.session_state.project_loaded:
    st.info(
         "Enter a local repository path or a public "
        "GitHub URL in the sidebar to begin."
    )

    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            question_type = message.get(
                "question_type"
            )

            if question_type:
                st.caption(
                     f"Route: `{question_type}`"
                )

            sources = message.get("sources", [])

            if sources:
                with st.expander(
                     "Sources inspected"
                ):
                    for source in sources:
                        st.code(
                            source,
                            language=None,
                        )

question = st.chat_input(
     "Ask a question about this repository..."
)
if question:
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    conversation_config = {
        "configurable":{
            "thread_id" :(
                st.session_state.thread_id
            )
        }
    }

    try:
        with st.chat_message("assistant"):
            with st.spinner(
                "Inspecting the codebase..."
            ):
                result = backend.graph.invoke(
                    {
                        "question": question,
                        "question_type": "",
                        "classification_reason": "",
                        "answer": "",
                        "messages": [
                            HumanMessage(
                                content=question
                            )
                        ],
                    },
                    config=conversation_config,
                )
                answer = result.get(
                     "answer",
                    "No answer was generated.",
                )

                sources = backend.collect_sources(
                    result["messages"]
                )

                displayed_sources = sources[-8:]

                st.markdown(answer)

                st.caption(
                    f"Route: "
                    f"`{result['question_type']}`"
                )

                if displayed_sources:
                    with st.expander(
                         "Sources inspected"
                    ):
                        for source in displayed_sources:
                            st.code(
                                source,
                            language=None,
                            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "question_type": result[
                    "question_type"
                ],

                "sources": displayed_sources,
            }
        )   

    except Exception as error:
        st.error(
            f"RepoLens failed to answer: {error}"
        )

