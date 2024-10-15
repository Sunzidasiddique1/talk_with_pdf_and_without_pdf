import logging
from typing import Dict, Any
import PyPDF2  # To handle PDF files

import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama

# Constants
PAGE_TITLE = "Llama 3.2 Chat"
PAGE_ICON = "🦙"
SYSTEM_PROMPT = "You are a friendly AI chatbot having a conversation with a human."
DEFAULT_MODEL = "llama3.2:1b"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_session_state() -> None:
    defaults: Dict[str, Any] = {
        "model": DEFAULT_MODEL,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "total_duration": 0,
        "num_predict": 2048,
        "seed": 1,
        "temperature": 0.5,
        "top_p": 0.9,
        "uploaded_text": "",  # Store uploaded text
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def create_sidebar() -> None:
    with st.sidebar:
        st.header("Inference Settings")
        st.session_state.system_prompt = st.text_area(
            label="System",
            value=SYSTEM_PROMPT,
            help="Sets the context in which to interact with the AI model. It typically includes rules, guidelines, or necessary information that help the model respond effectively.",
        )
        
        # Directly set the model without sliders
        st.session_state.model = "llama3.2:1b"
        st.session_state.seed = 1
        st.session_state.temperature = 0.5
        st.session_state.top_p = 0.9
        st.session_state.num_predict = 2048

        # PDF upload
        uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
        if uploaded_file is not None:
            # Read the PDF file
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            st.session_state.uploaded_text = text  # Store the text from the PDF

        # URL input
        url_input = st.text_input("Enter a URL (optional)", "")
        if url_input:
            # You may want to fetch and process the URL content here
            # For example, use requests or BeautifulSoup to scrape the content
            st.session_state.uploaded_text = f"Content from URL: {url_input}"  # Placeholder


def create_chat_model() -> ChatOllama:
    return ChatOllama(
        model=st.session_state.model,
        seed=st.session_state.seed,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        num_predict=st.session_state.num_predict,
    )


def create_chat_chain(chat_model: ChatOllama):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )
    return prompt | chat_model


def update_sidebar_stats(response: Any) -> None:
    total_duration = response.response_metadata["total_duration"] / 1e9
    st.session_state.total_duration = f"{total_duration:.2f} s"
    st.session_state.input_tokens = response.usage_metadata["input_tokens"]
    st.session_state.output_tokens = response.usage_metadata["output_tokens"]
    st.session_state.total_tokens = response.usage_metadata["total_tokens"]
    token_per_second = (
        response.response_metadata["eval_count"]
        / response.response_metadata["eval_duration"]
    ) * 1e9
    st.session_state.token_per_second = f"{token_per_second:.2f} tokens/s"


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown(
        """
        <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title(f"{PAGE_TITLE} {PAGE_ICON}")

    st.markdown("##### Chat")

    initialize_session_state()
    create_sidebar()

    chat_model = create_chat_model()
    chain = create_chat_chain(chat_model)

    msgs = StreamlitChatMessageHistory(key="special_app_key")
    if not msgs.messages:
        msgs.add_ai_message("How can I help you?")

    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input("Type your message here..."):
        st.chat_message("human").write(prompt)

        # Add uploaded text to prompt if it exists
        if st.session_state.uploaded_text:
            prompt = st.session_state.uploaded_text + "\n" + prompt
        
        with st.spinner("Thinking..."):
            config = {"configurable": {"session_id": "any"}}
            response = chain_with_history.invoke({"input": prompt}, config)
            logger.info({"input": prompt}, config)
            st.chat_message("ai").write(response.content)
            logger.info(response)
            update_sidebar_stats(response)

    if st.button("Clear Chat History"):
        msgs.clear()
        st.rerun()


if __name__ == "__main__":
    main()
