import os

from config.streamlit_config import apply_custom_styling
from interface.chat_assistant import ChatAssistant
from interface.streamlit_ui import MedicalRAG_UI


def main():
    apply_custom_styling()

    assistant = ChatAssistant()
    ui = MedicalRAG_UI(assistant)
    ui.render()


if __name__ == "__main__":
    main()
