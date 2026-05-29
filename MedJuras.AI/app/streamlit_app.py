import os, sys
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)
from interface.streamlit_ui import MedJurisUI
from interface.chat_assistant import ChatAssistant

if __name__ == '__main__':
    MedJurisUI(ChatAssistant()).run()
