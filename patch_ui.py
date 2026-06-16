from pathlib import Path
p=Path("app/interface/streamlit_ui.py")
t=p.read_text()
old="""                llm_provider = st.selectbox("""
