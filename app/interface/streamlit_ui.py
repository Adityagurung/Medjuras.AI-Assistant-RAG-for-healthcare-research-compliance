import base64
import time
import uuid
from pathlib import Path
from typing import Dict, List

import plotly.express as px
import streamlit as st
from monitoring.conversation_log import primary_tool_name, store_conversation_turn, update_star_rating
from monitoring.feedback import generate_message_id
from streamlit_chat import message
from streamlit_option_menu import option_menu


def _brand_icon_path():
    app_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[2]
    a = app_root / "images" / "medjuras_ai_icon.png"
    b = repo_root / "images" / "medjuras_ai_icon.png"
    c = Path("/app/app/images/medjuras_ai_icon.png")
    d = Path("/app/images/medjuras_ai_icon.png")
    for path in (a, b, c, d):
        if path.is_file():
            return path
    return a


def _brand_icon_data_uri():
    icon = _brand_icon_path()
    if not icon.is_file():
        return None
    encoded = base64.b64encode(icon.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


class MedicalRAG_UI:
    def __init__(self, chat_assistant):
        self.assistant = chat_assistant
        self.setup_page_config()
        self.init_session_state()
        self.user_to_detail_map = {
            "Medical Researcher": 2,  # "Technical"
            "Healthcare Provider": 1,  # "Detailed"
            "Patient": 0,  # "Simple"
        }

    def setup_page_config(self):
        icon = _brand_icon_path()
        page_icon = str(icon) if icon.is_file() else "🧬"
        st.set_page_config(
            page_title="Medjuras.AI",
            page_icon=page_icon,
            layout="wide",
            initial_sidebar_state="expanded",
        )

    def init_session_state(self):
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        if "show_feedback_thanks" not in st.session_state:
            st.session_state.show_feedback_thanks = False

    def _inject_ui_styles(self):
        if st.session_state.get("_ui_styles"):
            return
        st.markdown(
            """
            <style>
            .msg-metrics {
                font-size: 0.72rem;
                color: #6b7280;
                text-align: right;
                line-height: 1.5;
                padding-top: 0.25rem;
            }
            .star-done {
                color: #f5c518;
                font-size: 1.15rem;
                letter-spacing: 0.06rem;
            }
            .brand-title-main {
                font-size: 2.25rem;
                font-weight: 700;
                margin: 0;
                padding: 0;
                line-height: 1.1;
                color: rgb(49, 51, 63);
            }
            .brand-title-sidebar {
                font-size: 1.25rem;
                font-weight: 600;
                margin: 0;
                padding: 0;
                line-height: 1.15;
                color: rgb(49, 51, 63);
            }
            .brand-caption {
                font-size: 0.875rem;
                color: rgb(128, 132, 149);
                margin: 0.15rem 0 0 0;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.session_state._ui_styles = True

    def _render_brand_row(self, title: str, *, image_width: int = 40, subtitle: str | None = None, main: bool = False):
        data_uri = _brand_icon_data_uri()
        icon_px = image_width
        title_size = "2.25rem" if main else "1.25rem"
        title_weight = "700" if main else "600"
        img_html = (
            f"<img src='{data_uri}' alt='Medjuras logo' "
            f"style='width:{icon_px}px;height:{icon_px}px;object-fit:contain;flex-shrink:0;display:block;' />"
            if data_uri
            else ""
        )
        caption_html = (
            f"<div style='font-size:0.875rem;color:#808495;margin:0.15rem 0 0 0;'>{subtitle}</div>"
            if subtitle
            else ""
        )
        st.markdown(
            "<div style='display:flex;align-items:center;gap:0.5rem;margin:0 0 0.75rem 0;'>"
            + img_html
            + "<div style='display:flex;flex-direction:column;justify-content:center;min-width:0;'>"
            + f"<div style='font-size:{title_size};font-weight:{title_weight};margin:0;padding:0;line-height:1.1;color:#31333f;'>{title}</div>"
            + caption_html
            + "</div></div>",
            unsafe_allow_html=True,
        )

    def render_sidebar(self):
        with st.sidebar:
            self._render_brand_row("Medical RAG", image_width=40)

            # User type selection
            user_type = st.selectbox(
                "I am a:",
                ["Medical Researcher", "Healthcare Provider", "Patient"],
                index=1,  # Default to "Healthcare Provider" (index 1)
            )

            if st.button("New Conversation"):
                self.reset_conversation()

            # Settings
            with st.expander("⚙️ Settings"):
                llm_provider = st.selectbox(
                    "LLM Provider:",
                    ["openai", "anthropic"],
                    index=0,
                    format_func=lambda x: {
                        "openai": "ChatGPT GPT-OSS-120B",
                        "anthropic": "Claude Sonnet 4.5",
                    }[x],
                )
                response_detail = st.selectbox(
                    "Response Detail:",
                    ["Simple", "Detailed", "Technical"],
                    index=self.user_to_detail_map.get(
                        user_type, 1
                    ),  # Auto-set response detail based on user type but Default to "Detailed"
                )
                show_sources = st.checkbox("Show Sources", value=False)
        st.sidebar.text("© Aditya Gurung")

        # Returns settings dict for use in chat class
        return {
            "user_type": user_type,
            "response_detail": response_detail,
            "show_sources": show_sources,
            "llm_provider": llm_provider,
        }

    def _render_message_metrics(self, metrics):
        if not metrics:
            return
        rt = metrics.get("response_time_sec", 0)
        tokens = metrics.get("total_tokens", 0)
        model = metrics.get("model", "unknown")
        cost = round(float(metrics.get("estimated_cost_usd", 0)), 4)
        html = (
            "<div class='msg-metrics'>"
            + f"Response time: {rt}s &nbsp;·&nbsp; "
            + f"Tokens: {tokens} &nbsp;·&nbsp; "
            + f"Model: {model} &nbsp;·&nbsp; "
            + f"Est. cost: ${cost}"
            + "</div>"
        )
        st.markdown(html, unsafe_allow_html=True)

    def _render_star_feedback(self, message, settings):
        message_id = message.get("id")
        if not message_id:
            return
        if message.get("feedback_given"):
            stars = int(message.get("feedback_stars", 0))
            filled = "★" * stars
            empty = "☆" * max(0, 5 - stars)
            st.markdown('<span class="star-done">' + filled + empty + '</span>', unsafe_allow_html=True)
            return
        rating = st.feedback("stars", key=f"stars_{message_id}")
        if rating is not None:
            self.submit_star_feedback(message, rating + 1, settings)

    def _render_message_footer(self, message, settings):
        left, right = st.columns([2, 3])
        with left:
            self._render_star_feedback(message, settings)
        with right:
            self._render_message_metrics(message.get("metrics"))

    def render_chat_interface(self, settings):
        self._inject_ui_styles()

        if st.session_state.get("show_feedback_thanks"):
            st.toast("Thank you for your feedback!", icon="⭐")
            st.session_state.show_feedback_thanks = False

        self._render_brand_row(
            "Medjuras.AI",
            image_width=48,
            subtitle="Ask medical questions with confident sources",
            main=True,
        )

        # Display chat history
        for msg in st.session_state.chat_messages:
            self.render_message(msg, settings)

        # Promp User input
        if prompt := st.chat_input("Ask a medical question..."):
            self.handle_user_input(prompt, settings)

    def render_message(self, message, settings):
        role = message["role"]
        content = message["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)

        elif role == "assistant":
            with st.chat_message("assistant"):
                st.markdown(content)

                # Show citations if available
                citations = message.get("citations")

                # Only show if the user enabled it AND there are citations
                if settings.get("show_sources") and citations:
                    # normalize to list
                    if not isinstance(citations, list):
                        citations = [citations]

                    with st.expander(
                        f"📚 Citations ({len(citations)})", expanded=False
                    ):
                        for c in citations:
                            if isinstance(c, dict):
                                st.json(c)  # nicer for dicts
                            else:
                                st.write(c)  # fall back to plain text

                if message.get("id"):
                    self._render_message_footer(message, settings)
        elif role == "tool_call":
            with st.chat_message("assistant"):
                with st.expander(f"🔧 {message['tool_name']}"):
                    st.json(message["output"])

    def handle_user_input(self, prompt, settings):
        # Add user message to chat history above
        user_msg = {"role": "user", "content": prompt}
        st.session_state.chat_messages.append(user_msg)

        # user_type = settings.get("user_type", "Healthcare Provider")
        # response_detail = settings.get("response_detail")
        # user_type = st.session_state.get('user_type', 'Healthcare Provider')
        # override_detail = st.session_state.get('response_detail')

        # Get assistant response with current settings
        with st.spinner("Thinking..."):
            response = self.assistant.process_message(
                question=prompt, settings=settings
            )

        # Add assistant response with unique ID for feedback
        assistant_msg = {
            "role": "assistant",
            "content": response["answer"],
            "id": generate_message_id(),
            "user_query": prompt,
            "feedback_given": False,
            "used_tools": len(response.get("used_tools", [])) > 0,
            "metrics": dict(
                response_time_sec=response.get("response_time_sec", 0.0),
                total_tokens=response.get("tokens", 0),
                model=response.get("model", "unknown"),
                estimated_cost_usd=response.get("estimated_cost_usd", 0.0),
            ),
        }
        cits = response.get("citations") or []
        if isinstance(cits, (list, tuple)) and cits:
            assistant_msg["citations"] = list(cits)

        tool_used = primary_tool_name(response.get("used_tools", []))
        assistant_msg["tool_used"] = tool_used
        store_conversation_turn(
            conversation_id=st.session_state.conversation_id,
            message_id=assistant_msg["id"],
            user_query=prompt,
            assistant_response=response["answer"],
            user_type=settings.get("user_type"),
            response_detail=settings.get("response_detail"),
            tool_used=tool_used,
            session_id=st.session_state.get("session_id", "streamlit"),
            model=response.get("model"),
            prompt_tokens=response.get("prompt_tokens", 0),
            completion_tokens=response.get("completion_tokens", 0),
            total_tokens=response.get("tokens", 0),
            response_time_sec=response.get("response_time_sec", 0.0),
            estimated_cost_usd=response.get("estimated_cost_usd", 0.0),
            llm_provider=settings.get("llm_provider"),
        )
        st.session_state.chat_messages.append(assistant_msg)
        st.rerun()


    def submit_star_feedback(self, message, stars, settings):
        ok = update_star_rating(
            message_id=message["id"],
            rating=stars,
            user_type=settings.get("user_type"),
            response_detail=settings.get("response_detail"),
            tool_used=message.get("tool_used", "none"),
            session_id=st.session_state.get("session_id", "streamlit"),
        )
        if ok:
            for msg in st.session_state.chat_messages:
                if msg.get("id") == message["id"]:
                    msg["feedback_given"] = True
                    msg["feedback_stars"] = stars
                    break
            st.session_state.show_feedback_thanks = True
            st.rerun()
        else:
            st.error("Could not save rating. Check Postgres connection.")



    def reset_conversation(self):
        st.session_state.chat_messages = []
        st.session_state.conversation_id = str(uuid.uuid4())
        st.rerun()

    def render(self):
        self._inject_ui_styles()
        settings = self.render_sidebar()
        self.render_chat_interface(settings)

        # selected = option_menu(
        #    menu_title="Medical RAG",
        #    options=["Chat", "Analytics", "Settings"],
        #    icons=["chat-dots", "graph-up", "gear"],
        #    menu_icon="hospital",
        #    default_index=0,
        #    styles={
        #        "container": {"padding": "0!important", "background-color": "#fafafa"},
        #        "icon": {"color": "orange", "font-size": "25px"},
        #        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
        #        "nav-link-selected": {"background-color": "#02ab21"},
        #    }
        # )

        # Beautiful chat bubbles
        # message("Hello! I'm your medical assistant", key="assistant1")
        # message("What's the difference between Type 1 and Type 2 diabetes?", is_user=True, key="user1")
