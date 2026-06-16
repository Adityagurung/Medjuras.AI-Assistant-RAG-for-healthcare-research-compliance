from pathlib import Path
L=Path("app/interface/streamlit_ui.py").read_text().splitlines()
start=next(i for i,l in enumerate(L) if "Star rating" in l)
end=next(i for i,l in enumerate(L[start:], start) if l.strip().startswith("elif role"))
new=[
'                message_id = message.get("id")',
'                if message_id and not message.get("feedback_given"):',
'                    stars = st.select_slider(',
'                        "Rate this response (1-5 stars)",',
'                        options=[1, 2, 3, 4, 5],',
'                        value=5,',
'                        key=f"stars_{message_id}",',
'                    )',
'                    if st.button("Submit rating", key=f"rate_{message_id}"):',
'                        self.submit_star_feedback(message, stars, settings)',
'                elif message.get("feedback_given"):',
'                    st.caption("Rated " + str(message.get("feedback_stars", 0)) + " / 5 stars")',
]
L=L[:start]+new+L[end:]
Path("app/interface/streamlit_ui.py").write_text(chr(10).join(L)+chr(10))
print("stars")
