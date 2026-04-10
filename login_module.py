import streamlit as st

VALID_CREDENTIALS = {
    "user": "user@123"
}

def init_login_state():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "login_error" not in st.session_state:
        st.session_state.login_error = None


def is_logged_in():
    return st.session_state.get("logged_in", False)


def login_form():
    init_login_state()
    st.markdown("""
    <div style="max-width:520px; margin:auto; padding:2rem; background:#f8f9fc; border-radius:20px; box-shadow:0 10px 30px rgba(0,0,0,0.08);">
        <h1 style="text-align:center; color:#1E3A8A;">Welcome Back</h1>
        <p style="text-align:center; color:#4B5563;">Please log in with the username and password provided to access the Alzheimer's Detection app.</p>
    </div>
    """, unsafe_allow_html=True)

    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")

    if st.button("Login"):
        if username in VALID_CREDENTIALS and VALID_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.login_error = None
            st.experimental_rerun()
        else:
            st.session_state.login_error = "Invalid username or password."

    if st.session_state.login_error:
        st.error(st.session_state.login_error)

    st.caption("Use username: user and password: user@123")


def logout():
    st.session_state.logged_in = False
    st.experimental_rerun()
