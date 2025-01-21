import streamlit as st
import requests
from streamlit_lottie import st_lottie
from PIL import Image

# --- HERO SECTION ---
col1, col2 = st.columns(2, gap="small", vertical_alignment="center")
with col1:
    st.image("./assets/me.jpg", width=230)

with col2:
    st.title("This is me", anchor=False)
    st.write(
        "Bored with Life."
    )
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    with left_column:
        st.header("What I do")
        st.write("##")
        st.write(
            """
            On my YouTube channel I am creating tutorials for people who:
            - are looking for a way to leverage the power of Python in their day-to-day work.
            - are struggling with repetitive tasks in Excel and are looking for a way to use Python and VBA.
            - want to learn Data Analysis & Data Science to perform meaningful and impactful analyses.
            - are working with Excel and found themselves thinking - "there has to be a better way."

            If this sounds interesting to you, consider subscribing and turning on the notifications, so you don’t miss any content.
            """
        )
        st.write("[YouTube Channel >](https://www.youtube.com/watch?v=xvFZjo5PgG0)")
    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")


# --- EXPERIENCE & QUALIFICATIONS ---
st.write("\n")
st.subheader("Experience & Qualifications", anchor=False)
st.write(
    """
    - Pro in Pubg
    - Strong hands-on experience and knowledge in Marvel rivals and Valorant
    - Good understanding of statistical principles and their respective applications
    - Excellent team-player and displaying a strong sense of initiative on tasks
    """
)

# --- SKILLS ---
st.write("\n")
st.subheader("Hard Skills", anchor=False)
st.write(
    """
    - Programming: Python (Scikit-learn, Pandas), SQL, VBA
    - Data Visualization: PowerBi, MS Excel, Plotly
    - Modeling: Logistic regression, linear regression, decision trees
    - Databases: Postgres, MongoDB, MySQL
    """
)