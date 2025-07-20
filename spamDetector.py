import os
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set working directory to script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load the model and vectorizer
model = pickle.load(open('spam.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# App setup
st.set_page_config(page_title="Spam Detector", layout="wide")

def main():
    st.markdown(
        """
        <style>
            .main-title {
                font-size:36px !important;
                color:#4A90E2;
                font-weight:bold;
                text-align:center;
                padding-bottom: 10px;
            }
            .sub-title {
                font-size:20px;
                color:gray;
                text-align:center;
                padding-bottom: 30px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('<div class="main-title">📧 Email Spam Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Detects whether an email is Spam or Not Spam (Ham) using Machine Learning</div>', unsafe_allow_html=True)
    
    # Side-by-side layout
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("📝 Enter Email")
        user_input = st.text_area("Paste your email content below:", height=200)

        if st.button("🚀 Classify Email"):
            if user_input.strip() == "":
                st.warning("⚠️ Please enter a valid email.")
            else:
                data = [user_input]
                vec = cv.transform(data).toarray()
                result = model.predict(vec)[0]
                probs = model.predict_proba(vec)[0]

                # Display result
                if result == 0:
                    st.success("✅ This is NOT a Spam Email.")
                else:
                    st.error("🚫 This IS a Spam Email.")

                # Save to session for use in right column
                st.session_state['probs'] = probs

    with col2:
        st.subheader("📊 Spam Prediction Confidence")
        if 'probs' in st.session_state:
            probs = st.session_state['probs']
            labels = ['Ham (Not Spam)', 'Spam']
            colors = ['#66bb6a', '#ef5350']

            fig, ax = plt.subplots(figsize=(5, 3))
            bars = ax.bar(labels, probs * 100, color=colors)
            ax.set_ylabel('Probability (%)')
            ax.set_ylim(0, 100)
            ax.set_title('Prediction Probability')
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, yval + 2, f'{yval:.2f}%', ha='center', fontsize=10)

            st.pyplot(fig)
        else:
            st.info("Prediction chart will appear here after classification.")

if __name__ == "__main__":
    main()
