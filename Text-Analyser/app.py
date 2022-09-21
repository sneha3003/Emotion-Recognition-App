import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import pickle
import joblib

model = joblib.load(open('emo_cls.pkl', 'rb'))


def predict_emotions(text):
    results = model.predict([text])
    return results[0]


def get_prediction_probab(text):
    results = model.predict_proba([text])
    return results


def main():
    st.title('Emotion Recognition App')

    with st.form(key='emo_recog'):
        raw_text = st.text_area('Type Here')
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        col1, col2 = st.columns(2)

        pred = predict_emotions(raw_text)
        prob = get_prediction_probab(raw_text)

    else:
        col1, col2 = st.columns(2)

        pred = predict_emotions(raw_text)
        prob = get_prediction_probab(raw_text)

    with col1:
        st.success('Original Text')
        st.write(raw_text)
        st.success("Prediction")
        st.write(pred)
        st.write("Confidence:{}".format(np.max(prob)))

    with col2:
        st.success('Prediction Probability')
        prob_df = pd.DataFrame(prob, columns=model.classes_)
        prob_df_clean = prob_df.T.reset_index()
        prob_df_clean.columns = ["Emotions", "Probability"]

        fig = alt.Chart(prob_df_clean).mark_bar().encode(x='Emotions', y='Probability', color='Emotions')
        st.altair_chart(fig, use_container_width=True)


if __name__ == '__main__':
    main()