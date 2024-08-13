import streamlit as st
from pycaret.regression import *

if 'df' in st.session_state:
    with st.sidebar:
        problem = st.selectbox("Problem type:", ["Regression", "Classification"])
        target = st.selectbox("Target variable:", st.session_state['df'].columns)

    if problem == "Regression":
        s = ClassificationExperiment()
        s.setup(
            data=st.session_state["df"],
            target=target,
            session_id=42,
            preprocess=True,
            ignore_low_variance=True,
            remove_multicollinearity=False,
            ignore_features=["id"],
            fold=5,
            fold_strategy="stratifiedkfold",
            fold_shuffle=True,
            log_experiment=True,
            experiment_name="classification",
        )

        best_model = s.compare_models()

else:
    st.write('### Please upload your dataset')


