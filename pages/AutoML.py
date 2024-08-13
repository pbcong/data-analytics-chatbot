import streamlit as st

if 'df' in st.session_state:
    st.dataframe(st.session_state['df'])
    target = 0
    with st.sidebar:
        problem = st.selectbox("Problem type:", ["Regression", "Classification"])
        target = st.selectbox("Target variable:", st.session_state['df'].columns)

    if st.button("test"):
        if problem == "Classification" and target!=0:
            from pycaret.classification import *
            with st.spinner(text="testing different models..."):
                setup(
                    data=st.session_state["df"],
                    target=target,
                    ignore_features=["id"],
                )
            
                best_model = compare_models()
                compare_df = pull()


        if problem == "Regression" and target!=0:
            from pycaret.regression import *
            with st.spinner(text="testing different models..."):
                setup(
                    data=st.session_state["df"],
                    target=target,
                    session_id=42,
                    ignore_features=["id"],
                    fold=5,
                    fold_strategy="stratifiedkfold",
                    log_experiment=True,
                    experiment_name="classification",
                )
            
                best_model = compare_models()
                compare_df = pull()

        st.dataframe(compare_df)

        st.write(best_model)
        save_model(best_model, 'best_model')

        with open('best_model.pkl', 'rb') as f:
            st.download_button("Download best model", f, "best_model.pkl")
else:
    st.write('### Please upload your dataset')


