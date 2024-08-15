import streamlit as st

if 'df' in st.session_state:
    st.dataframe(st.session_state['df'].head(), use_container_width=True)
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
                    train_size=0.8,
                    session_id=42,
                    ignore_features=["id"],
                )
            
                best_model = compare_models()
                compare_df = pull()

        st.dataframe(compare_df, use_container_width=True)

        st.write(best_model)
        save_model(best_model, 'best_model')

        with open('best_model.pkl', 'rb') as f:
            st.download_button("Download best model", f, "best_model.pkl")
else:
    st.write('### Please upload your dataset')


