import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def predict_salary(experience,test_score,interview_score):
    input=np.array([[experience,test_score,interview_score]]).astype(np.float64)
    prediction=model.predict(input)
    pred='{}'.format(prediction)
    return float(pred)

def main():
    st.title("hiring prediction")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Hiring Prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    experience = st.text_input("experience","Type Here")
    test_score = st.text_input("test_score","Type Here")
    interview_score = st.text_input("interview_score","Type Here")
    safe_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> Your salary is greater than the average salary</h2>
       </div>
    """
    danger_html="""  
      <div style="background-color:#F08080;padding:10px >
       <h2 style="color:black ;text-align:center;"> Your salary is less than the average salary </h2>
       </div>
    """

    if st.button("Predict"):
        output=predict_salary(experience,test_score,interview_score)
        st.success('your expected salary is {}'.format(output))

        if output < 63000:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()