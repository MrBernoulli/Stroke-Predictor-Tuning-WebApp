import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.utils import shuffle
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title("👨🏾‍⚕️ Is the Patient at HIgh Risk of Suffering a Stroke?")
    st.sidebar.title("Hyperparameter Tuning🔧")
    st.markdown(
    """
<style>
.sidebar .sidebar-content {
    background-image: linear-gradient(#2e7bcf,#2e7bcf);
    color: yellow;
},
</style>

""",
    unsafe_allow_html=True,
)
    #st.sidebar.markdown("👨🏾‍⚕️ Is the patient at risk of having a stroke?")
    #st.markdown("👨🏾‍⚕️ Is the patient at risk of having a stroke?")
    
    @st.cache(persist=True) #Using the st.cache function decorator to cache the output of a function to disk,
    #so that if the function or its inputs remain unchanged, we do not unneccesarily call it again.
    def load_data():
        #data = pd.read_csv('mushrooms.csv')
        data = pd.read_csv('stroke_pred.csv')
        #label = LabelEncoder()
        #for col in data.columns:
            #data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist=True)
    def split(df):
        #y = df.type
        #x = df.drop(columns=["type"])
        frac = df.stroke.value_counts()[1]/df.stroke.value_counts()[0]
        ones = df.iloc[[i for i,s in enumerate(df.stroke) if s==1]]
        zeros = df.iloc[[i for i,s in enumerate(df.stroke) if s==0]]
       
        df_train_ones, df_test_ones = train_test_split(ones,test_size=0.5, random_state=50)
        df_train_zeros, df_test_zeros = train_test_split(zeros,test_size=frac*0.5, random_state=50)

        train = shuffle(pd.concat([df_train_ones,df_train_zeros]))
        test = shuffle(pd.concat([df_test_ones,df_test_zeros]))

        x_train, y_train = train.drop(columns=["stroke"]), train.stroke
        x_test, y_test = test.drop(columns=["stroke"]), test.stroke

        return x_train, x_test, y_train, y_test

    def plot_metrics(metrics_list):
        accuracy = model.score(x_train,y_train)
        st.write("Accuracy on Training Set: ",accuracy.round(2))

        accuracy = model.score(x_test,y_test)
        st.write("Accuracy on Test Set: ",accuracy.round(2))

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix on Training Set")
            plot_confusion_matrix(model, x_train, y_train, display_labels = class_names)
            st.pyplot()
        
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix on Test Set")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve on Training Set")
            plot_roc_curve(model, x_train, y_train)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve on Test Set")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve on Training Set")
            plot_precision_recall_curve(model,x_train,y_train)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve on Test Set")
            plot_precision_recall_curve(model,x_test,y_test)
            st.pyplot()


    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if st.sidebar.checkbox("Show raw data", False):
        #st.subheader("Dataset (Scaled)")
        #st.write(df)

        st.subheader("Source: https://www.kaggle.com/fedesoriano/stroke-prediction-dataset")
        st.write(pd.read_csv("healthcare-dataset-stroke-data.csv"))
    
    class_names = ['no stroke', 'stroke']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))
    
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Set Hyperparameters")
        C = st.sidebar.number_input("L2 regularization parameter", 0.001,1000.0,step=0.01,key="C")
        kernel = st.sidebar.radio("Kernel", ("linear","poly","rbf","sigmoid"), key="kernel")
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale","auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics would you like to plot?", ('Confusion Matrix', 'Precision-Recall Curve', 'ROC Curve'))

        if st.sidebar.button("Classify", key="classify"):
            st.header("Support Vector Machine (SVM) Results")
            model = SVC(C=1/C, kernel = kernel, gamma=gamma, class_weight="balanced")
            model.fit(x_train, y_train)
            #accuracy = model.score(x_test,y_test)
            #y_pred = model.predict(x_test)
            #st.write("Accuracy: ",accuracy.round(2))
            #st.write("Precision: ",precision_score(y_test,y_pred, labels=class_names).round(2))
            #st.write("Recall: ",recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("L2 regularization parameter", 0.001,1000.0,step=0.01,key="C_LR")
        max_iter = st.sidebar.slider("Maximum Iterations", 100, 500, key="max_iter")

        metrics = st.sidebar.multiselect("What metrics would you like to plot?", ('Confusion Matrix', 'Precision-Recall Curve', 'ROC Curve'))

        if st.sidebar.button("Classify", key="classify"):
            st.header("Logistic Regression Results")
            model = LogisticRegression(C=1/C, max_iter=max_iter,class_weight='balanced')
            model.fit(x_train, y_train)
            ##y_pred = model.predict(x_test)
            #st.write("Accuracy: ",accuracy.round(2))
            #st.write("Precision: ",precision_score(y_test,y_pred, labels=class_names).round(2))
            #st.write("Recall: ",recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("Number of decision trees",100,500,step=10,key="n_estimators")
        max_depth = st.sidebar.number_input("Maximum tree depth",1,20,step=1,key="max_depth")
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True','False'),key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics would you like to plot?", ('Confusion Matrix', 'Precision-Recall Curve', 'ROC Curve'))

        if st.sidebar.button("Classify", key="classify"):
            st.header("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators,
            max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1,class_weight="balanced")
            model.fit(x_train, y_train)
            #accuracy = model.score(x_test,y_test)
            #y_pred = model.predict(x_test)
            #st.write("Accuracy: ",accuracy.round(2))
            #st.write("Precision: ",precision_score(y_test,y_pred, labels=class_names).round(2))
            #st.write("Recall: ",recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
if __name__ == '__main__':
    main()


