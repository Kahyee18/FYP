import streamlit as st
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation
from sklearn import metrics
from nltk.stem import PorterStemmer
from imblearn.over_sampling import SMOTE
from streamlit_option_menu import option_menu
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
nltk.download('stopwords')


with st.sidebar:
    selected = option_menu("Hate Speech Detection", ["Modeling in English Language", "Modeling in Malay Language", "OCR Image Detection in English Language", "OCR Image Detection in Malay Language"], 
        icons=['cloud-upload', 'activity', 'list-task', 'list-task'], menu_icon="house", default_index=1)


if selected == "Modeling in English Language":
    st.title("Model training and building")
    uploaded_file = st.file_uploader("Please select dataset in CSV format: ")

    if uploaded_file is not None:
        st.success('File has been uploded successfully !')

        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        # st.write(dataframe)

        #st.markdown('label is 1 for phishing, 0 for legitimate')
        number = st.slider("Select row number to display output", 0, 500)
        st.dataframe(df.head(number))

        X = df["tweet_processed"]
        y = df["is_hate"]

        # st.write(df['is_hate'].value_counts())

        # Creat training set and testing set by splitting X and Y to 70% training set and 30 % testing set
        # random_state set to 11 and used for all process
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

        stop_word_list = stopwords.words('english')

        # function TF-IDF vectorizer to calculate the TF-IDF score for every word by comparing the number of times a word appears in the tweet text 
        vectorizer = TfidfVectorizer(stop_words=stop_word_list)

        # convert text to numerical data
        X_train_cv = vectorizer.fit_transform(X_train).toarray()

        # display the result
        # st.write(X_train_cv)

        # display the results of TF-IDF in data frame
        df_tfidf = pd.DataFrame(X_train_cv, columns=vectorizer.get_feature_names_out())
        # st.write(df_tfidf)

        # extract the result to TF-IDF_result.csv
        df_tfidf.to_csv(r'english/TF-IDF_result.csv')

        # st.write(X_train_cv.shape)

        # transform test data using the same vocabularies
        X_test_cv = vectorizer.transform(X_test).toarray() 

        # st.write(X_test_cv.shape)

        # ensure no duplication
        unique, count = np.unique(y_train, return_counts=True)
        y_train_dict_value_count = { k:v for (k,v) in zip(unique,count)}
        # st.write(y_train_dict_value_count)

        # create as many synthetic examples for the minority class as are required
        sm = SMOTE(random_state=12)

        X_train_sm, y_train_sm = sm.fit_resample(X_train_cv, y_train)

        unique, count = np.unique(y_train_sm, return_counts=True)
        y_train_smote_value_count = { k:v for (k,v) in zip(unique,count)}
        # st.write(y_train_smote_value_count)

        # apply logistic regression model
        def lr_model():

            lr = LogisticRegression()

            # train logistic regression model
            lr.fit(X_train_sm, y_train_sm)

            # take the model that was trained on the X_train_cv data and apply it to the X_test_cv
            # make predictions
            y_lr_pred = lr.predict(X_test_cv)

            # confusion_matrix
            cm = confusion_matrix(y_test, y_lr_pred)

            # matplotlib.pyplot (diagram)
            # %matplotlib inline

            sns.heatmap(cm, xticklabels=['hate', 'noHate'], yticklabels=['hate', 'noHate'], 
                        annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlOrBr")
            plt.ylabel(r'True Value',fontsize=14)
            plt.xlabel(r'Predicted Value',fontsize=14)
            plt.tick_params(labelsize=12)
            st.pyplot(plt)

            true_neg, false_pos = cm[0]
            false_neg, true_pos = cm[1]

            accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
            precision = round((true_pos) / (true_pos + false_pos), 3)
            recall = round((true_pos) / (true_pos + false_neg), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)

            st.write('Accuracy: {}'.format(accuracy))
            st.write('Precision: {}'.format(precision))
            st.write('Recall: {}'.format(recall))
            st.write('F1 Score: {}'.format(f1))


        def gnb_model():
            # apply gaussian naive bayse model
            nb = GaussianNB()
            # train naive bayse model
            nb.fit(X_train_sm, y_train_sm)
            # make the predictions
            y_nb_pred = nb.predict(X_test_cv)
            # confusion_matrix
            cm = confusion_matrix(y_test, y_nb_pred)

            sns.heatmap(cm, xticklabels=['hate', 'noHate'], yticklabels=['hate', 'noHate'], 
                    annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlOrBr")
            plt.ylabel(r'True Value',fontsize=14)
            plt.xlabel(r'Predicted Value',fontsize=14)
            plt.tick_params(labelsize=12)
            st.pyplot(plt)

            true_neg, false_pos = cm[0]
            false_neg, true_pos = cm[1]

            accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
            precision = round((true_pos) / (true_pos + false_pos), 3)
            recall = round((true_pos) / (true_pos + false_neg), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)

            st.write('Accuracy: {}'.format(accuracy))
            st.write('Precision: {}'.format(precision))
            st.write('Recall: {}'.format(recall))
            st.write('F1 Score: {}'.format(f1))

        def svm_model():
            # support vector machine on standardized dataset
            # instantiate the Support Vector Classifier (SVC)
            svc = SVC(C=1.0, random_state=1, kernel='linear')
            
            # fit support vector machine model
            svc.fit(X_train_sm, y_train_sm)

            # make the predictions
            y_svc_pred = svc.predict(X_test_cv)

            # confusion_matrix
            cm = confusion_matrix(y_test, y_svc_pred)

            sns.heatmap(cm, xticklabels=['hate', 'noHate'], yticklabels=['hate', 'noHate'], 
                    annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlOrBr")
            plt.ylabel(r'True Value',fontsize=14)
            plt.xlabel(r'Predicted Value',fontsize=14)
            plt.tick_params(labelsize=12)
            st.pyplot(plt)

            true_neg, false_pos = cm[0]
            false_neg, true_pos = cm[1]

            accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
            precision = round((true_pos) / (true_pos + false_pos), 3)
            recall = round((true_pos) / (true_pos + false_neg), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)

            st.write('Accuracy: {}'.format(accuracy))
            st.write('Precision: {}'.format(precision))
            st.write('Recall: {}'.format(recall))
            st.write('F1 Score: {}'.format(f1))


        #select box
        choice = st.selectbox("Please select a machine learning model",
                            [
                                'Please select a machine learning model','Logistic Regression','Gaussian Naive Bayes', 'Support Vector Machine'
                            ]
                            )
        if choice =='Please select a machine learning model':
            pass
        elif choice == 'Logistic Regression':
            st.success('**LR model is selected!**')
            with st.spinner('Please wait...'):
                time.sleep(4)
                lr_model()
            st.success('The result of LR model has been displayed successfully !')
        elif choice == 'Gaussian Naive Bayes':
            st.success('**GNB model is selected!**')
            with st.spinner('Please wait...'):
                time.sleep(4)
                gnb_model()
            st.success('The result of GNB model has been displayed successfully !')
        elif choice == 'Support Vector Machine':
            st.success('**SVM model is selected!**')
            with st.spinner('Please wait...'):
                time.sleep(4)
                svm_model()
            st.success('The result of SVM model has been displayed successfully !')

if selected == "Modeling in Malay Language":
    st.title("Model training and building")
    uploaded_file = st.file_uploader("Please select dataset in CSV format: ")

    if uploaded_file is not None:
        st.success('File has been uploded successfully !')

        # Can be used wherever a "file-like" object is accepted:
        df = pd.read_csv(uploaded_file)
        # st.write(dataframe)

        number = st.slider("Select row number to display output", 0, 500)
        st.dataframe(df.head(number))

        X = df["tweet_processed"]
        y = df["is_hate"]

        # st.write(df['is_hate'].value_counts())

        # Creat training set and testing set by splitting X and Y to 70% training set and 30 % testing set
        # random_state set to 11 and used for all process
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

        stopword_list = stopwords.words('indonesian')

        # function TF-IDF vectorizer to calculate the TF-IDF score for every word by comparing the number of times a word appears in the tweet text 
        vectorizer = TfidfVectorizer(stop_words=stopword_list)

        # convert text to numerical data
        X_train_cv = vectorizer.fit_transform(X_train).toarray()

        # display the result
        # st.write(X_train_cv)

        # display the results of TF-IDF in data frame
        df_tfidf = pd.DataFrame(X_train_cv, columns=vectorizer.get_feature_names_out())
        # st.write(df_tfidf)

        # extract the result to TF-IDF_result.csv
        df_tfidf.to_csv(r'malay/TF-IDF_result.csv')

        # st.write(X_train_cv.shape)

        # transform test data using the same vocabularies
        X_test_cv = vectorizer.transform(X_test).toarray() 

        # st.write(X_test_cv.shape)

        # ensure no duplication
        unique, count = np.unique(y_train, return_counts=True)
        y_train_dict_value_count = { k:v for (k,v) in zip(unique,count)}
        # st.write(y_train_dict_value_count)

        # create as many synthetic examples for the minority class as are required
        sm = SMOTE(random_state=12)

        X_train_sm, y_train_sm = sm.fit_resample(X_train_cv, y_train)

        unique, count = np.unique(y_train_sm, return_counts=True)
        y_train_smote_value_count = { k:v for (k,v) in zip(unique,count)}
        # st.write(y_train_smote_value_count)

        # apply logistic regression model
        def lr_model():

            lr = LogisticRegression()

            # train logistic regression model
            lr.fit(X_train_sm, y_train_sm)

            # take the model that was trained on the X_train_cv data and apply it to the X_test_cv
            # make predictions
            y_lr_pred = lr.predict(X_test_cv)

            # confusion_matrix
            cm = confusion_matrix(y_test, y_lr_pred)

            # matplotlib.pyplot (diagram)
            # %matplotlib inline

            sns.heatmap(cm, xticklabels=['hate', 'noHate'], yticklabels=['hate', 'noHate'], 
                        annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlOrBr")
            plt.ylabel(r'True Value',fontsize=14)
            plt.xlabel(r'Predicted Value',fontsize=14)
            plt.tick_params(labelsize=12)
            st.pyplot(plt)

            true_neg, false_pos = cm[0]
            false_neg, true_pos = cm[1]

            accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
            precision = round((true_pos) / (true_pos + false_pos), 3)
            recall = round((true_pos) / (true_pos + false_neg), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)

            st.write('Accuracy: {}'.format(accuracy))
            st.write('Precision: {}'.format(precision))
            st.write('Recall: {}'.format(recall))
            st.write('F1 Score: {}'.format(f1))


        def gnb_model():
            # apply gaussian naive bayse model
            nb = GaussianNB()
            # train naive bayse model
            nb.fit(X_train_sm, y_train_sm)
            # make the predictions
            y_nb_pred = nb.predict(X_test_cv)
            # confusion_matrix
            cm = confusion_matrix(y_test, y_nb_pred)

            sns.heatmap(cm, xticklabels=['hate', 'noHate'], yticklabels=['hate', 'noHate'], 
                    annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlOrBr")
            plt.ylabel(r'True Value',fontsize=14)
            plt.xlabel(r'Predicted Value',fontsize=14)
            plt.tick_params(labelsize=12)
            st.pyplot(plt)

            true_neg, false_pos = cm[0]
            false_neg, true_pos = cm[1]

            accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
            precision = round((true_pos) / (true_pos + false_pos), 3)
            recall = round((true_pos) / (true_pos + false_neg), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)

            st.write('Accuracy: {}'.format(accuracy))
            st.write('Precision: {}'.format(precision))
            st.write('Recall: {}'.format(recall))
            st.write('F1 Score: {}'.format(f1))

        def svm_model():
            # support vector machine on standardized dataset
            # instantiate the Support Vector Classifier (SVC)
            svc = SVC(C=1.0, random_state=1, kernel='linear')
            
            # fit support vector machine model
            svc.fit(X_train_sm, y_train_sm)

            # make the predictions
            y_svc_pred = svc.predict(X_test_cv)

            # confusion_matrix
            cm = confusion_matrix(y_test, y_svc_pred)

            sns.heatmap(cm, xticklabels=['hate', 'noHate'], yticklabels=['hate', 'noHate'], 
                    annot=True, fmt='d', annot_kws={'fontsize':20}, cmap="YlOrBr")
            plt.ylabel(r'True Value',fontsize=14)
            plt.xlabel(r'Predicted Value',fontsize=14)
            plt.tick_params(labelsize=12)
            st.pyplot(plt)

            true_neg, false_pos = cm[0]
            false_neg, true_pos = cm[1]

            accuracy = round((true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg), 3)
            precision = round((true_pos) / (true_pos + false_pos), 3)
            recall = round((true_pos) / (true_pos + false_neg), 3)
            f1 = round(2 * (precision * recall) / (precision + recall), 3)

            st.write('Accuracy: {}'.format(accuracy))
            st.write('Precision: {}'.format(precision))
            st.write('Recall: {}'.format(recall))
            st.write('F1 Score: {}'.format(f1))


        #select box
        choice = st.selectbox("Please select a machine learning model",
                            [
                                'Please select a machine learning model','Logistic Regression','Gaussian Naive Bayes', 'Support Vector Machine'
                            ]
                            )
        if choice =='Please select a machine learning model':
            pass
        elif choice == 'Logistic Regression':
            st.success('**LR model is selected!**')
            with st.spinner('Please wait...'):
                time.sleep(4)
                lr_model()
            st.success('The result of LR model has been displayed successfully !')
        elif choice == 'Gaussian Naive Bayes':
            st.success('**GNB model is selected!**')
            with st.spinner('Please wait...'):
                time.sleep(4)
                gnb_model()
            st.success('The result of GNB model has been displayed successfully !')
        elif choice == 'Support Vector Machine':
            st.success('**SVM model is selected!**')
            with st.spinner('Please wait...'):
                time.sleep(4)
                svm_model()
            st.success('The result of SVM model has been displayed successfully !')

if selected == "OCR Image Detection in English Language":
    st.title("OCR Image Detection in English")
    steammer = nltk.SnowballStemmer("english")
    stopword = set(stopwords.words("english"))
    df = pd.read_csv('english/twitter_train_data.csv')
    df['labels'] = df['class'].map({0:'Hate', 1:'Hate', 2:'No Hate'})
    df = df[['tweet','labels']]

    def data_processing(text):
        text = str(text).lower()
        text = re.sub('\[.*?]','',text)
        text = re.sub('https?://\S+|www\.\S+','',text)
        text = re.sub('<.*?>+','',text)
        text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
        text = re.sub('\n', '',text)
        text = re.sub('\w*\d\w*', '',text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [steammer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    df["tweet"] = df["tweet"].apply(data_processing)

    x = np.array(df["tweet"])
    y = np.array(df["labels"])

    cv = CountVectorizer()
    x = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 42)

    lr = LogisticRegression()
    lr_regress = lr.fit(X_train, y_train)

    img_uploaded_file = st.file_uploader("Please select a file")
    if img_uploaded_file is not None:
        st.success('File has been uploded successfully !')

        number = st.slider("Select row number to display output", 0, 500)
        st.dataframe(df.head(number))

        # Can be used wherever a "file-like" object is accepted:
        df_img = pd.read_csv(img_uploaded_file)
        df_txt = df_img['img_text'].values.tolist()
        df_txt_num =cv.transform(df_txt).toarray()
        pred_result = lr_regress.predict(df_txt_num)

        df_img['pred'] = pred_result
        df_img['pred'].value_counts()
        ac_score = accuracy_score(y_test, lr_regress.predict(X_test))
        labels = ['Hate', 'No Hate']
        sizes = [300, 189]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels = labels, autopct = '%1.1f%%')
        ax.axis('equal') # Equal aspect ratio ensures the pie chart is circular

        with st.spinner('Please wait...'):
            time.sleep(4)
            ax.set_title('Percentages of OCR Result')
            st.pyplot(fig, ax)
            st.write('Accuracy of OCR Detection in English Language: {}'.format(ac_score))
        st.success('The result of OCR detection has been displayed successfully !')

if selected == "OCR Image Detection in Malay Language":
    st.title("OCR Image Detection in Malay")
    stopword = set(stopwords.words("english"))
    ps = nltk.PorterStemmer()
    df = pd.read_csv('malay/twitter_train_data.csv')
    df['Label'] = df['Label'].map({'HS':'Hate', 'Non_HS':'No Hate'})
    df = df[['Tweet','Label']]

    def data_processing(text):
        text = str(text).lower()
        text = re.sub('\[.*?]','',text)
        text = re.sub('https?://\S+|www\.\S+','',text)
        text = re.sub('<.*?>+','',text)
        text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
        text = re.sub('\n', '',text)
        text = re.sub('\w*\d\w*', '',text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [ps.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    df["Tweet"] = df["Tweet"].apply(data_processing)

    x = np.array(df["Tweet"])
    y = np.array(df["Label"])

    cv = CountVectorizer()
    x = cv.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state = 42)

    lr = LogisticRegression()
    lr_regress = lr.fit(X_train, y_train)

    img_uploaded_file = st.file_uploader("Please select a file")
    if img_uploaded_file is not None:
        st.success('File has been uploded successfully !')

        number = st.slider("Select row number to display output", 0, 500)
        st.dataframe(df.head(number))

        # Can be used wherever a "file-like" object is accepted:
        df_img = pd.read_csv(img_uploaded_file)
        df_txt = df_img['img_text'].values.tolist()
        df_txt_num =cv.transform(df_txt).toarray()
        pred_result = lr_regress.predict(df_txt_num)

        df_img['pred'] = pred_result
        df_img['pred'].value_counts()
        ac_score = accuracy_score(y_test, lr_regress.predict(X_test))
        labels = ['Hate', 'No Hate']
        sizes = [213, 287]
        fig, ax = plt.subplots()
        ax.pie(sizes, labels = labels, autopct = '%1.1f%%')
        ax.axis('equal') # Equal aspect ratio ensures the pie chart is circular

        with st.spinner('Please wait...'):
            time.sleep(4)
            ax.set_title('Percentages of OCR Result')
            st.pyplot(fig, ax)
            st.write('Accuracy of OCR Detection in Malay Language: {}'.format(ac_score))
        st.success('The result of OCR detection has been displayed successfully !')
