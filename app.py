
import streamlit as st
from gensim.models import Word2Vec

from glove import Glove
import pickle



st.title(" ✨Check Nearest word in Eucledian space of कोंकणी Embeddings `✨` ")

embeddingSelection = st.radio('⚡ Select Trained Embedding ⚡: ',
                  ('Word2vec', 'Glove', 'FastText'))

if(embeddingSelection == 'Word2vec'):
    #model is trained on konkani devnagri
    st.success(" Word2vec Loaded succesfully")
    konkaniWord2Vec = Word2Vec.load("Weights/konk2vec_w2v_v1.model")
elif(embeddingSelection == 'Glove'):
    # konkaniWord2Vec= 
    konkaniGlove = Glove.load('Weights/glove.model')
    with open('Weights/gloveCorpusDict.pickle', 'rb') as handle:
        corpusDict = pickle.load(handle)
    konkaniGlove.add_dictionary(corpusDict)
    st.success("Glove Loaded succesfully ")

else: #FasText
    st.error("FastText Not Handled")


textInput= st.text_input("Input a word in Devanagri Script 👨‍💻","कोंकणी")

returnOutput= st.slider("How Many Similar Words?", 1, 10,10)

st.write('You selected:', returnOutput)



if st.button('🔎 Find Nearest Match 🔍'):

    if (embeddingSelection == 'Word2vec'):
    
        if (textInput in konkaniWord2Vec.wv.vocab):
            #loads page
            similarWords = konkaniWord2Vec.wv.most_similar(textInput,topn=int(returnOutput))
            #for word in similarWords:
                #st.info((word[0]))
            html = ""  
            for word in similarWords:
              html=html+"<button style='background-color: #0000FF;  border: none;  color: white;  padding: 15px 32px;  text-align: center;  text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer'>"+word[0]+"</button>"
            st.markdown(html,unsafe_allow_html=True)
        else:
            # Out of Vocabulary
            st.error("Input Text not found in Vocabulary.")

    elif(embeddingSelection == 'Glove'):

        similarWords = konkaniGlove.most_similar(textInput,int(returnOutput))
        #for word in similarWords:
            #st.info((word[0]))
        html = ""  
        for word in similarWords:
            html=html+"<button style='background-color: #0000FF;  border: none;  color: white;  padding: 15px 32px;  text-align: center;  text-decoration: none;display: inline-block;font-size: 16px;margin: 4px 2px;cursor: pointer'>"+word[0]+"</button>"
        st.markdown(html,unsafe_allow_html=True)

    else: #FasText
        st.error(" FasText Not Handled")
    
