import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
import pickle
import os
import gradio as gr
import matplotlib

matplotlib.use('Agg')



def get_answer(input_txt):
    input_text = input_txt
    d, idx = faiss_index.search(sentence_model.encode([input_text]), 1)
    prompt = f"Please respond to the question based on what is known, which is as follows:```{contents[idx[0][0]]}```,The problem for users is:```{input_text}```,All responses are output in both Chinese and English"
    ans = get_test(prompt)
    return ans

if __name__ == '__main__':
    if os.path.exists("contents.pkl")==False:

        loader = DirectoryLoader("new_data")
        documents = loader.load()

        text_spliter = CharacterTextSplitter(chunk_size=300,chunk_overlap=50)

        split_docs = text_spliter.split_documents(documents)
        contents = [i.page_content for i in split_docs]
        with open("contents.pkl","wb") as f:
            pickle.dump(contents,f)

    else:
        with open("contents.pkl","rb") as f:
            contents = pickle.load(f)

    sentence_model = SentenceTransformer("moka-ai_m3e-base")

    print("...Vector service loading...")
    if os.path.exists("faiss_index.pkl")==False:
        faiss_index = faiss.IndexFlatL2(sentence_model.get_sentence_embedding_dimension())
        faiss_index.add(sentence_model.encode(contents))
        with open("faiss_index.pkl","wb") as f:
            pickle.dump(faiss_index,f)
    else:
        with open("faiss_index.pkl","rb") as f:
            faiss_index = pickle.load(f)

    iface = gr.Interface(get_answer, inputs="text", outputs="text",
                         title="Southern Cross AI", description="Enter a question about Australia to get the answer")

    iface.launch()

def get_test(prompt):
    '''
    This part needs to be tailored to the interface of the locally deployed big model you are using, and is user-defined since it contains interface information
    :param prompt:System prompt for input
    :return: solution
    '''
    pass