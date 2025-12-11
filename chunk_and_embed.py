from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import os
import pickle

# recursiveCharacterTextSplitter counts characters, so multiply chunk_size and chunk_overlap by 5 (assuming average word length is 5) 
def chunk(text, chunk_size=3750, chunk_overlap=750):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

documents = []

for filename in os.listdir('./Documents/'):
    with open(f'./Documents/{filename}', 'r', encoding='utf-8') as f:
        text = f.read()
        chunks = chunk(text)
        for i, textChunk in enumerate(chunks):
            documents.append({
                'text': textChunk,
                'source': filename,
                'chunk_id': i
            })

embeddingModel = SentenceTransformer('BAAI/bge-base-en-v1.5')

for i, doc in enumerate(documents):
    embedding = embeddingModel.encode(doc['text'])
    doc['embedding'] = embedding

with open('embedded_documents.pkl', 'wb') as f:
    pickle.dump(documents, f)

print("Documents successfully chunked and embedded")
