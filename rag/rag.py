import pymupdf
from openai import OpenAI
from transformers import BertTokenizer, BertModel
import torch
import sys
import json
import faiss
import numpy as np
import os

# Set the environment variable to allow multiple OpenMP runtimes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize OpenAI client
client = OpenAI(
    api_key="*****************************************************",
    base_url="https://api.llama-api.com"
)

def interact_with_llm(model, relevant_chunks, question):
    context = " ".join(relevant_chunks)
    formatted_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        model=model,
        stream=False
    )
    
    return response

def load_chunk_pdf(pdf_file):
    # Initialize a list to store the text chunks
    text_chunks = []
    
    # Open the PDF file
    doc = pymupdf.open(pdf_file)
    print(f"Opened PDF file: {pdf_file}")

    # Iterate through each page in the document
    total_pages = len(doc)
    for page_num in range(total_pages):
        page = doc.load_page(page_num)
        
        # Extract text from the page
        page_text = page.get_text()
        
        # Split the text into chunks of 1000 characters without overlap
        chunk_size = 1000
        for i in range(0, len(page_text), chunk_size):
            chunk = page_text[i:i+chunk_size]
            text_chunks.append(chunk)
        
        # Log progress for each page processed
        print(f"Processed page {page_num + 1}/{total_pages}")

    # Close the document
    doc.close()
    print("Closed PDF file.")
    
    return text_chunks

def handle_uploaded_file(file):
    uploads_dir = os.path.join(os.getcwd(), 'rag/uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    file_path = os.path.join(uploads_dir, file.name)
    with open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    
    return file_path

def embed_chunks(text_chunks):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []
    for i in range(0, len(text_chunks), 8):
        batch_chunks = text_chunks[i:i+8]
        inputs = tokenizer(batch_chunks, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        for j in range(len(batch_chunks)):
            embeddings.append(outputs.last_hidden_state[j, 0, :].numpy())
        print(f"Processed {i + len(batch_chunks)}/{len(text_chunks)} chunks")
    return np.array(embeddings)

def create_vector_database(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print("Vector database created and embeddings added.")
    return index

def save_vector_database(index, file_path):
    faiss.write_index(index, file_path)
    print(f"Vector database saved to {file_path}.")

def load_vector_database(file_path):
    index = faiss.read_index(file_path)
    print(f"Vector database loaded from {file_path}.")
    return index

def search_vector_database(index, query_embedding, k=5):
    distances, indices = index.search(np.array([query_embedding]), k)
    return indices[0]

def embed_query(query, tokenizer, model):
    inputs = tokenizer([query], return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :].numpy()

# def interact_with_llm(model, relevant_chunks, question):
#   context = " ".join(relevant_chunks)
#   formatted_prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#   response = client.completions.create(
#       model=model,
#       prompt=formatted_prompt,
#       max_tokens=512,
#       temperature=0,
#       stream=False
#   )
#   return response

def main(pdf_file, question):
    
    # Load chunks from PDF
    print("Loading chunks from PDF...")
    text_chunks = load_chunk_pdf(pdf_file)
    print(f"Total text chunks loaded: {len(text_chunks)}")

    pdf_filename = os.path.basename(pdf_file)
    pdf_name, _ = os.path.splitext(pdf_filename)
    db_dir_path = os.path.join(os.getcwd(), 'rag', 'db')
    os.makedirs(db_dir_path, exist_ok=True)

    vector_db_path = os.path.join(db_dir_path, f"{pdf_name}.faiss")
    
    if not os.path.exists(vector_db_path):
        embeddings = embed_chunks(text_chunks)
        index = create_vector_database(embeddings)
        save_vector_database(index, vector_db_path)
    else:
        index = load_vector_database(vector_db_path)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    query_embedding = embed_query(question, tokenizer, model)

    indices = search_vector_database(index, query_embedding)
    relevant_chunks = [text_chunks[i] for i in indices]

    # model_id = "meta-llama/Llama-2-13b-chat-hf"
    model_id = "llama3-70b"
    # model_id = "gpt-3.5-turbo"
    print("\n\n Question: ", question)
    response = interact_with_llm(model_id, relevant_chunks, question)
        # Assuming 'answer' contains the response as shown in your example
    for key, value in response:
      if key == 'choices':
        for choice in value:
          answer_text = choice.message.content
          print("\n\nAnswer:", answer_text)
          return answer_text
    # for key, value in answer:
    #   if key == 'choices':
    #     for choice in value:
    #       answer_text = choice.text
    #       # Find the position of the substring "Question:"
    #       question_index = answer_text.find("Question:")
    #       # Print the text up to the "Question:" substring
    #       if question_index != -1:
    #         answer_text = answer_text[:question_index].strip()
    #       print("\n\nAnswer:", answer_text)
    #       return answer_text



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_pdf>")
        sys.exit(1)

    pdf_file = sys.argv[1]
    main(pdf_file)