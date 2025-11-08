
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub, GPT4All
# from langchain.chains import RetrievalQA
from huggingface_hub import InferenceClient

from transformers import AutoModelForCausalLM, AutoTokenizer

from transformers import pipeline

from langchain_text_splitters import CharacterTextSplitter

class RAGPipeline():
    
    def __init__(self, llm, embeddings, vectorstore):

        self.HF_API_TOKEN = "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"
        # self.llm = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.1",model_kwargs={"temperature": 0.5, "max_length": 512})
        self.vectorstore = None
        self.embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
        self.embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_name)

    def embed_documents(self, file_path):
        loader = PDFPlumberLoader(file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500,separator="\n",chunk_overlap=50)
        
        all_chunks = []

        for page in documents:
            # Extract text 
            text_chunks = text_splitter.split_text(page.page_content)
            all_chunks.extend(text_chunks)

            # Extract tables 

            tables = page.extract_tables()
            for table in tables:
                table_text = self.convert_table_to_string(table)
                all_chunks.append(table_text)
        
        # chunks = text_splitter.split_documents(documents=documents)

        vectorstor = FAISS.from_documents(all_chunks,embedding=self.embeddings)
        vectorstor.save_local("faiss_index")
        
        print(f"Docuement Content : {all_chunks}")

    def convert_table_to_string(self, table):
        
        table_string = ""
        for row in table:
            table_string += " | ".join(map(str, row)) + "\n"  # Join each row with '|'
        return table_string.strip()
    
    def answer_query(self, query):
        query_embeddings = self.embeddings.embed_query(query)
        db = FAISS.load_local("faiss_index",embeddings=self.embeddings,allow_dangerous_deserialization=True)
        docs = db.similarity_search_by_vector(embedding=query_embeddings,k=2)

        context = " ".join([doc.page_content for doc in docs])

        prompt = f"""Answer the following question based on the provided context:

        Context: {context}
        
        Question: {query}
        
        Answer:"""
        
        model_name = "microsoft/phi-2"

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

        # prompt = "Write a short poem about AI and humanity."

        print(f"Prompt : {prompt} \n\n++++++++++++++")

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        print(f"+++++++++++++++++++++++++\n\nResponse : {response}")

        return response
