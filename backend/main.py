
from fastapi import FastAPI,Request
from .rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline(llm=None,embeddings=None,vectorstore=None)

@app.post("/embed")
def embed_docs():
    rag.embed_documents("data/bajaj_finserv_factsheet_Oct.pdf")
    return {'status':'Documents Embeded successfully.'}

@app.post("/query")
async def query_docs(request: Request):
    data = await request.json()
    query = data.get("query", "")
    answer = rag.answer_query(query)
    return {"answer": answer}



