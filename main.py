import os
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from googletrans import Translator
from dotenv import load_dotenv
from rag_utils import process_pdf, get_rag_chain

# Load environment variables
load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory setup
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
current_pdf_path = os.path.join(UPLOAD_DIR, "current.pdf")

# Templates directory
templates = Jinja2Templates(directory="templates")

# Request schema
class QuestionRequest(BaseModel):
    question: str
    language: str = "en"  # default language


@app.get("/", response_class=HTMLResponse)
def serve_upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open(current_pdf_path, "wb") as f:
        f.write(contents)

    print("📄 PDF uploaded and saved to:", current_pdf_path)

    process_pdf(current_pdf_path)

    print("✅ RAG chain created.")
    return RedirectResponse(url="/chat", status_code=303)


@app.get("/chat", response_class=HTMLResponse)
def serve_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/ask")
def ask_question(req: QuestionRequest):
    rag_chain = get_rag_chain()
    if rag_chain is None:
        return {"answer": "❗Please upload a PDF first."}

    try:
        response = rag_chain.invoke(req.question)
        answer = response.get("result") if isinstance(response, dict) else str(response)

        if req.language != "en":
            translator = Translator()
            answer = translator.translate(answer, dest=req.language).text

        return {"answer": answer}
    except Exception as e:
        return {"answer": f"⚠️ Error: {str(e)}"}
