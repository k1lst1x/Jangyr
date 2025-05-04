import os, re, shutil, zipfile, time, tempfile
from pathlib import Path

from django.shortcuts import render
from django.http import JsonResponse
from django.utils import timezone
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.utils.html import escape
import markdown

from .forms import DocumentUploadForm
from .models import Chat

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from django.db import transaction, IntegrityError, DatabaseError

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")

from dotenv import load_dotenv

load_dotenv()

CHROMA_DB_DIR = os.path.join(settings.BASE_DIR, "chroma_db")

def save_uploaded_file(f) -> Path:
    """Сохраняем загруженный файл и возвращаем путь."""
    name = default_storage.save(f.name, f)
    return Path(settings.MEDIA_ROOT) / name

def extract_if_zip(path: Path) -> Path:
    """Если загружен ZIP — распаковываем и возвращаем папку, иначе сам файл."""
    if zipfile.is_zipfile(path):
        extract_dir = path.with_suffix("")          # <имя_архива>
        with zipfile.ZipFile(path) as z:
            z.extractall(extract_dir)
        return extract_dir
    return path

def collect_documents(root: Path, user_id: int, set_name: str) -> list[Document]:
    docs = []
    loaders_by_ext = {
        ".txt": TextLoader,
        ".md":  UnstructuredMarkdownLoader,
        ".pdf": PyPDFLoader,
        ".docx": UnstructuredWordDocumentLoader,
    }

    for file in root.rglob("*"):
        if file.is_dir():
            continue
        loader_cls = loaders_by_ext.get(file.suffix.lower(), UnstructuredFileLoader)

        try:
            for d in loader_cls(str(file)).load():
                d.metadata.update({
                    "file_path": str(file.relative_to(root)),      # показываемое имя
                    "origin":    str(file.relative_to(root)),      # ← запоминаем «родителя»
                    "set_name":  set_name,
                    "user_id":   str(user_id),
                })
                docs.append(d)
        except Exception:
            continue
    return docs

def build_vectorstore(docs: list[Document], user_id: int, set_name: str):
    if not docs:
        return
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    path = Path(CHROMA_DB_DIR) / f"{user_id}_{set_name}"
    if path.exists():
        shutil.rmtree(path)           # пересоздаём при повторной загрузке

    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        persist_directory=str(path)
    ).persist()

def ask_openai(message: str, user_key: str = "default") -> str:
    """Диалог с ассистентом без ежедневных лимитов."""
    thread_key = f"openai_thread_{user_key}"
    from django.core.cache import cache
    thread_id = cache.get(thread_key)
    if not thread_id:
        thread_id = client.beta.threads.create().id
        cache.set(thread_key, thread_id, None)

    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=message)
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=ASSISTANT_ID)

    while True:
        run_status = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if run_status.status == "completed":
            break
        if run_status.status in ("failed", "cancelled"):
            return "❌ Ошибка обработки запроса."
        time.sleep(1)

    msg = client.beta.threads.messages.list(thread_id=thread_id).data[0]
    return msg.content[0].text.value

def query_docs(question: str, set_name: str, user_id: int) -> str:
    path = Path(CHROMA_DB_DIR) / f"{user_id}_{set_name}"
    if not path.exists():
        return "⚠️ Сначала загрузите документы."

    db = Chroma(
        persist_directory=str(path),
        embedding_function=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    )
    context = "\n\n".join([d.page_content[:1000] for d in db.similarity_search(question, k=10)])
    prompt = (
        "Ты эксперт по нормативным документам. Используя контекст ниже, дай исчерпывающий ответ.\n\n"
        f"{context}\n\nВопрос: {question}"
    )
    return ask_openai(prompt, user_key=str(user_id))

# ---  summary.py  -----------------------------------------------------
# utils/summary.py  (или рядом с views.py)
def make_rich_summary(docs: list[Document], set_name: str, user_id: int) -> str:
    """
    Возвращает расширенный (до 250 слов) обзор каждого уникального документа.
    Без списка вопросов.
    """
    picked, seen = [], set()
    for d in docs:
        origin = d.metadata["origin"]
        if origin in seen or len(d.page_content.strip()) < 80:
            continue
        picked.append(d)
        seen.add(origin)
        if len(picked) == 8:                      # хватит 8 файлов
            break

    parts = [
        f"Файл: {d.metadata['origin']}\nТекст (фрагмент):\n{d.page_content[:4000]}"
        for d in picked
    ] or ["(текст не извлечён, но документ присутствует)"]

    joined_parts = {'\n\n'.join(parts)}

    prompt = f"""
Ты — эксперт по внутренним нормативным документам (ВНД) и НПА.
Анализ файлов с метаданными пропускай.
Отформатируй финальный анализ чтобы его можно было удобно читать.
Составь развернутый, но емкий обзор (до 250 слов на документ). 
Нужно отразить:
• основное назначение и область применения; 
• ключевые обязанности / изменения; 
• затронутые подразделения или должности; 
• возможные риски несоблюдения. 
Не добавляй вопросов и выводов, только фактический анализ.

Набор: {set_name}
----------------------------------------------------
{joined_parts}
----------------------------------------------------
Формат ответа:
### Анализ
- <имя файла без его расширения> — <аннотация 200–250 слов>
"""
    return ask_openai(prompt, user_key=f"{user_id}_{set_name}_rich")

# ----------  ОСНОВНАЯ ВЬЮШКА ЧАТА  ------------------------------------

@login_required
def chatbot(request):
    user = request.user
    chats = Chat.objects.filter(user=user).order_by("created_at")
    upload_form = DocumentUploadForm()

    if request.method == "POST":
        raw_msg  = request.POST.get("message", "").strip()
        uploaded = request.FILES.get("file")

        # Что будем сохранять в Chat.message
        user_message = (
            raw_msg
            if raw_msg
            else (f"Загрузил файл: {uploaded.name}" if uploaded else "")
        )

        try:
            # --- 1) сохраняем "черновик" чата ------------------------
            draft = Chat.objects.create(
                user    = user,
                message = user_message,
                response= "⌛ Запуск ИИ…",
            )

            # --- 2) длинные вычисления -------------------------------
            if uploaded:
                path      = save_uploaded_file(uploaded)
                obj_root  = extract_if_zip(path)
                set_name  = obj_root.stem
                docs      = collect_documents(obj_root, user.id, set_name)
                build_vectorstore(docs, user.id, set_name)

                summary   = make_rich_summary(docs, set_name, user.id)

                unique_files = {d.metadata["origin"] for d in docs}
                resp_text = (
                    f"{summary}\n\n"
                    f"Чтобы уточнить: `вопрос по {set_name}: <ваш вопрос>`"
                )

            elif re.match(r"(?i)^вопрос по ([^:]+):", raw_msg):
                set_name, question = re.match(r"(?i)^вопрос по ([^:]+):\s*(.+)$", raw_msg).groups()
                resp_text = query_docs(question, set_name, user.id)

            else:
                resp_text = ask_openai(raw_msg, user_key=str(user.id))

            # --- 3) обновляем черновик после вычислений --------------
            draft.response = markdown.markdown(
                resp_text, extensions=["fenced_code", "tables", "nl2br", "extra"]
            )
            draft.save(update_fields=["response"])

        except Exception as exc:
            draft.response = f"❌ Ошибка сервера: {exc}"
            draft.save(update_fields=["response"])
            return JsonResponse({"message": escape(user_message),
                                "response": draft.response}, status=500)

        # return JsonResponse({"message": escape(user_message), "response": draft.response})
        return JsonResponse({"message": escape(user_message), "response": draft.response, "set_name": set_name})

    return render(request, "home.html", {"chats": chats, "upload_form": upload_form})

def index(request):
    return render(request, 'index.html')