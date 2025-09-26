# app.py
import os
import io
import re
import zlib
import base64
import json
import time
import tempfile
import subprocess
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import streamlit as st
from streamlit_mermaid import st_mermaid

# PDF parsing (for reading pages only)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# FAISS (direct)
import faiss

# OpenAI official client (works with sk-proj + project id)
from openai import OpenAI

# Optional online renderer
try:
    import requests  # used only for Kroki fallback
except Exception:
    requests = None


# =========================
# App Setup
# =========================
st.set_page_config(page_title="PDF RAG — Multi-Chat (OpenAI + FAISS)", page_icon="💬", layout="wide")
st.title("💬 PDF RAG — Multi-Chat (OpenAI + FAISS)")
st.caption("Create multiple chats, upload different PDFs per chat, save & resume. OpenAI project keys supported. Diagrams are generated as JSON → Mermaid → SVG/PNG.")

SAVE_ROOT = "saved_chats"
os.makedirs(SAVE_ROOT, exist_ok=True)


# -----------------------------
# Sidebar: Auth & Settings
# -----------------------------
with st.sidebar:
    st.header("🔐 OpenAI")
    OPENAI_API_KEY = st.text_input("API Key (sk-proj… or sk-…)", value=os.getenv("OPENAI_API_KEY",""), type="password")
    PROJECT_ID = st.text_input("Project ID (proj_…)", help="Required if your key starts with sk-proj-")

    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    st.header("⚙️ Models")
    CHAT_MODEL = st.text_input("Chat model", value="gpt-4o-mini")
    EMBED_MODEL = st.text_input("Embedding model", value="text-embedding-3-small")

    st.header("🧩 Chunking & Retrieval")
    CHUNK_SIZE = st.slider("Chunk size", 400, 2000, 1000, step=100)
    CHUNK_OVERLAP = st.slider("Chunk overlap", 0, 400, 150, step=10)
    TOP_K = st.slider("Top-K", 1, 10, 4)
    MAX_CHARS_CONTEXT = st.slider("Max chars per retrieved chunk", 400, 2000, 1200, step=100)
    TEMPERATURE = st.slider("LLM temperature", 0.0, 1.0, 0.2, step=0.05)

    st.markdown("---")
    st.caption("Tip: Top-K=3–5, chunk ≤ 1,200 chars, overlap 100–200.")


# =========================
# State (multi-chat)
# =========================
def _ensure_state():
    if "chats" not in st.session_state:
        st.session_state.chats: Dict[str, Dict[str, Any]] = {}  # chat_id -> chat dict
    if "current_chat" not in st.session_state:
        st.session_state.current_chat: Optional[str] = None
_ensure_state()


# =========================
# OpenAI client (project key aware)
# =========================
def get_openai_client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        st.error("Missing OpenAI API key. Add it in the sidebar.")
        st.stop()
    kwargs = {"api_key": key}
    if key.startswith("sk-proj"):
        if not PROJECT_ID.strip():
            st.error("You’re using an sk-proj key. Please enter your Project ID (starts with proj_…).")
            st.stop()
        kwargs["project"] = PROJECT_ID.strip()
    return OpenAI(**kwargs)


# =========================
# General Helpers
# =========================
def _file_sha(file_bytes: bytes) -> str:
    import hashlib
    return hashlib.sha256(file_bytes).hexdigest()[:12]

def now_id() -> str:
    return time.strftime("chat_%Y%m%d_%H%M%S")

def chat_dir(chat_id: str) -> str:
    d = os.path.join(SAVE_ROOT, chat_id)
    os.makedirs(d, exist_ok=True)
    return d

def doclist_to_text_meta(docs: List[Document]) -> Tuple[list[str], list[dict]]:
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    return texts, metas


# =========================
# PDF Load + Split
# =========================
def load_and_split_pdfs(uploaded_files: List[io.BytesIO], chunk_size: int, chunk_overlap: int) -> Tuple[List[Document], list]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks: List[Document] = []
    stats = []

    for uf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pdf_pages = loader.load()  # one Document per page
            for d in pdf_pages:
                d.metadata["filename"] = getattr(uf, "name", os.path.basename(tmp_path))
                if isinstance(d.metadata.get("page"), int):
                    d.metadata["page"] = d.metadata["page"] + 1  # 1-indexed
            chunks = splitter.split_documents(pdf_pages)
            for i, c in enumerate(chunks):
                c.metadata["chunk_id"] = i + 1
            all_chunks.extend(chunks)
            stats.append({"file": getattr(uf, "name", os.path.basename(tmp_path)),
                          "pages": len(pdf_pages),
                          "chunks": len(chunks)})
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    return all_chunks, stats


# =========================
# Embeddings + FAISS
# =========================
def embed_texts(client: OpenAI, model: str, texts: List[str], batch_size: int = 128) -> np.ndarray:
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
    return np.array(vecs, dtype="float32")

def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-8, None)
    return vecs / norms

def build_faiss_index(client: OpenAI, model: str, texts: list[str]) -> faiss.Index:
    raw = embed_texts(client, model, texts)
    vecs = _normalize(raw)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
    index.add(vecs)
    return index

def faiss_search(index: faiss.Index, query: str, client: OpenAI, model: str, texts: list[str], metas: list[dict], k: int):
    q = embed_texts(client, model, [query])[0].astype("float32").reshape(1, -1)
    q = _normalize(q)
    D, I = index.search(q, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if idx == -1:
            continue
        results.append((float(D[0][rank]), int(idx), texts[idx], metas[idx]))
    return results


# =========================
# Chat I/O
# =========================
def format_context(hits, max_chars: int = 1200) -> str:
    parts = []
    for i, (_, idx, txt, meta) in enumerate(hits, 1):
        fname = meta.get("filename", "doc")
        page = meta.get("page", "?")
        cid = meta.get("chunk_id", idx)
        parts.append(f"[Source {i}: {fname} p.{page} c{cid}]\n{txt[:max_chars]}")
    return "\n\n".join(parts)

def cite_tag(meta: dict) -> str:
    return f"[{meta.get('filename','doc')} p.{meta.get('page','?')}]"

def stream_chat(client: OpenAI, model: str, messages, temperature: float = 0.2):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        stream=True,
    )
    for ev in stream:
        delta = ev.choices[0].delta
        if delta and delta.content:
            yield delta.content


# =========================
# Diagram helpers (JSON → Mermaid)
# =========================
_slug_rx = re.compile(r"[^A-Za-z0-9_]")

def slugify_id(s: str) -> str:
    """Create a safe Mermaid node id (no spaces/punct)."""
    s = (s or "").strip().replace(" ", "_")
    s = _slug_rx.sub("", s)
    if not s:
        s = "N"
    if s[0].isdigit():
        s = f"n_{s}"
    return s[:48]  # keep ids short

def graph_json_to_mermaid(graph: Dict[str, Any], orientation: str = "TD") -> str:
    """
    Graph schema:
    {
      "nodes": [{"id": "optional", "label": "string"}],
      "edges": [{"source": "string", "target": "string", "label": "optional"}]
    }
    """
    nodes = graph.get("nodes", []) or []
    edges = graph.get("edges", []) or []

    id_for: Dict[str, str] = {}
    out_nodes = []

    # Build node ids
    for n in nodes:
        label = (n.get("label") or n.get("name") or n.get("id") or "Node").strip()
        raw_id = n.get("id") or label
        nid = slugify_id(raw_id)
        i = 2
        base = nid
        while nid in id_for.values():
            nid = f"{base}_{i}"
            i += 1
        id_for[label] = nid
        out_nodes.append(f'{nid}["{label}"]')

    def _resolve(s: str) -> str:
        if not s:
            return "N"
        s = s.strip()
        if s in id_for.values():
            return s
        if s in id_for:
            return id_for[s]
        return slugify_id(s)

    out = [f"flowchart {orientation}"]
    for line in out_nodes:
        out.append(line)
    for e in edges:
        src = _resolve(e.get("source", ""))
        dst = _resolve(e.get("target", ""))
        lab = e.get("label")
        if lab:
            out.append(f'{src} --|{lab}|--> {dst}')
        else:
            out.append(f"{src} --> {dst}")
    return "\n".join(out)

def mmdc_available() -> bool:
    try:
        subprocess.run(["mmdc", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return True
    except Exception:
        return False

def export_mermaid_via_mmdc(mermaid_code: str) -> Tuple[str, str]:
    """Export using Mermaid CLI: returns (svg_path, png_path)."""
    tmpdir = tempfile.mkdtemp()
    mmd_path = os.path.join(tmpdir, "diagram.mmd")
    svg_path = os.path.join(tmpdir, "diagram.svg")
    png_path = os.path.join(tmpdir, "diagram.png")
    with open(mmd_path, "w", encoding="utf-8") as f:
        f.write(mermaid_code)
    subprocess.run(["mmdc", "-i", mmd_path, "-o", svg_path], check=True)
    subprocess.run(["mmdc", "-i", mmd_path, "-o", png_path], check=True)
    return svg_path, png_path

def export_mermaid_via_kroki(mermaid_code: str) -> Optional[bytes]:
    """Render SVG via kroki.io (requires internet & requests)."""
    if requests is None:
        return None
    data = zlib.compress(mermaid_code.encode("utf-8"))[2:-4]
    b64 = base64.urlsafe_b64encode(data).decode("ascii")
    url = f"https://kroki.io/mermaid/svg/{b64}"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.content


# =========================
# Persistence (per-chat)
# =========================
def save_chat_to_disk(chat: Dict[str, Any]):
    d = chat_dir(chat["id"])
    with open(os.path.join(d, "messages.json"), "w") as f:
        json.dump(chat["messages"], f, indent=2)
    with open(os.path.join(d, "files.json"), "w") as f:
        json.dump(chat.get("files", []), f, indent=2)
    if "texts" in chat and "metas" in chat:
        with open(os.path.join(d, "texts.json"), "w") as f:
            json.dump(chat["texts"], f)
        with open(os.path.join(d, "metas.json"), "w") as f:
            json.dump(chat["metas"], f)
    if chat.get("index") is not None:
        faiss.write_index(chat["index"], os.path.join(d, "index.faiss"))
    st.success(f"Saved chat → {d}")

def load_chats_from_disk():
    loaded = 0
    for name in os.listdir(SAVE_ROOT):
        d = os.path.join(SAVE_ROOT, name)
        if not os.path.isdir(d):
            continue
        chat_id = name
        try:
            with open(os.path.join(d, "messages.json")) as f:
                messages = json.load(f)
        except Exception:
            messages = []
        try:
            with open(os.path.join(d, "files.json")) as f:
                files = json.load(f)
        except Exception:
            files = []
        texts = metas = index = None
        tpath = os.path.join(d, "texts.json")
        mpath = os.path.join(d, "metas.json")
        ipath = os.path.join(d, "index.faiss")
        if os.path.exists(tpath) and os.path.exists(mpath) and os.path.exists(ipath):
            try:
                with open(tpath) as f:
                    texts = json.load(f)
                with open(mpath) as f:
                    metas = json.load(f)
                index = faiss.read_index(ipath)
            except Exception:
                pass
        st.session_state.chats[chat_id] = {
            "id": chat_id,
            "title": chat_id.replace("_", " "),
            "messages": messages,
            "files": files,
            "texts": texts,
            "metas": metas,
            "index": index,
        }
        loaded += 1
    st.success(f"Loaded {loaded} chats from disk." if loaded else "No saved chats found.")


# =========================
# Chat Manager UI (sidebar)
# =========================
with st.sidebar:
    st.markdown("---")
    st.subheader("💬 Chats")

    if st.button("➕ New Chat", use_container_width=True):
        cid = now_id()
        st.session_state.chats[cid] = {
            "id": cid,
            "title": f"Chat {len(st.session_state.chats)+1}",
            "messages": [],
            "files": [],
            "texts": None,
            "metas": None,
            "index": None,
        }
        st.session_state.current_chat = cid

    if st.button("📂 Load Saved Chats", use_container_width=True):
        load_chats_from_disk()

    if st.session_state.chats:
        for cid, chat in sorted(st.session_state.chats.items(), key=lambda kv: kv[0], reverse=True):
            label = f"• {chat.get('title', cid)}"
            if st.button(label, key=f"btn_{cid}", use_container_width=True):
                st.session_state.current_chat = cid

    if st.session_state.current_chat:
        if st.button("💾 Save Current Chat", type="primary", use_container_width=True):
            save_chat_to_disk(st.session_state.chats[st.session_state.current_chat])


# =========================
# Main Pane — per-chat UI
# =========================
if not st.session_state.current_chat:
    st.info("Create a **New Chat** in the sidebar to get started.")
    st.stop()

chat = st.session_state.chats[st.session_state.current_chat]
st.subheader(f"Current: {chat['title']}")

# Upload + Build index (per chat)
uploaded = st.file_uploader("Upload PDFs for this chat", type=["pdf"], accept_multiple_files=True, key=f"uploader_{chat['id']}")
c1, c2 = st.columns([1, 1])

with c1:
    if st.button("Build / Rebuild Index for this Chat", use_container_width=True):
        if not uploaded:
            st.warning("Please upload at least one PDF.")
        else:
            client = get_openai_client()
            t0 = time.time()
            docs, stats = load_and_split_pdfs(uploaded, CHUNK_SIZE, CHUNK_OVERLAP)
            texts, metas = doclist_to_text_meta(docs)

            with st.status("Embedding & indexing…", expanded=False) as s:
                index = build_faiss_index(client, EMBED_MODEL, texts)
                chat["texts"] = texts
                chat["metas"] = metas
                chat["index"] = index
                files_meta = [{"name": getattr(uf, "name", "uploaded.pdf"), "sha": _file_sha(uf.getvalue())} for uf in uploaded]
                chat["files"] = files_meta
                s.update(label="Index built ✅", state="complete")

            st.success(f"Index built with {len(texts)} chunks in {time.time()-t0:.1f}s")
            with st.expander("Files indexed"):
                for s_ in stats:
                    st.write(f"• {s_['file']} — {s_['pages']} pages → {s_['chunks']} chunks")

with c2:
    if chat.get("index") is not None:
        st.info("Index ready for this chat. Ask below ✨")
    else:
        st.info("Upload PDFs and click **Build / Rebuild Index for this Chat**")

st.divider()

# Show chat history
for role, content in chat["messages"]:
    st.chat_message(role).markdown(content)

# Chat input (per chat)
user_prompt = st.chat_input("Ask about these PDFs… (e.g., 'Summarize Section 2', 'diagram the process')")
if user_prompt:
    chat["messages"].append(("user", user_prompt))
    st.chat_message("user").markdown(user_prompt)

    if chat.get("index") is None or chat.get("texts") is None or chat.get("metas") is None:
        msg = "No index yet for this chat. Upload PDFs and build the index first."
        st.chat_message("assistant").warning(msg)
        chat["messages"].append(("assistant", msg))
    else:
        client = get_openai_client()
        hits = faiss_search(chat["index"], user_prompt, client, EMBED_MODEL, chat["texts"], chat["metas"], k=TOP_K)

        if not hits:
            msg = "No relevant context found. Try broader terms or rebuild the index with different chunk settings."
            st.chat_message("assistant").warning(msg)
            chat["messages"].append(("assistant", msg))
        else:
            context = format_context(hits, max_chars=MAX_CHARS_CONTEXT)
            lower = user_prompt.lower()

            # ---------- Diagram mode: JSON → Mermaid ----------
            if any(k in lower for k in ["diagram", "architecture", "flowchart", "uml"]):
                sys = (
                    "You are a diagram structurer. Output ONLY valid JSON (no prose) describing a directed graph.\n"
                    "Schema:\n"
                    "{\n"
                    '  \"nodes\": [{\"id\": \"optional string\", \"label\": \"string\"}],\n'
                    '  \"edges\": [{\"source\": \"string\", \"target\": \"string\", \"label\": \"optional string\"}]\n'
                    "}\n"
                    "Rules: ≤ 20 nodes; concise labels (≤ 5 words); no markdown, no backticks, no comments."
                )
                messages = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"Build a graph for this context:\n{context}\n\nFocus on key components and their flow."}
                ]
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    json_accum = ""
                    for delta in stream_chat(client, CHAT_MODEL, messages, TEMPERATURE):
                        json_accum += delta or ""
                        placeholder.code(json_accum, language="json")

                    def safe_parse_json(s: str):
                        try:
                            return json.loads(s)
                        except Exception:
                            m = re.search(r"\{[\s\S]*\}", s)
                            if m:
                                return json.loads(m.group(0))
                            raise

                    try:
                        graph = safe_parse_json(json_accum)
                        mermaid_code = graph_json_to_mermaid(graph, orientation="TD")
                        st.write("Generated diagram:")
                        st_mermaid(mermaid_code)

                        # Export buttons: Mermaid-CLI first, then Kroki fallback
                        exported = False
                        if mmdc_available():
                            try:
                                svg_path, png_path = export_mermaid_via_mmdc(mermaid_code)
                                with open(svg_path, "rb") as f:
                                    st.download_button("⬇️ Download SVG", f, file_name="diagram.svg", mime="image/svg+xml")
                                with open(png_path, "rb") as f:
                                    st.download_button("⬇️ Download PNG", f, file_name="diagram.png", mime="image/png")
                                exported = True
                            except Exception as e:
                                st.caption(f"Mermaid CLI export failed: {e}")

                        if not exported and requests is not None:
                            try:
                                svg_bytes = export_mermaid_via_kroki(mermaid_code)
                                if svg_bytes:
                                    st.download_button("⬇️ Download SVG (Kroki)", svg_bytes, file_name="diagram.svg", mime="image/svg+xml")
                                    exported = True
                            except Exception as e:
                                st.caption(f"Kroki export failed: {e}")

                        if not exported:
                            st.caption("Export unavailable. Install Mermaid CLI (`npm i -g @mermaid-js/mermaid-cli`) or enable internet + `requests` for Kroki fallback.")

                        with st.expander("Sources"):
                            for i, h in enumerate(hits, 1):
                                st.write(f"{i}. {cite_tag(h[3])}")

                        chat["messages"].append(("assistant", "(Diagram rendered above)"))
                    except Exception as e:
                        st.error(f"Could not build diagram JSON: {e}")
                        chat["messages"].append(("assistant", "Diagram generation failed. Try again or refine your request."))

            # ---------- Normal grounded QA ----------
            else:
                sys = (
                    "You are a concise, grounded assistant.\n"
                    "Answer ONLY using the provided context from the user's PDFs.\n"
                    "If the answer isn't in the context, say you don't have enough information.\n"
                    "Cite sources inline like [filename p.#] after facts that use them."
                )
                messages = [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_prompt}"}
                ]
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    out = ""
                    for delta in stream_chat(client, CHAT_MODEL, messages, TEMPERATURE):
                        out += delta or ""
                        placeholder.markdown(out)
                    with st.expander("Sources"):
                        for i, h in enumerate(hits, 1):
                            st.write(f"{i}. {cite_tag(h[3])}")
                chat["messages"].append(("assistant", out))
