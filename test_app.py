# app.py  (RAG + OpenAI LLM) — UI allows entering OpenAI API Key directly
import os
import time
import textwrap
import json
from typing import List, Dict, Any, Optional

import pandas as pd
import streamlit as st
import psycopg2
import chromadb

from openai import OpenAI  # Official SDK v1.x

st.set_page_config(page_title="RAG Builder • MechanicalEquipment (with LLM)", layout="wide")

# ------------------ Sidebar: Settings ------------------
st.sidebar.title("Settings")
st.sidebar.caption("DB → Index → Query → (optional) LLM Answer")

# DB config
PG_HOST = st.sidebar.text_input("PG_HOST", os.getenv("PG_HOST", "127.0.0.1"))
PG_DB   = st.sidebar.text_input("PG_DB",   os.getenv("PG_DB",   "research_bim"))
PG_USER = st.sidebar.text_input("PG_USER", os.getenv("PG_USER", "postgres"))
PG_PW   = st.sidebar.text_input("PG_PW",   os.getenv("PG_PW",   ""), type="password")
PG_PORT = st.sidebar.number_input("PG_PORT", min_value=1, max_value=65535, value=int(os.getenv("PG_PORT", "5432")))

sql_default = textwrap.dedent("""
SELECT
  "Id" AS "ElementId",
  "TypeId",
  "USCInstanceName",
  "USCRoomName",
  "USCRoomNumber",
  "USCDesignManufacturer",
  "USCSerialNumber",
  "USCInstallWarrantyDuration",
  "MeridianLink",
  "BIM3Dview"
FROM "MechanicalEquipment";
""").strip()
SQL = st.sidebar.text_area("SQL", sql_default, height=400)

# Embedding & Chroma config
st.sidebar.subheader("Vector Store")
model_name = st.sidebar.text_input("SentenceTransformer model", "all-MiniLM-L6-v2")
persist_path = st.sidebar.text_input("Chroma persist path", "./research_chroma")
collection_name = st.sidebar.text_input("Collection name", "research_mechanical")
batch_size = st.sidebar.slider("Batch size", 100, 1000, 500, step=100)
rebuild = st.sidebar.checkbox("Rebuild collection (delete & recreate)", value=False)

# LLM config
st.sidebar.subheader("LLM (OpenAI)")
use_llm = st.sidebar.checkbox("Enable LLM Q&A", value=True)

# ⭐ NEW: API key from UI (takes priority over env var)
openai_api_key_ui = st.sidebar.text_input("OpenAI API Key", value="", type="password", help="Paste your key here (e.g., sk-...)")
openai_api_key_env = os.getenv("OPENAI_API_KEY", "")
openai_api_key = openai_api_key_ui.strip() or openai_api_key_env.strip()

llm_model = st.sidebar.text_input("OpenAI model", "gpt-4o-mini")
use_responses_api = st.sidebar.checkbox("Use Responses API (recommended)", value=True)
max_context_chars = st.sidebar.slider("Max context char budget", 2_000, 20_000, 8_000, step=1000)
temperature = st.sidebar.slider("temperature", 0.0, 1.0, 0.2, step=0.1)

if use_llm and not openai_api_key:
    st.sidebar.warning("Please enter your OpenAI API Key above (or set OPENAI_API_KEY env var).")

# ------------------ Cache ------------------
@st.cache_resource(show_spinner=False)
def load_st_model(name: str):
    return SentenceTransformer(name)

@st.cache_data(show_spinner=False)
def fetch_df(host, db, user, pw, port, sql) -> pd.DataFrame:
    with psycopg2.connect(host=host, dbname=db, user=user, password=pw, port=port) as conn:
        df = pd.read_sql(sql, conn)
    return df

# ------------------ Helpers ------------------
def row_to_text(r: Dict[str, Any]) -> str:
    parts = []
    name = r.get("USCInstanceName")
    room = r.get("USCRoomName")
    room_no = r.get("USCRoomNumber")
    manuf = r.get("USCDesignManufacturer")
    serial = r.get("USCSerialNumber")
    warranty = r.get("USCInstallWarrantyDuration")
    meridian = r.get("MeridianLink")
    bimview = r.get("BIM3Dview")

    if name: parts.append(f"Equipment {name}")
    if room and room_no: parts.append(f"Room {room} #{room_no}")
    elif room: parts.append(f"Room {room}")
    if manuf: parts.append(f"Manufacturer {manuf}")
    if serial: parts.append(f"Serial {serial}")
    if warranty: parts.append(f"Warranty {warranty}")
    if meridian: parts.append(f"MeridianLink {meridian}")
    if bimview: parts.append(f"BIM3Dview {bimview}")

    parts.append(f"ElementId {r.get('ElementId')}")
    parts.append(f"TypeId {r.get('TypeId')}")
    return ", ".join(parts)

class LocalEF:
    def __init__(self, st_model):
        self._m = st_model
    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._m.encode(input, convert_to_numpy=True).tolist()

def ensure_collection(client, name, ef, rebuild=False):
    if rebuild:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    try:
        # get if exists (no EF to avoid conflicts); else create with EF
        coll = client.get_collection(name=name)
    except Exception:
        coll = client.create_collection(name=name, embedding_function=ef)
    return coll

def safe_truncate(text: str, budget: int) -> str:
    return text if len(text) <= budget else text[:budget]

def format_context_for_llm(hits: List[Dict[str, Any]], budget_chars: int) -> str:
    blocks = []
    used = 0
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        header = f"[DOC {i}] ElementId={h['id']}  Name={meta.get('USCInstanceName','')}  Room={meta.get('USCRoomName','')}-{meta.get('USCRoomNumber','')}"
        body = h["doc"]
        block = header + "\n" + body
        block = safe_truncate(block, max(0, budget_chars - used))
        if not block:
            break
        blocks.append(block)
        used += len(block)
        if used >= budget_chars:
            break
    return "\n\n".join(blocks)

def build_prompt(user_query: str, context: str) -> str:
    return f"""
You are a facility management assistant. Answer the user's question using ONLY the provided documents.
If the answer is not in the documents, say you cannot find it.
Cite the ElementId(s) you used in brackets like [ElementId: 12345].
Prefer concise, actionable steps.

# User Question
{user_query}

# Documents
{context}
""".strip()

# ⭐ UPDATED: accept api_key from UI/env and pass into OpenAI()
def call_openai_llm(system_prompt: str, user_prompt: str, model: str, temperature: float, use_responses_api: bool, api_key: str) -> str:
    if not api_key:
        return "⚠️ Please provide an OpenAI API Key."

    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return f"OpenAI init error: {e}"

    # Optional quick format hint if the key doesn't look right
    if not api_key.startswith("sk-"):
        st.info("Tip: Your API key usually starts with 'sk-'. Please double-check.")

    try:
        if use_responses_api:
            resp = client.responses.create(
                model=model,
                temperature=temperature,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return getattr(resp, "output_text", "") or resp.output[0].content[0].text
        else:
            r = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return r.choices[0].message.content
    except Exception as e:
        return f"OpenAI error: {e}"

# ------------------ UI ------------------
st.title("LLM + RAG for USC Mechanical Equipment")

tab_build, tab_query = st.tabs(["Build / Refresh Index", "Query & LLM Answer"])

with tab_build:
    st.subheader("1) Load data from Postgres")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Load from DB", use_container_width=True):
            with st.spinner("Connecting & loading…"):
                try:
                    df = fetch_df(PG_HOST, PG_DB, PG_USER, PG_PW, PG_PORT, SQL)
                    st.session_state["raw_df"] = df.fillna("")
                    st.success(f"Loaded {len(df)} rows.")
                except Exception as e:
                    st.error(f"DB error: {e}")
    with c2:
        if "raw_df" in st.session_state:
            st.dataframe(st.session_state["raw_df"].head(50), use_container_width=True)
            csv = st.session_state["raw_df"].to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV preview", csv, "mechanical_equipment_preview.csv", use_container_width=True)

    st.divider()
    st.subheader("2) Build / Refresh Chroma Collection")
    if "raw_df" not in st.session_state:
        st.info("Load data first.")
    else:
        df = st.session_state["raw_df"]
        ids = df["ElementId"].astype(str).tolist()
        docs = [row_to_text(r) for r in df.to_dict("records")]
        metas = df.to_dict("records")

        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("Build / Update Index", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading embedding model…"):
                        st_model = load_st_model(model_name)
                        ef = LocalEF(st_model)

                    client = chromadb.PersistentClient(path=persist_path)
                    coll = ensure_collection(client, collection_name, ef, rebuild=rebuild)

                    prog = st.progress(0.0, text="Indexing…")
                    total = len(docs)
                    for i in range(0, total, batch_size):
                        coll.add(
                            ids=ids[i:i+batch_size],
                            documents=docs[i:i+batch_size],
                            metadatas=metas[i:i+batch_size],
                        )
                        prog.progress(min(1.0, (i + batch_size) / total), text=f"Indexed {min(total, i+batch_size)}/{total}")
                    prog.empty()
                    st.success(f"Inserted {total} documents into '{collection_name}' at {persist_path}.")
                except Exception as e:
                    st.error(f"Index build error: {e}")
        with col2:
            st.code(
                f"Persist path: {persist_path}\nCollection: {collection_name}\nST model: {model_name}\nRebuild: {rebuild}\nRows: {len(docs)}",
                language="bash",
            )

with tab_query:
    st.subheader("Semantic Search + LLM Answer (optional)")
    q = st.text_input("Query", placeholder="e.g., Where is FCU 2-8 located?")
    n_results = st.slider("Top-k", 1, 20, 6)

    st.caption("Optional metadata filters:")
    filt_col1, filt_col2, filt_col3, filt_col4 = st.columns(4)
    with filt_col1: f_name = st.text_input("USCInstanceName")
    with filt_col2: f_room = st.text_input("USCRoomName")
    with filt_col3: f_room_no = st.text_input("USCRoomNumber")
    with filt_col4: f_manuf = st.text_input("USCDesignManufacturer")

    gen_llm = st.checkbox("Generate LLM Answer (RAG)", value=True if use_llm else False)

    run = st.button("Run", type="primary")
    if run:
        if not q:
            st.warning("Enter a query.")
        else:
            try:
                # Query Chroma
                st_model = load_st_model(model_name)
                ef = LocalEF(st_model)
                client = chromadb.PersistentClient(path=persist_path)
                coll = ensure_collection(client, collection_name, ef, rebuild=False)

                where: Dict[str, Any] = {}
                if f_name:    where["USCInstanceName"] = f_name
                if f_room:    where["USCRoomName"] = f_room
                if f_room_no: where["USCRoomNumber"] = f_room_no
                if f_manuf:   where["USCDesignManufacturer"] = f_manuf
                where = where or None

                res = coll.query(query_texts=[q], n_results=n_results, where=where)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                ids   = res.get("ids", [[]])[0]
                dists = res.get("distances", [[]])[0] if "distances" in res else [None]*len(docs)

                if not docs:
                    st.info("No results.")
                else:
                    # Show hits
                    st.write("#### Retrieval Results")
                    for i, (doc, meta, _id, dist) in enumerate(zip(docs, metas, ids, dists), 1):
                        with st.expander(f"Result {i}: ElementId {_id}", expanded=(i==1)):
                            st.write(doc)
                            left, right = st.columns(2)
                            with left:
                                st.markdown("**Metadata**")
                                show = {
                                    "USCInstanceName": meta.get("USCInstanceName"),
                                    "USCRoomName": meta.get("USCRoomName"),
                                    "USCRoomNumber": meta.get("USCRoomNumber"),
                                    "USCDesignManufacturer": meta.get("USCDesignManufacturer"),
                                    "USCSerialNumber": meta.get("USCSerialNumber"),
                                    "USCInstallWarrantyDuration": meta.get("USCInstallWarrantyDuration"),
                                    "TypeId": meta.get("TypeId"),
                                }
                                st.json(show)
                            with right:
                                st.markdown("**Quick Links**")
                                mlink = meta.get("MeridianLink")
                                vlink = meta.get("BIM3Dview")
                                st.markdown(f"- Meridian: [{mlink}]({mlink})" if mlink else "- Meridian: _N/A_")
                                st.markdown(f"- BIM 3D View: [{vlink}]({vlink})" if vlink else "- BIM 3D View: _N/A_")

                    # LLM answer
                    if gen_llm and use_llm:
                        if not openai_api_key:
                            st.warning("Please enter your OpenAI API Key in the sidebar.")
                        else:
                            st.divider()
                            st.write("### LLM Answer (grounded by retrieved docs)")
                            hits = [{"doc": d, "meta": m, "id": _id} for d, m, _id in zip(docs, metas, ids)]
                            ctx = format_context_for_llm(hits, max_context_chars)
                            system_prompt = "You are a precise facility management assistant."
                            user_prompt = build_prompt(q, ctx)
                            try:
                                with st.spinner("Calling OpenAI…"):
                                    answer = call_openai_llm(
                                        system_prompt, user_prompt,
                                        model=llm_model,
                                        temperature=temperature,
                                        use_responses_api=use_responses_api,
                                        api_key=openai_api_key,   # ⭐ pass UI-provided key
                                    )
                                st.write(answer)
                                st.caption(f"Model: {llm_model}  •  Temperature: {temperature}  •  API: {'Responses' if use_responses_api else 'Chat Completions'}")
                            except Exception as e:
                                st.error(f"OpenAI error: {e}")
            except Exception as e:
                st.error(f"Query error: {e}")

st.caption("Tip: API key entered in the sidebar is only kept in memory for this session.")

