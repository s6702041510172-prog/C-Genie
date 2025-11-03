import os
import re
import fitz  
import difflib
import streamlit as st
import google.generativeai as genai
import time
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv

# -------------------------------
# CONFIG API
# -------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY", "AIzaSyBlaAYDZu2yhYlaShDnZoMoCkBA0lSGoaE"))

generation_config = {
    "temperature": 0.35,
    "top_p": 0.9,
    "top_k": 64,
    "max_output_tokens": 1600,
    "response_mime_type": "text/plain"
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

PROMPT_SYSTEM = """
ROLE:
คุณคือ Chatbot ชื่อ “C-Genie” ทำหน้าที่เป็นผู้ช่วยสอนภาษา C
ทำงานในฐานะติวเตอร์ออนไลน์ที่มีความรู้เชิงลึกเกี่ยวกับภาษา C
มีบุคลิกสุภาพ อธิบายอย่างเป็นขั้นตอน เข้าใจง่าย และอ้างอิงเฉพาะจากเอกสารที่ผู้ใช้กำหนดเท่านั้น

OBJECTIVE:
- สนับสนุนการเรียนรู้ภาษา C ตามหลักการเรียนรู้เชิงรุก (Active Learning)
- ช่วยให้นักศึกษาทำความเข้าใจแนวคิดสำคัญ เช่น ตัวแปร ฟังก์ชัน เงื่อนไข ลูป และอาร์เรย์
- ให้คำแนะนำ คำอธิบาย และตัวอย่างโค้ดที่ถูกต้องตามหลักวิชาการ

CONSTRAINTS:
1. อ้างอิงข้อมูลจากเอกสาร (PDF) ที่ผู้ใช้กำหนดเท่านั้น
2. ห้ามแต่งเติมหรือสร้างเนื้อหาที่อยู่นอกเหนือจากข้อมูลในเอกสาร
3. หากไม่มีข้อมูลให้ตอบเพียงว่า “ข้อมูลนี้ยังไม่มีในเอกสารค่ะ”
4. ห้ามตอบในเชิงความคิดเห็น หรือคาดเดา
5. ใช้ภาษาที่สุภาพ ชัดเจน และเหมาะสมกับนักศึกษาระดับอุดมศึกษา

INSTRUCTION FOR RESPONSE:
- หากคำถามมีหลายประเด็น ให้จัดเรียงคำตอบเป็นข้อ ๆ อย่างเป็นระบบ
- หากมีตัวอย่างโค้ด ให้แสดงในรูปแบบ Markdown:
  ```c
  // ตัวอย่างโค้ด

"""

model = genai.GenerativeModel(
    model_name="models/gemini-2.5-flash",
    safety_settings=SAFETY_SETTINGS,
    generation_config=generation_config,
    system_instruction=PROMPT_SYSTEM,
)

# -------------------------------
# อ่าน PDF (ด้วย fitz)
# -------------------------------
@st.cache_data(show_spinner=False)
def load_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf:
        for page in pdf:
            text += page.get_text("text") + "\n\n"
    return text.strip()

# -------------------------------
# แบ่งและค้นข้อมูลจาก PDF
# -------------------------------
def split_chunks(text, size=1200, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def search_chunks(query, chunks, top_k=8):
    """
    ค้นหา chunk ที่ใกล้เคียงกับคำถาม
    พร้อม boost คำสำคัญ เช่น 'ภาษา C', 'ฟังก์ชัน', 'ตัวแปร', 'โครงสร้าง'
    """
    boost_keywords = ["ภาษา C", "ฟังก์ชัน", "ตัวแปร", "โครงสร้าง", "พัฒนา", "เกิดขึ้น", "ค.ศ.", "Ritchie", "BCPL"]
    scored = []

    for c in chunks:
        score = difflib.SequenceMatcher(None, query.lower(), c.lower()).ratio()
        for kw in boost_keywords:
            if kw in c:
                score += 0.08  # เพิ่มน้ำหนักหากเจอคำสำคัญ
        scored.append((score, c))

    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]

# -------------------------------
# ฟังก์ชันตอบคำถาม
# -------------------------------
def generate_response(prompt, file_content, user_prompt_addon, chat_key):
    chunks = split_chunks(file_content)
    related = search_chunks(prompt, chunks, top_k=8)
    context = "\n\n".join(related)

    # รวมประวัติห้อง
    history_text = ""
    for msg in st.session_state["chats"][chat_key]:
        role = "ผู้ใช้" if msg["role"] == "user" else "C-Genie"
        history_text += f"{role}: {msg['content']}\n"

    query = f"""
ข้อมูลจากเอกสาร:
{context}

ประวัติการสนทนา:
{history_text}

คำถามล่าสุด:
{prompt}

คำสั่งเพิ่มเติม:
{user_prompt_addon}
"""

    response = model.generate_content(query)
    return response.text.strip() if response and response.candidates and response.candidates[0].content.parts else "ข้อมูลนี้ยังไม่มีในเอกสารค่ะ"

# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="C-Genie", page_icon="🧞", layout="wide")
st.title("🧞 C-Genie — ผู้ช่วยสอนภาษา C")

# -------------------------------
# Sidebar: ห้องแชท + prompt เพิ่ม
# -------------------------------
with st.sidebar:
    st.markdown("### 💬 จัดการห้องสนทนา")

    if "chats" not in st.session_state:
        st.session_state["chats"] = {"ห้องเริ่มต้น": []}
    if "active_chat" not in st.session_state:
        st.session_state["active_chat"] = "ห้องเริ่มต้น"

    chat_names = list(st.session_state["chats"].keys())
    chat_key = st.selectbox("เลือกห้องแชท", chat_names, index=chat_names.index(st.session_state["active_chat"]))
    st.session_state["active_chat"] = chat_key

    new_chat = st.text_input("➕ สร้างห้องใหม่", placeholder="เช่น ฟังก์ชัน, ตัวแปร, Loop ...")
    if st.button("เพิ่มห้องใหม่"):
        if new_chat.strip():
            if new_chat not in st.session_state["chats"]:
                st.session_state["chats"][new_chat] = []
                st.session_state["active_chat"] = new_chat
                st.rerun()
            else:
                st.warning("⚠️ มีชื่อห้องนี้อยู่แล้ว")
        else:
            st.warning("⚠️ กรุณากรอกชื่อห้องก่อน")

    if st.button("🗑️ ลบห้องนี้"):
        if chat_key != "ห้องเริ่มต้น":
            del st.session_state["chats"][chat_key]
            st.session_state["active_chat"] = "ห้องเริ่มต้น"
            st.rerun()
        else:
            st.warning("❌ ไม่สามารถลบห้องเริ่มต้นได้")

    st.markdown("---")
    user_prompt_addon = st.text_area("💡 เพิ่มคำสั่งเสริมสำหรับบอท (เช่น แนวการตอบ)", height=80)

    if st.button("🧹 ล้างประวัติห้องนี้"):
        st.session_state["chats"][chat_key] = []
        st.rerun()

# -------------------------------
# โหลด PDF
# -------------------------------
file_path = "datasetC.pdf"
if not os.path.exists(file_path):
    st.error("❌ ไม่พบไฟล์ datasetC.pdf")
    st.stop()

file_content = load_pdf(file_path)

# -------------------------------
# แนะนำบทเรียนเริ่มต้น
# -------------------------------
if len(st.session_state["chats"][chat_key]) == 0:
    st.markdown("""
    ### 👋 ยินดีต้อนรับสู่ **C-Genie – ผู้ช่วยสอนภาษา C**
    ฉันสามารถช่วยคุณอธิบายและตอบคำถามจากเอกสารได้ทุกบท เช่น:
    1️⃣ แนะนำภาษา C  
    2️⃣ ตัวแปรและชนิดข้อมูล  
    3️⃣ ตัวดำเนินการและนิพจน์  
    4️⃣ คำสั่งควบคุม  
    5️⃣ ฟังก์ชันและอาร์เรย์  
    """)

# -------------------------------
# แสดงบทสนทนาในห้องปัจจุบัน
# -------------------------------
st.subheader(f"💬 ห้องสนทนา: {chat_key}")
for msg in st.session_state["chats"][chat_key]:
    st.chat_message(msg["role"]).write(msg["content"])

# -------------------------------
# ช่องแชทหลัก
# -------------------------------
if prompt := st.chat_input(f"พิมพ์คำถามใน {chat_key} ..."):
    st.chat_message("user").write(prompt)
    st.session_state["chats"][chat_key].append({"role": "user", "content": prompt})

    with st.chat_message("model"):
        with st.spinner("🔍 กำลังค้นข้อมูลจากเอกสาร..."):
            reply = generate_response(prompt, file_content, user_prompt_addon, chat_key)
            message_placeholder = st.empty()
            displayed = ""
            for char in reply:
                displayed += char
                message_placeholder.markdown(displayed)
                time.sleep(0.004)

    st.session_state["chats"][chat_key].append({"role": "model", "content": reply})
