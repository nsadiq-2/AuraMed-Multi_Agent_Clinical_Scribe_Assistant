import os
import streamlit as st
from groq import Groq

# 1. Page Configuration
st.set_page_config(
    page_title="AuraMed AI - Agentic Scribe", page_icon="🩺", layout="wide"
)

# 2. Secure API Key Initialization
# Hugging Face Spaces reads from Settings -> Secrets. Locally/Colab reads from env or input.
if "GROQ_API_KEY" in os.environ:
    groq_api_key = os.environ["GROQ_API_KEY"]
else:
    # Fallback for local testing or Colab
    groq_api_key = st.sidebar.text_input(
        "Enter Groq API Key", type="password"
    )

# Instantiate Groq Client
client = Groq(api_key=groq_api_key) if groq_api_key else None

# 3. App Architecture & Header
st.title("🩺 AuraMed AI: Agentic Scribe & Safety Assistant")
st.markdown(
    "Transform unstructured patient consultation notes into structured electronic health records with an automated agent safety evaluation layer."
)

if not client:
    st.warning("⚠️ Please provide a Groq API Key to proceed.")
    st.stop()

# 4. UI Layout Splits
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Raw Consultation Ingest")
    raw_notes = st.text_area(
        "Paste ambient dialogue transcripts or hasty doctor notes here:",
        height=300,
        placeholder="Example:\nPatient John Doe, 45, presents with acute lower back pain for 3 days. Rated 7/10. No radiation. Past medical history of hypertension. Currently taking Lisinopril. Appears distressed while sitting. Advised Ibuprofen 400mg every 6 hours and dynamic stretching. Follow up if pain persists.",
    )
    submit_btn = st.button("Run Multi-Agent Processing", type="primary")

# 5. Execution Pipeline
if submit_btn and raw_notes:
    with col2:
        # --- AGENT 1: Clinical Scribe ---
        with st.status("Agent 1: Structuring Clinical Records...", expanded=True) as status:
            try:
                scribe_prompt = (
                    "You are an expert Clinical Medical Scribe. Analyze the following raw medical notes "
                    "and structure them into a formal SOAP Note format (Subjective, Objective, Assessment, Plan). "
                    "Maintain precise medical terminology.\n\n"
                    f"Raw Notes:\n{raw_notes}"
                )
                
                # Using fast Llama 3.3 70B model via Groq
                scribe_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": scribe_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.2,
                )
                
                structured_soap = scribe_response.choices[0].message.content
                status.update(label="Agent 1: SOAP Note Generated!", state="complete")
                
                # Render results
                st.subheader("🩺 Structured SOAP Record")
                st.markdown(structured_soap)
                
            except Exception as e:
                st.error(f"Scribe Agent Failed: {e}")
                st.stop()
                
        st.write("---")

        # --- AGENT 2: Safety & Cross-Reference Agent ---
        with st.status("Agent 2: Running Safety Audit...", expanded=True) as status:
            try:
                safety_prompt = (
                    "You are a Clinical Pharmacist and Medical Safety Agent. Review the following "
                    "Structured SOAP medical note. Identify any potential drug interactions, allergen risks, "
                    "omitted critical checks, or red flags. Give a distinct 'Risk Assessment Level' (Low/Medium/High) "
                    "followed by bullet points explaining your safety evaluation.\n\n"
                    f"Structured Record:\n{structured_soap}"
                )
                
                safety_response = client.chat.completions.create(
                    messages=[{"role": "user", "content": safety_prompt}],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1,
                )
                
                safety_evaluation = safety_response.choices[0].message.content
                status.update(label="Agent 2: Safety Check Finalized!", state="complete")
                
                # Render results
                st.subheader("🛡️ Safety & Interaction Intelligence")
                st.markdown(safety_evaluation)
                
            except Exception as e:
                st.error(f"Safety Agent Failed: {e}")