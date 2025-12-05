# app.py
import os
import re
from datetime import datetime
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# ----------------------------
# Env & Config
# ----------------------------
load_dotenv()
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "")

st.set_page_config(page_title="AI Test Design Assistant", page_icon="✅", layout="wide")

# Sidebar: API key + model config
with st.sidebar:
    st.title("⚙️ Settings")
    api_key = st.text_input(
        "OpenAI API Key",
        value=DEFAULT_API_KEY,
        type="password",
        help="Pulled from .env if present. Overrides accepted."
    )
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    save_files = st.checkbox("Save outputs to server folder", value=False)
    save_dir = st.text_input("Save directory", value="outputs")
    st.caption("Outputs are also available as downloads from the UI.")

# Guard: API key
if not api_key:
    st.warning("Please provide an OpenAI API key in the sidebar.")
    st.stop()

# ----------------------------
# LLM init (cached)
# ----------------------------
@st.cache_resource(show_spinner=False)
def get_llm(_api_key: str, _model: str, _temperature: float):
    return ChatOpenAI(api_key=_api_key, model=_model, temperature=_temperature)

llm = get_llm(api_key, model, temperature)

# ----------------------------
# Prompts
# ----------------------------
TEST_CASE_PROMPT = PromptTemplate(
    input_variables=["requirement"],
    template=(
        "You are a senior QA engineer. Given this software requirement:\n\n"
        "\"\"\"\n{requirement}\n\"\"\"\n\n"
        "Generate a set of **comprehensive** test cases including:\n"
        "- Functional test cases\n- Edge cases\n- Negative test cases\n\n"
        "Use the following **strict structure** per test case (do not skip fields):\n"
        "```\n"
        "Test Case ID: TC-XXX\n"
        "Title: <short name>\n"
        "Preconditions: <list or N/A>\n"
        "Steps:\n"
        "1) ...\n2) ...\n"
        "Expected Result: ...\n"
        "Priority: High/Medium/Low\n"
        "Traceability: <requirement reference or N/A>\n"
        "```\n"
        "Finish with a small **coverage summary**."
    )
)

ARTIFACT_PROMPT = PromptTemplate(
    input_variables=["requirement"],
    template=(
        "You are a senior QA lead. Based on the requirement below, produce the following **test artifacts** "
        "in **Markdown**:\n\n"
        "\"\"\"\n{requirement}\n\"\"\"\n\n"
        "1. **Test Plan**: objectives, scope (in/out), assumptions, responsibilities, entry/exit criteria, risks & mitigation.\n"
        "2. **User Scenarios**: 6–10 realistic end-to-end scenarios (happy path + alternatives).\n"
        "3. **Test Data**: a table of representative inputs (valid/invalid/edge) with rationale.\n"
        "4. **Acceptance Criteria**: bullet list, clear and measurable.\n"
        "Ensure formatting is clean and skimmable."
    )
)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)
# ----------------------------
# Core generators
# ----------------------------
def generate_with_prompt(prompt: PromptTemplate, requirement: str) -> str:
    completion = llm.invoke([{"role": "user", "content": prompt.format(requirement=requirement)}])
    return completion.content

def save_to_file(filename: str, content: str, folder: Optional[str] = None) -> str:
    folder = folder or "."
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path

# ----------------------------
# UI – Input
# ----------------------------
st.title("🧪 AI Test Design Assistant")

col_left, col_right = st.columns([2, 1], gap="large")

with col_left:
    st.subheader("Requirement")
    requirement_text = st.text_area(
        "Paste your requirement",
        height=220,
        placeholder="Describe the feature or paste your PRD/US here...",
    )
    uploaded = st.file_uploader("Or upload a .txt / .md file", type=["txt", "md"])
    if uploaded and not requirement_text:
        requirement_text = uploaded.read().decode("utf-8", errors="ignore")

with col_right:
    st.subheader("Generate")
    gen_cases = st.checkbox("Generate Test Cases", value=True)
    gen_artifacts = st.checkbox("Generate Test Artifacts (Plan, Scenarios, Data, ACs)", value=True)
    st.caption("Tip: You can run either or both.")

# ----------------------------
# Actions
# ----------------------------
run = st.button("🚀 Generate")

if run:
    if not requirement_text or len(requirement_text.strip()) == 0:
        st.error("Please provide a requirement.")
        st.stop()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs = {}

    try:
        with st.spinner("Thinking… generating outputs with the LLM"):
            if gen_cases:
                cases = generate_with_prompt(TEST_CASE_PROMPT, requirement_text)
                outputs["Test Cases (Markdown/Text)"] = cases

            if gen_artifacts:
                artifacts = generate_with_prompt(ARTIFACT_PROMPT, requirement_text)
                outputs["Test Artifacts (Markdown)"] = artifacts

    except Exception as e:
        st.exception(e)
        st.stop()

    # ----------------------------
    # Display & Downloads
    # ----------------------------
    st.success("Done! Scroll to view results. You can download or save locally.")

    for label, content in outputs.items():
        st.markdown(f"### {label}")
        # Render markdown if it looks like markdown; otherwise plain text
        st.markdown(content)

        # Downloads
        default_ext = ".md" if "Artifacts" in label else ".md"
        file_bytes = content.encode("utf-8")
        st.download_button(
            f"⬇️ Download {label}",
            data=file_bytes,
            file_name=f"{label.replace(' ', '_').lower()}_{ts}{default_ext}",
            mime="text/markdown",
        )

        # Optional server save
        if save_files:
            raw_fname = f"{label}_{ts}{default_ext}"
            fname = sanitize_filename(raw_fname)
            #fname = f"{label.replace(' ', '_').lower()}_{ts}{default_ext}"
            path = save_to_file(fname, content, folder=save_dir)
            st.caption(f"Saved to `{path}`")

# Footer
st.divider()
st.caption("Built with Streamlit + LangChain OpenAI. No data is persisted unless you enable saving.")

