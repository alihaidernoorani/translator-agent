from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import asyncio
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("❌ API key not found. Please set it in your .env file.")
    st.stop()

# Streamlit page config
st.set_page_config(page_title="🌐 AI Translator", page_icon="🌍", layout="centered")
st.markdown(
    "<h1 style='text-align: center'>🌐 AI Translator</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Translate text into 30+ languages using AI</p>",
    unsafe_allow_html=True,
)

# Set up AI model
external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)
config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Language options
LANGUAGES = sorted([
    "Arabic", "Bengali", "Chinese (Simplified)", "Chinese (Traditional)", "Dutch",
    "English", "French", "German", "Greek", "Gujarati", "Hebrew", "Hindi", "Italian",
    "Japanese", "Kannada", "Korean", "Malay", "Marathi", "Persian", "Polish",
    "Portuguese", "Punjabi", "Russian", "Spanish", "Swahili", "Tamil", "Telugu",
    "Thai", "Turkish", "Urdu", "Vietnamese"
])

# Input and language selection
with st.form("translate_form"):
    user_input = st.text_area("✏️ Enter your text below", height=160)
    target_language = st.selectbox("🌍 Select target language", options=LANGUAGES)
    submitted = st.form_submit_button("🔁 Translate")

if submitted and user_input.strip():
    with st.spinner("Translating..."):
        agent = Agent(
            name="Translator agent",
            instructions=f"You are a translator. Translate the text into {target_language}.",
            model=model
        )

        async def translate():
            return await Runner.run(agent, input=user_input, run_config=config)

        try:
            result = asyncio.run(translate())
            st.success(f"✅ Translation to {target_language} complete!")
            st.text_area("📝 Translated Output", value=result.final_output, height=160)
        except Exception as e:
            st.error(f"⚠️ An error occurred during translation: {e}")
