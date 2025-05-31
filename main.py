from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, RunConfig
import asyncio
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Translator app")

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("API key not available. Please put API key in env file")

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

user_input = st.text_area("Enter your text:")

language = st.selectbox(label="Select target language", options=["Urdu","French","German"])

agent = Agent(
    name="Translator agent",
    instructions=f"You are a translator agent. You have the task of translating the given English text into {language}",
    model=model
)

if st.button("Translate") and user_input.strip():
    async def translate():
        return await Runner.run(agent, input=user_input, run_config=config)

    # Run the async function safely in Streamlit
    result = asyncio.run(translate())

    st.success(f"Translated to {language}")
    st.header("Translated Text")
    st.text(result.final_output)