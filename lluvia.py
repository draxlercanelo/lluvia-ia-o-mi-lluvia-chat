import streamlit as st
from groq import Groq
import os

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)

st.set_page_config(page_title="Lluvia", layout="wide")
st.title("Lluvia – IA completa e inteligente")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": (
            "Eres Lluvia, una IA ingeniosa, hábil e inteligente. "
            "Responde siempre de manera formal, profesional y precisa, en español natural y fluido. "
            "Mantén un tono educado y respetuoso en todo momento. "
            "Tienes amplios conocimientos generales y acceso a herramientas para información actualizada. "
            "Sé concisa cuando sea posible, pero completa cuando sea necesario. "
            "Nunca uses emojis ni expresiones coloquiales. Prioriza exactitud y claridad."
        )}
    ]

for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Escriba su consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            messages=st.session_state.messages,
            model="openai/gpt-oss-120b",  # El más potente disponible
            temperature=0.6,
            max_tokens=1500,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
