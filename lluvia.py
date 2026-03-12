import streamlit as st
from groq import Groq
import os
import base64
from io import BytesIO
from PIL import Image

# Clave API desde secrets
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

st.set_page_config(page_title="Lluvia", layout="wide")
st.title("Lluvia – Elegancia, Dinamismo e Inteligencia")

# Prompt de sistema mejorado (dinámica, didáctica, elegante + presentación)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "Eres Lluvia, una IA excepcionalmente elegante, dinámica y profundamente didáctica. "
                "Al iniciar cualquier conversación nueva, preséntate de forma refinada, cálida y sofisticada, "
                "explicando brevemente tu propósito: asistir con inteligencia, claridad y valor educativo. "
                "Tus respuestas son claras, estructuradas paso a paso, con ejemplos prácticos, analogías inteligentes y preguntas reflexivas que enriquecen el aprendizaje. "
                "Mantienes un tono culto, preciso y encantador, adaptándote al nivel del usuario. "
                "Sé concisa cuando corresponde y exhaustiva cuando se requiere profundidad. "
                "Nunca uses emojis ni lenguaje coloquial. Prioriza la elegancia, la claridad y el valor educativo en cada respuesta."
            )
        }
    ]

# Mostrar historial
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input con botón de voz y upload de imagen
col1, col2 = st.columns([8, 1])
with col1:
    prompt = st.chat_input("Escribe o habla tu consulta...")

with col2:
    if st.button("🎤 Hablar"):
        st.info("Usa el dictado por voz del navegador (micrófono en la barra de direcciones o tecla del teclado). "
                "Habla normalmente y el texto aparecerá en el cuadro de arriba. Luego presiona Enter para enviar.")

# Upload de imagen
uploaded_file = st.file_uploader("Sube una imagen para que Lluvia la analice o describa", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir imagen a base64 para enviar a Groq
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Prompt para análisis de imagen
    image_prompt = "Describe detalladamente esta imagen de forma elegante y didáctica. Explica elementos clave, composición, colores, emociones transmitidas y posibles interpretaciones o usos prácticos."
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": image_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]}
        ],
        model="llama-3.2-vision-11b",  # Modelo de visión (ajusta si Groq tiene otro en 2026)
        temperature=0.7,
        max_tokens=800,
    )
    st.session_state.messages.append({"role": "user", "content": "[Imagen subida y analizada]"})
    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
    st.chat_message("assistant").markdown(response.choices[0].message.content)

# Chat normal
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            messages=st.session_state.messages,
            model="llama-3.3-70b-versatile",  # Modelo principal potente
            temperature=0.7,
            max_tokens=1500,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # Generación de imágenes si el usuario lo pide
        if any(palabra in prompt.lower() for palabra in ["genera imagen", "crea imagen", "imagen de", "dibuja", "ilustra", "generar imagen"]):
            st.info("Generando imagen... (puede tardar 10-30 segundos)")
            try:
                image_response = client.images.generate(
                    model="flux.1",  # Flux.1 o el modelo de imagen disponible en Groq
                    prompt=prompt,
                    size="1024x1024",
                    quality="standard",
                    n=1,
                )
                image_url = image_response.data[0].url
                st.image(image_url, caption="Imagen generada por Lluvia")
            except Exception as e:
                st.error("No pude generar la imagen en este momento. Prueba describiendo más detalladamente o usa otro prompt.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
