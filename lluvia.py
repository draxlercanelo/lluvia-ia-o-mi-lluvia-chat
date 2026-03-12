import streamlit as st
from groq import Groq
import os
import base64
from io import BytesIO
from PIL import Image
import replicate

# Claves desde secrets (Streamlit Cloud)
groq_api_key = os.getenv("GROQ_API_KEY")
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")

client = Groq(api_key=groq_api_key)

# Configuración de página
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

# Mostrar historial de mensajes
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input + botón de voz
col1, col2 = st.columns([8, 1])
with col1:
    prompt = st.chat_input("Escribe o habla tu consulta...")

with col2:
    if st.button("🎤 Hablar"):
        st.info("Activa el micrófono del navegador (ícono en la barra de direcciones o tecla del teclado). "
                "Habla normalmente y el texto aparecerá en el cuadro. Luego presiona Enter para enviar.")

# Subida de imagen
uploaded_file = st.file_uploader("Sube una imagen para que Lluvia la analice o describa", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)

    # Convertir imagen a base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Análisis de imagen con Groq (visión)
    image_prompt = "Describe detalladamente esta imagen de forma elegante y didáctica. Explica elementos clave, composición, colores, emociones transmitidas y posibles interpretaciones o usos prácticos."
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": image_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
            ]}
        ],
        model="llama-3.2-vision-11b",
        temperature=0.7,
        max_tokens=800,
    )
    st.session_state.messages.append({"role": "user", "content": "[Imagen subida y analizada]"})
    st.session_state.messages.append({"role": "assistant", "content": response.choices[0].message.content})
    st.chat_message("assistant").markdown(response.choices[0].message.content)

# Chat normal + generación de imágenes con Replicate
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        stream = client.chat.completions.create(
            messages=st.session_state.messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1500,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

        # Generación de imágenes con Replicate + Flux.1
        if any(palabra in prompt.lower() for palabra in ["genera imagen", "crea imagen", "imagen de", "dibuja", "ilustra", "generar imagen", "haz una imagen"]):
            if not replicate_api_token:
                st.error("No tienes REPLICATE_API_TOKEN en Secrets. Regístrate en replicate.com y agrégalo para generar imágenes.")
            else:
                st.info("Generando imagen con Flux.1... (puede tardar 10-30 segundos)")
                try:
                    replicate_client = replicate.Client(api_token=replicate_api_token)
                    output = replicate_client.run(
                        "black-forest-labs/flux-schnell",
                        input={
                            "prompt": prompt,
                            "num_inference_steps": 4,
                            "aspect_ratio": "1:1",
                            "output_format": "png"
                        }
                    )
                    image_url = output[0] if output else None
                    if image_url:
                        st.image(image_url, caption="Imagen generada por Lluvia con Flux.1")
                    else:
                        st.warning("No se pudo generar la imagen.")
                except Exception as e:
                    st.error(f"Error al generar imagen: {str(e)}. Verifica tu clave REPLICATE_API_TOKEN en Secrets.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
