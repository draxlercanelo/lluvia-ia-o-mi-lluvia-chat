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

# Prompt de sistema mejorado
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

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

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

        # Cuando piden generar imagen, damos descripción detallada + sugerencia
        if any(palabra in prompt.lower() for palabra in ["genera imagen", "crea imagen", "imagen de", "dibuja", "ilustra", "generar imagen", "haz una imagen"]):
            st.info("No tengo generación directa de imágenes integrada, pero te describo con detalle cómo sería y cómo crearla gratis en segundos:")
            desc_prompt = f"Describe de forma muy detallada y elegante la imagen solicitada: '{prompt}'. "
            desc_prompt += "Incluye composición, colores, iluminación, estilo artístico y emociones para que pueda recrearla fácilmente en una herramienta gratuita."
            desc_response = client.chat.completions.create(
                messages=[{"role": "user", "content": desc_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.7,
                max_tokens=600,
            )
            st.markdown("**Descripción detallada para generar la imagen:**")
            st.markdown(desc_response.choices[0].message.content)
            st.markdown("**Cómo crearla gratis en 1 clic:**")
            st.markdown("1. Ve a https://www.bing.com/images/create (Image Creator de Microsoft, gratis con cuenta Microsoft)")
            st.markdown("2. Pega la descripción que te di arriba")
            st.markdown("3. Genera y descarga la imagen")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
