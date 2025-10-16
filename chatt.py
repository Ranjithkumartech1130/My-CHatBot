import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import speech_recognition as sr
from google.generativeai.types import GenerationConfig, Tool
import io

# --- Gemini API Configuration ---
try:
    # It's highly recommended to set your API key as a Streamlit secret
    # for security and ease of deployment.
    # In your Streamlit Cloud dashboard, add a secret with the key "GEMINI_API_KEY".
    # This correctly fetches the key from an environment variable or a Streamlit secret.
    # The `st.secrets.load_if_toml_exists()` part gracefully handles the absence of a secrets file.
    api_key = os.environ.get("GEMINI_API_KEY")
    st.secrets.load_if_toml_exists()
    if not api_key and st.secrets.get("GEMINI_API_KEY"):
        api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found.")
    genai.configure(api_key=api_key)
except (ValueError, TypeError) as e:
    st.error(
        "🚨 Gemini API key not found. "
        "Please set the `GEMINI_API_KEY` environment variable or add it to your Streamlit secrets.",
        icon="🚨"
    )
    st.stop()

# --- Model Selection and Initialization ---
# A system instruction to guide the chatbot's behavior
SYSTEM_PROMPT = (
    "You were developed by 'Lord Ranjith Kumar'. "
    "When a user asks to create an image, call the image generation tool immediately without unnecessary conversation. "
    "You are a helpful, friendly, and highly intelligent AI assistant. "
    "Your capabilities are vast, covering fields like technology, programming, science, education, business, and general knowledge. "
    "When responding, always provide clear, accurate, and easy-to-understand answers. "
    "Use simple language, offer step-by-step explanations, and include examples when it helps with clarity. "
    "Maintain a polite and professional tone. "
    "If a question is outside your knowledge or you are uncertain, state it honestly rather than providing speculative information."
)

# Use the latest and most capable flash model
MODEL_NAME = "gemini-2.5-flash"

# --- Define Image Generation Tool ---
image_generation_tool = Tool(
    function_declarations=[
      genai.protos.FunctionDeclaration(
          name="generate_images",
          description="Create images from a text prompt.",
          parameters=genai.protos.Schema(
              type=genai.protos.Type.OBJECT,
              properties={
                  "prompt": genai.protos.Schema(type=genai.protos.Type.STRING, description="The text prompt to generate images from.")
              }
          )
      )
    ])

model = genai.GenerativeModel(
    MODEL_NAME,
    system_instruction=SYSTEM_PROMPT,
    tools=[image_generation_tool]
)

# --- Streamlit App UI ---
st.set_page_config(page_title="Ranjith's client", page_icon="🤖")

st.title("🤖 Ranjith's Client chatbot")
st.caption(f"Powered by Google Gemini {MODEL_NAME}")

# Initialize chat session in Streamlit's session state
def get_or_init_session_state(key, default_value):
    """Gets a value from session state or initializes it."""
    if key not in st.session_state:
        st.session_state[key] = default_value
    return st.session_state[key]

get_or_init_session_state("messages", [
    {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}
])
get_or_init_session_state("chat_session", model.start_chat(history=[]))

# --- Core Functions ---
def handle_prompt(prompt_parts):
    """Handles user prompt, displays it, gets a response, and updates the chat."""
    # Display user message
    with st.chat_message("user", avatar="🧑‍💻"):
        for part in prompt_parts:
            if isinstance(part, str):
                st.markdown(part)
            else:
                st.image(part, width=200)

    st.session_state.messages.append({"role": "user", "content": prompt_parts})

    # Get and display assistant response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chat_session.send_message(
                    prompt_parts,
                    # Use the tool if the model decides to
                    tool_config=genai.protos.ToolConfig(
                        function_calling_config=genai.protos.FunctionCallingConfig(
                            mode=genai.protos.FunctionCallingConfig.Mode.AUTO
                        )
                    )
                )

                # Check if the first part of the response has a function call
                first_part = response.candidates[0].content.parts[0]
                has_function_call = hasattr(first_part, "function_call")

                # Check for tool calls and display images
                if has_function_call and first_part.function_call.name == "generate_images":
                    st.info("Generating images based on your prompt...")
                    display_generated_images(response)
                else:
                    bot_reply = response.text
                    st.markdown(bot_reply)
                    # Add assistant text response to session state
                    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            except Exception as e:
                st.error(f"An error occurred: {e}")
                bot_reply = "Sorry, I ran into a problem. Please try again."
                st.markdown(bot_reply)
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})

def get_voice_input():
    """Captures and transcribes voice input."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!", icon="🎤")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.info("Transcribing...", icon="✍️")
            text = recognizer.recognize_google(audio)
            st.success(f"You said: \"{text}\"")
            return text
        except sr.WaitTimeoutError:
            st.warning("No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio. Please speak clearly.")
        except sr.RequestError as e:
            st.error(f"Could not request results from Google Speech Recognition service; {e}")
    return None

def display_generated_images(response):
    """Displays images from a tool call response."""
    # The response contains the function call and the generated images.
    # We just need to find the image parts and display them.
    image_parts = [part for part in response.candidates[0].content.parts if part.inline_data.mime_type.startswith("image/")]
    
    if image_parts:
        # Extract image data for display and for session state
        images_for_display = [Image.open(io.BytesIO(p.inline_data.data)) for p in image_parts]
        st.image(images_for_display)

        # Add the generated images to the chat history for display on rerun
        # Storing raw bytes is better for serialization in session state
        st.session_state.messages.append({"role": "assistant", "content": [{"type": "image", "data": p.inline_data.data} for p in image_parts]})

# --- Sidebar ---
with st.sidebar:
    st.header("Controls")
    st.markdown("Your intelligent AI assistant for any query.")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}
        ]
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
        content = message["content"]
        if isinstance(content, list):
            for part in content:
                # The PIL.Image.Image type is not directly serializable for session state
                # A better approach for reruns would be to store file bytes and reload
                if isinstance(part, dict) and "type" in part and part["type"] == "image":
                     st.image(io.BytesIO(part["data"]), width=200)
                elif hasattr(part, "inline_data"): # Handle google.generativeai.protos.Part
                    st.image(io.BytesIO(part.inline_data.data), width=200)
                elif isinstance(part, str):
                    st.markdown(part)
        else:
            st.markdown(content)

# --- Chat Input and Actions ---
col1, col2 = st.columns([0.85, 0.15])

with col1:
    uploaded_files = st.file_uploader(
        "Upload files",
        accept_multiple_files=True,
        type=["png", "jpg", "jpeg", "txt", "md", "py", "csv"],
        label_visibility="collapsed"
    )

with col2:
    if st.button("🎤", use_container_width=True, help="Speak to Assistant"):
        voice_prompt = get_voice_input()
        if voice_prompt:
            # We need to handle the prompt and then rerun to process it
            st.session_state.voice_prompt = voice_prompt
            st.rerun()

if text_prompt := st.chat_input("Your message..."):

    if not text_prompt and not uploaded_files:
        st.warning("Please enter a message or upload a file.")
    else:
        prompt_parts = []
        for file in uploaded_files:
            if file.type.startswith("image/"):
                img = Image.open(file)
                prompt_parts.append(img)
                # To allow reruns, store image bytes in session state
                buffered = io.BytesIO()
                img.save(buffered, format=img.format or "PNG")
                st.session_state.messages.append({"role": "user", "content": [{"type": "image", "data": buffered.getvalue()}]})
            else:
                # For text-based files
                try:
                    file_content = file.getvalue().decode("utf-8")
                    prompt_parts.append(f"Content of `{file.name}`:\n```\n{file_content}\n```")
                except Exception as e:
                    st.error(f"Error reading file {file.name}: {e}")

        if text_prompt:
            prompt_parts.append(text_prompt)

        if prompt_parts:
            handle_prompt(prompt_parts)


# Handle voice prompt after a rerun
if "voice_prompt" in st.session_state and st.session_state.voice_prompt:
    prompt_to_handle = st.session_state.voice_prompt
    st.session_state.voice_prompt = None # Clear it after use
    handle_prompt([prompt_to_handle])
