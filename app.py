
import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from gtts import gTTS

# Load translation model
model_name = 'Helsinki-NLP/opus-mt-hi-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translation function
def translate_hindi_to_english(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translation = model.generate(**tokens)
    english_text = tokenizer.decode(translation[0], skip_special_tokens=True)
    return english_text

# Streamlit UI
st.title("🧠 Multilingual Learning Assistant")
st.write("Translate Hindi ➜ English and Listen")

hindi_text = st.text_area("✍️ Enter text in Hindi:")

if st.button("🔄 Translate"):
    if hindi_text.strip():
        english_text = translate_hindi_to_english(hindi_text)
        st.success("✅ Translated English:")
        st.write(english_text)

        tts = gTTS(english_text)
        tts.save("translated_audio.mp3")
        st.audio("translated_audio.mp3", format="audio/mp3")
