import streamlit as st
import whisperx
import openai
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set OpenAI API key and configuration
openai.api_key = 'Your-API-KEY'
openai.api_base = 'https://api.openai.com'
openai.api_type = 'openai'
openai.api_version = '2023-03-15-preview'

os.environ["OPENAI_API_TYPE"] = openai.api_type
os.environ["OPENAI_API_VERSION"] = openai.api_version
os.environ["OPENAI_API_BASE"] = openai.api_base
os.environ["OPENAI_API_KEY"] = openai.api_key

device = "cpu"  # Change to "cuda" for GPU
batch_size = 14
compute_type = "int8"
HF_KEY = "YOUR HF TOKEN"

# Load WhisperX model
model = whisperx.load_model("large-v2", device=device, compute_type=compute_type)

# Load T5 model and tokenizer for summarization
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")


def convert_speech_text(audio_file, language_option):
    try:
        language_map = {
            "English": "en",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
        }
        language_code = language_map.get(language_option, "en")

        # Load audio file
        audio = whisperx.load_audio(audio_file)
        if audio is None:
            raise ValueError("Failed to load audio file. Please check the file path and format.")

        # Transcribe audio
        result = model.transcribe(audio, batch_size=batch_size, language=language_code)
        if result is None or "language" not in result:
            raise ValueError("Transcription failed. Please check the audio file and model settings.")
        detected_language = str(result["language"])

        # Load alignment model
        model_a, metadata = whisperx.load_align_model(language_code=language_code, device=device)
        if model_a is None or metadata is None:
            raise ValueError("Failed to load alignment model. Please check your internet connection and model settings.")

        # Align transcription
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
        if result is None or "segments" not in result:
            raise ValueError("Alignment failed. Please check the audio file and model settings.")

        # Perform diarization
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=HF_KEY, device=device)
        if diarize_model is None:
            raise ValueError("Failed to load diarization model. Please check your internet connection and model settings.")
        diarize_segments = diarize_model(audio)
        if diarize_segments is None:
            raise ValueError("Diarization failed. Please check the audio file and model settings.")

        # Assign speakers to segments
        result = whisperx.assign_word_speakers(diarize_segments, result)
        if result is None or "segments" not in result:
            raise ValueError("Speaker assignment failed. Please check the audio file and model settings.")

        # Prepare transcript output
        output_dict = {}
        output_str = ""
        combined_text = ""
        i = 1
        for segment in result["segments"]:
            if "speaker" not in segment:
                continue
            text = segment["text"]
            speaker = segment["speaker"]
            inner_dict = {"Speaker": speaker, "Text": text}
            output_dict[i] = inner_dict
            i += 1
            output_str += f"Speaker: {speaker},\nText: {text}\n"
            combined_text += text + " "  # Combine text for summarization

        return output_dict, output_str.strip(), combined_text.strip(), detected_language

    except Exception as e:
        print(f"Error in convert_speech_text: {e}")
        return None, None, None, None


def summarize_transcription(transcription, language_option):
    try:
        # Preprocess input text for T5
        input_text = f"summarize: {transcription}"
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

        # Generate summary
        summary_ids = t5_model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                        early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return summary
    except Exception as e:
        print(f"Error in summarize_transcription: {e}")
        return None


def process_audio_file(audio_file, output_type="Transcript", language_option="English"):
    try:
        transcript_dict, transcription_str, combined_text, detected_language = convert_speech_text(audio_file, language_option)

        if output_type == "Summary":
            summary = summarize_transcription(combined_text, language_option)
            return summary
        else:
            return transcription_str
    except Exception as e:
        print(f"Error in process_audio_file: {e}")
        return None




st.set_page_config(page_title="Audio and Video Transcription and Summary", page_icon="🎤", layout="centered")

# Add custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #333;
        text-align: center;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        background-color: #fff;
        text-align: center;
        margin-bottom: 1rem;
        position: relative;
    }
    .upload-box:hover {
        border-color: #aaa;
    }
    .button {
        background-color: #007bff;
        color: #fff;
        font-size: 1rem;
        padding: 0.6rem 1.2rem;
        border-radius: 5px;
        text-align: center;
        display: block;
        margin: 0 auto;
    }
    .button:hover {
        background-color: #0056b3;
    }
    .color-bar {
        display: flex;
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 10px;
        border-top-left-radius: 10px;
        border-top-right-radius: 10px;
        overflow: hidden;
    }
    .color-bar div {
        flex: 1;
        height: 100%;
    }
    .color-bar .color1 { background-color: #FF5733; }
    .color-bar .color2 { background-color: #FF8D1A; }
    .color-bar .color3 { background-color: #FFC300; }
    .color-bar .color4 { background-color: #28B463; }
    .color-bar .color5 { background-color: #3498DB; }
    </style>
""", unsafe_allow_html=True)


# Main layout
st.markdown("<div class='main'>", unsafe_allow_html=True)

# Title and description
st.markdown("<div class='title'>🎤 Audio and Video Transcription and Summary</div>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'><strong style='color:green;'>*Note:</strong> <em><strong>Upload an audio file to get a transcript or summary. Supported formats: MP3, WAV, M4A.</strong></em></p>",
            unsafe_allow_html=True)

# Use columns for layout
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"], help="Supported formats: MP3, WAV, M4A")
with col2:
    output_type = st.selectbox("Select Output Type", ["Transcript", "Summary"], help="Choose whether you want a transcript or summary.")

# Add a language selection option
language_option = st.selectbox("Select Output Language",
                               ["English", "Spanish", "French", "German", "Italian", "Portuguese"],
                               help="Select the language for the output.")

# Add a button to process the file
if st.button("Process", key="process_button", help="Click to process the uploaded file"):
    if uploaded_file is not None:
        with st.spinner("Processing..."):
            temp_file_path = "temp_audio_file"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            result = process_audio_file(temp_file_path, output_type, language_option)
            if result:
                st.text_area("Result", result, height=400)
            else:
                st.error("An error occurred while processing the file.")
            os.remove(temp_file_path)  # Clean up the temporary file
    else:
        st.warning("Please upload an audio file.")

st.markdown("</div>", unsafe_allow_html=True)

