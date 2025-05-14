import streamlit as st
import requests
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

# ------------------ Hugging Face Caption Generator ------------------ #
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
load_dotenv()
token = os.getenv("HF_TOKEN")

if not token:
    st.error("❌ Missing Hugging Face API token. Please check your .env file and set HF_TOKEN.")
    st.stop()

headers = {"Authorization": f"Bearer {token}"}

def generate_caption(prompt, max_tokens=150):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.75,
            "top_p": 0.9,
            "do_sample": True
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"].replace(prompt, "").strip()
        elif isinstance(result, dict) and "error" in result:
            return f"⚠️ Hugging Face API error: {result['error']}"
        else:
            return "⚠️ Unexpected response from API."
    except requests.exceptions.RequestException as e:
        return f"❌ Request failed: {e}"

# ------------------ ML Caption Recommendation ------------------ #
caption_data = [
    ("Sunsets and palm trees 🌴🌅", "chill"),
    ("Coffee first, adulting second ☕️", "relatable"),
    ("Chasing dreams, not people ✨", "motivational"),
    ("Beach days are the best days 🏖️", "happy"),
    ("Just vibing and thriving 😎", "chill"),
    ("Adventures fill your soul 🌍", "adventurous"),
    ("Life’s too short for bad vibes 💫", "motivational"),
    ("Happiness looks good on me 😄", "happy"),
    ("Weekend mode: ON 🛌📺", "lazy"),
    ("Smiles, sunshine, and selfies ☀️📸", "happy"),
    ("Wander often, wonder always 🚶‍♂️🌠", "adventurous"),
    ("Stay wild, moon child 🌙", "dreamy"),
    ("City lights and late nights 🌃", "urban"),
    ("Peace, love, and sandy feet 🌊", "chill"),
    ("No filter needed when you’re glowing 🌟", "confident"),
    ("Living my best life 💃✨", "confident"),
    ("Good vibes only 🌈", "happy"),
    ("Lost in the moment 🌀", "dreamy"),
    ("Sippin’ on sunshine 🍹☀️", "chill"),
    ("Catch flights, not feelings ✈️💔", "sassy"),
    ("Making memories all day long 📸", "happy"),
    ("Born to stand out 🌟", "confident"),
    ("Take only pictures, leave only footprints 👣", "adventurous"),
    ("Keepin' it real since day one 🔥", "bold"),
    ("Too glam to give a damn 💅", "sassy"),
    ("On cloud nine ☁️", "happy"),
    ("Netflix, snacks, repeat 🍿", "lazy"),
    ("Elegance never goes out of style 👠", "confident"),
    ("Heart full of wanderlust ✨🌍", "adventurous"),
    ("Doing nothing, and proud of it 🛋️", "lazy"),
    ("Don’t quit your daydream 💭", "dreamy"),
    ("Slaying in silence 💋", "bold"),
    ("Can’t hear the haters over my playlist 🎧", "bold"),
    ("Less perfection, more authenticity 🌿", "relatable"),
    ("Fuelled by caffeine and ambition ☕🚀", "motivational"),
    ("Throwing sass like confetti 🎉", "sassy"),
    ("Not lost, just exploring 🧭", "adventurous"),
    ("Creating my own sunshine ☀️", "motivational"),
    ("Pillow talks and pastel skies 🌸", "dreamy"),
    ("Keep calm and scroll on 📱", "relatable"),
    ("Catch me under the stars ✨", "dreamy"),
    ("Soft heart, strong mind 💗🧠", "confident"),
    ("Out of office, into nature 🌲", "chill"),
    ("Flipping pages and sipping tea 📖🍵", "chill"),
    ("Sparkle every step of the way ✨👟", "motivational"),
    ("Brunch is always a good idea 🥞", "relatable"),
    ("Sweet as honey, stings when needed 🐝", "sassy"),
    ("Midnight thoughts, morning coffee ☕🌙", "dreamy"),
    ("Just a girl building her empire 👑", "motivational"),
    ("Unbothered and moisturized 💅", "bold"),
]

def get_captions_by_mood(mood):
    return [caption for caption, m in caption_data if m == mood]

def recommend_captions_by_mood(input_caption, mood, top_n=3):
    if not input_caption.strip():
        raise ValueError("Input caption cannot be empty.")
    
    captions = get_captions_by_mood(mood)
    if not captions:
        raise ValueError(f"No captions found for mood: '{mood}'")
    
    captions_with_input = captions + [input_caption]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(captions_with_input)

    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]

    return [captions[i] for i in similar_indices]

# ------------------ Streamlit UI ------------------ #
st.set_page_config(page_title="📸 Instagram Caption Generator", layout="wide")
st.title("📸 Mood-Based Instagram Caption Assistant")

tab1, tab2 = st.tabs(["🧠 LLM-Based Captions", "📊 ML-Based Recommendations"])

# --- LLM-Based Caption Generator --- #
with tab1:
    st.subheader("✨ Hugging Face Caption Generator")
    image = st.file_uploader("Upload a photo (optional)", type=["jpg", "jpeg", "png"])
    description = st.text_area("Describe the photo in words:")

    hf_mood_options = ["Select a mood...", "Humorous", "Romantic", "Introspective", 
                       "Adventurous", "Chill", "Empowering", "Motivational", "Dreamy", 
                       "Sassy", "Bold", "Confident", "Relatable", "Lazy", "Urban"]
    hf_mood = st.selectbox("Choose the mood for your caption", hf_mood_options)

    if st.button("Generate Captions with LLM"):
        if hf_mood == "Select a mood...":
            st.warning("Please select a valid mood.")
            st.stop()

        with st.spinner("Crafting your perfect caption..."):
            if not image and not description:
                st.warning("Please upload a photo and/or provide a description.")
                st.stop()

            if image:
                image_data = Image.open(image)
                st.image(image_data, caption="Uploaded Photo", use_container_width=True)

            if image and description:
                photo_desc = (
                    f"The user uploaded a photo and described it as follows:\n"
                    f"'{description}'."
                )
            elif image:
                photo_desc = "The user uploaded a photo. You can infer a possible description from it."
            else:
                photo_desc = f"The user provided the following description of the photo:\n'{description}'."

            prompt = (
                f"You are an AI Instagram caption generator. Below is the image context:\n\n"
                f"{photo_desc}\n\n"
                f"The desired mood for the caption is **{hf_mood.lower()}**.\n"
                f"Generate 3 creative and mood-matching Instagram captions."
            )

            hf_output = generate_caption(prompt)
            st.markdown("### 📌 Generated Captions:")
            st.markdown("**LLM Output:**")
            st.write(hf_output)

# --- ML-Based Caption Recommender --- #
with tab2:
    st.subheader("🔍 ML-Based Caption Recommender")
    moods = ["Select a mood..."] + sorted(set(mood for _, mood in caption_data))
    selected_mood = st.selectbox("Select a mood:", moods)
    input_caption = st.text_input("Enter your caption to match:")
    top_n = st.slider("How many similar captions to suggest?", 1, 5, 3)

    if st.button("Get Similar Captions with ML"):
        if selected_mood == "Select a mood...":
            st.warning("Please select a valid mood.")
            st.stop()
        if input_caption:
            try:
                recommendations = recommend_captions_by_mood(input_caption, selected_mood, top_n)
                st.markdown("### 📖 Recommended Captions:")
                for i, cap in enumerate(recommendations, 1):
                    st.write(f"{i}. {cap}")
            except ValueError as ve:
                st.error(str(ve))
        else:
            st.warning("Please enter a caption to get recommendations.")
