import pytest
from recommender_LLM import recommend_captions_by_mood, generate_caption, get_captions_by_mood

# ---------------- Positive Test Cases ---------------- #

def test_recommend_valid_caption():
    input_caption = "Feeling happy and vibrant"
    mood = "happy"
    results = recommend_captions_by_mood(input_caption, mood, top_n=3)
    assert len(results) == 3
    assert all(isinstance(c, str) for c in results)

def test_generate_caption_with_description():
    description = "A calm sunset by the beach with waves rolling in"
    mood = "Chill"
    prompt = (
        f"You are an AI Instagram caption generator. The user provided the following photo description:\n"
        f"{description}\n"
        f"The mood of the caption should be **{mood.lower()}**.\n"
        f"Generate 3 creative and mood-matching Instagram captions."
    )
    output = generate_caption(prompt)
    assert isinstance(output, str)
    assert len(output) > 0

def test_recommend_top_one():
    input_caption = "Lazy mornings are the best"
    mood = "lazy"
    result = recommend_captions_by_mood(input_caption, mood, top_n=1)
    assert len(result) == 1

# ---------------- Negative Test Cases ---------------- #

def test_recommend_empty_caption():
    with pytest.raises(ValueError):
        recommend_captions_by_mood("", "happy", top_n=3)

def test_recommend_invalid_mood():
    input_caption = "Sunset vibes"
    mood = "nonexistent"
    with pytest.raises(ValueError, match="No captions found for mood"):
        recommend_captions_by_mood(input_caption, mood, top_n=3)


def test_recommend_too_many_results():
    input_caption = "Adventures in the mountains"
    mood = "adventurous"
    results = recommend_captions_by_mood(input_caption, mood, top_n=100)
    assert len(results) <= len(get_captions_by_mood("adventurous"))
