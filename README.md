# Instagram Caption Generator

This project provides a mood-based Instagram caption assistant using both a Machine Learning (ML) recommender and a Large Language Model (LLM) via Hugging Face's API. The app is built using **Streamlit**.

---

## Features

### LLM-Based Caption Generator

Generates creative Instagram captions using the Hugging Face `Mixtral-8x7B-Instruct` model based on image descriptions and selected moods.

### ML-Based Caption Recommender

Recommends captions using **TF-IDF** vectorization and **cosine similarity** from a curated mood-tagged caption dataset.

---

## Files in This Project

```

├── .env                  # Stores Hugging Face API token (not tracked by Git)
├── .gitignore            # Specifies files/folders to be ignored by Git
├── .github/workflows/    # Contains GitHub Actions CI workflow
│   └── app.yml          # Workflow for testing and CI
├── recommender.py        # ML logic for caption recommendation
├── requirements.txt      # Python dependencies
├── Test_Cases.py         # Unit tests for caption generation and recommendation

```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Agile_Project_SEE.git
cd Agile_Project_SEE
```

### 2. Set Up Environment Variables

Create a `.env` file in the root directory and add your Hugging Face API token:

```env
HF_TOKEN=your_huggingface_token_here
```

### 3. Install Dependencies

To install the necessary dependencies for the project, run the following command:

```bash
pip install -r requirements.txt
```

This will install the required Python libraries, including `streamlit`, `scikit-learn`, and `transformers`.

### 4. Run the Streamlit App

If you have an app file (e.g., `recommender.py`), run it with the following command:

```bash
streamlit run recommender.py
```

> **Note:** If the Streamlit UI is not yet created, this repo primarily runs as a backend/test suite at this stage.

### 5. Run Tests

To ensure that everything is working as expected, you can run unit tests using `pytest`:

```bash
pytest Test_Cases.py
```

---

## Acknowledgments

- Hugging Face for the LLM API
- scikit-learn for ML utilities
- Streamlit for rapid UI development
