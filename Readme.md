# 🎵 MoodTune AI

MoodTune AI is a complete end-to-end Python project that performs real-time emotion detection and acts as a dynamic Spotify music recommender. It uses a fine-tuned Vision Transformer (ViT) to classify your facial expressions through your webcam and instantly curates a Spotify playlist that matches your current mood.

## 🚀 Features

- **Real-Time Emotion Detection**: Low-latency (<180ms) inference using HuggingFace's `trpakov/vit-face-expression`.
- **Intelligent Music Curation**: Maps 7 distinct emotion classes to specific Spotify audio features (valence, energy, danceability, tempo, acousticness).
- **Beautiful UI**: A highly responsive Streamlit dashboard featuring live webcam feeds, Plotly radar charts, interactive mood timelines, and Spotify track cards.
- **Data Science Ready**: Includes a comprehensive, interview-ready Jupyter notebook (`data/fer2013_eda.ipynb`) that explores the FER2013 dataset, class imbalances, and augmentation strategies.
- **Graceful Fallbacks**: Fully functional without Spotify credentials by using a built-in mock track generator.

## 📁 Project Structure

```
moodtune/
├── app.py                  # Main Streamlit app
├── emotion/
│   └── detector.py         # ViT model + OpenCV webcam pipeline
├── recommender/
│   ├── spotify.py          # Spotipy engine + caching
│   └── emotion_map.py      # Emotion → audio feature mapping
├── data/
│   └── fer2013_eda.ipynb   # FER2013 EDA + augmentation logic
├── components/
│   ├── radar_chart.py      # Plotly emotion radar chart
│   ├── album_cards.py      # 6-track album art card UI
│   └── mood_timeline.py    # Mood history timeline chart
├── requirements.txt
└── README.md
```

## 🛠️ Installation & Setup

1. **Clone the repository and enter the directory**:
   ```bash
   git clone <repo-url>
   cd moodtune
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   uv venv
   # On Windows: .venv\Scripts\activate
   # On Mac/Linux: source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. **(Optional) Add Spotify Credentials**:
   To fetch real recommendations, create a `.env` file in the root directory:
   ```ini
   SPOTIPY_CLIENT_ID=your_spotify_client_id
   SPOTIPY_CLIENT_SECRET=your_spotify_client_secret
   ```
   *If you skip this step, the app will run in "Mock Mode" and provide sample tracks so you can still test the UI.*

4. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## 🧠 Data Science Notebook

To explore the data science methodologies behind this project, open the Jupyter Notebook:
```bash
jupyter notebook data/fer2013_eda.ipynb
```
*(Requires the `fer2013.csv` dataset to be downloaded from Kaggle and placed in the project root).*

---
*Built with ❤️ using Streamlit, HuggingFace, OpenCV, Plotly, and Spotipy.*
