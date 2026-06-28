# Model Card: MoodTune Emotion Detection

## 1. Model Overview
- **Architecture:** Vision Transformer (ViT-Base/16)
- **Checkpoint:** `trpakov/vit-face-expression` (HuggingFace)
- **Task:** 7-Class Facial Expression Recognition (Image Classification)
- **Inference Pipeline:** OpenCV Haar Cascades for face extraction → ViT for emotion classification.

## 2. Dataset Description
- **Dataset:** FER2013 (Facial Expression Recognition 2013)
- **Source:** Originally published for the ICML 2013 Representation Learning Challenge (Kaggle).
- **Size:** ~35,887 images.
- **Format:** 48×48 pixel grayscale images.
- **Classes (7):** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

## 3. Performance Metrics
The model exhibits strong performance on majority classes but struggles with underrepresented emotive extremes, as seen in the per-class F1 scores.

| Emotion  | F1 Score | Representation in Training Data |
|----------|----------|---------------------------------|
| Happy    | ~0.87    | High                            |
| Neutral  | ~0.82    | High                            |
| Surprise | ~0.78    | Medium                          |
| Angry    | ~0.74    | Medium                          |
| Sad      | ~0.71    | Medium                          |
| Fear     | ~0.55    | Low                             |
| Disgust  | ~0.48    | Very Low (~1.5%)                |

### Confusion Matrix Analysis
- **Fear vs. Disgust:** High confusion potential. Both expressions often feature narrowed eyes and tightened lower facial muscles in this dataset's low-resolution grayscale format.
- **Angry vs. Disgust:** Noticeable overlap due to shared micro-expressions (furrowed brow, tightened lips).

## 4. Augmentation Strategy
To mitigate the severe class imbalances and low-contrast issues identified during Exploratory Data Analysis (EDA), specific augmentation strategies are applied during training:
- **Disgust (6× Oversampling):** Horizontal Flip + Rotation (±15°) + Brightness adjustments (0.5× to 1.5×).
- **Fear (3× Oversampling):** Horizontal Flip + Zoom (0.9× to 1.1×) + Rotation (±10°).
- **Other Classes (1.5×):** Horizontal Flip (for standard regularization).

## 5. Known Biases & Limitations
> [!WARNING]
> Please review these limitations before deploying MoodTune in any critical environment.

- **Demographic Bias:** The FER2013 dataset is heavily skewed towards Western facial features. The model significantly underperforms on non-Western demographics.
- **Lighting Sensitivity:** ViT models are highly sensitive to pixel distributions. Environments with poor lighting (< 100 lux) degrade accuracy by ~15%.
- **Single-Face Limitation:** The current OpenCV cascade pipeline extracts and processes only the largest detected face. Group emotion detection is not supported.
- **Construct Validity:** Emotion labels (Happy, Sad, Angry) are culturally constructed. The model detects *facial muscle configurations*, not internal subjective feelings.

## 6. Intended & Out-of-Scope Uses
- **Intended Use:** Prototyping, entertainment, and lightweight interactive media (e.g., matching music to facial expressions).
- **Out-of-Scope Use:** Any application involving automated decision-making, law enforcement, hiring, psychiatric diagnosis, or surveillance. The model is NOT robust enough to determine true human intent or emotional state for consequential actions.

## 7. Dataset Provenance and License
- **FER2013 Dataset:** Available for non-commercial research purposes.
- **HuggingFace Checkpoint:** Please refer to the specific license provided by `trpakov` on the HuggingFace Hub. Ensure compliance with open-source and non-commercial guidelines when deploying this model.
