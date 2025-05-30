# 🎵 Dynamic Music Review Generator

A hybrid AI system that analyzes song attributes using fuzzy logic and generates engaging music reviews with both rule-based and generative AI approaches.

## 🌟 Features

- **Dual-Analysis Engine**:
  - Fuzzy logic evaluation of 7 musical dimensions
  - FLAN-T5 LLM for natural language generation
- **Comprehensive Attribute Scoring**:
  - Energy (0-100) • Tempo (0-100) • Vocals (0-10)
  - Instrumentation (0-10) • Melodic Complexity (0-10)
  - Lyrical Sentiment (-1 to 1) • Overall Mood (0-100)
- **Smart Feedback System**:
  - Collects and stores user feedback

## 🛠️ Technologies Used

| Component               | Technology               |
|-------------------------|--------------------------|
| Backend Framework       | Flask (Python)           |
| Fuzzy Logic             | scikit-fuzzy             |
| Language Model          | FLAN-T5 (Hugging Face)   |
| Frontend                | HTML/CSS/JS              |
| Data Persistence        | JSON                     |
