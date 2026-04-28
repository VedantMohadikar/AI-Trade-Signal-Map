# 📊 AI Trade Signal Map

An interactive machine learning visualization tool that maps **trading signals (Buy / Sell / Hold)** based on market features like **momentum (returns)** and **volatility**.

---

## 🚀 Project Overview

This project demonstrates how machine learning can be used to classify trading signals and visualize them in a 2D feature space.

* 📈 Uses synthetic price data (can be replaced with real data)
* 🤖 Trains ML models like Random Forest / Logistic Regression
* 🎯 Predicts Buy / Sell / Hold signals
* 🎨 Visualizes decision boundaries in a **signal map**

---

## 🧠 Features

* Generate synthetic financial time-series data
* Feature engineering:

  * Returns (Momentum)
  * Rolling Volatility
* Multi-class classification:

  * **Buy (2)**
  * **Hold (1)**
  * **Sell (0)**
* Visualization:

  * Scatter plot of signals
  * Decision boundary using trained model

---

## 🛠️ Tech Stack

* Python 3.13
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Project Structure

```
AI-Trade-Signal-Map/
│── main.py
│── requirements.txt
│── pyproject.toml
│── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/VedantMohadikar/AI-Trade-Signal-Map.git
cd AI-Trade-Signal-Map
```

---

### 2️⃣ Create virtual environment (recommended)

Using `uv`:

```
uv venv
uv sync
```

OR using pip:

```
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main script:

```
python main.py
```

---

## 📊 Output

The program generates:

* A 2D visualization of:

  * X-axis → Momentum (Returns)
  * Y-axis → Volatility
* Colored regions showing model decision boundaries:

  * 🔵 Sell
  * 🟣 Hold
  * 🔷 Buy

---

## 🔧 Model Options

You can switch models in `main.py`:

```
MODEL_TYPE = 'RandomForest'  # or 'LogisticRegression'
```

---

## 📌 Future Improvements

* Use real stock market data (e.g., Yahoo Finance API)
* Add more technical indicators (RSI, MACD, Moving Averages)
* Backtesting strategy performance
* Deploy as a web app (Streamlit / Flask)
* Integrate live trading signals

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Vedant Mohadikar**
GitHub: https://github.com/VedantMohadikar

---
