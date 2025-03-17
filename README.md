# 📈 Stock Prediction App

## 🌍 Live Demo
**Access the app here:** https://stockpredictionbysakshi.streamlit.app/

---

## 📌 About This Project
This **Stock Prediction App** allows users to **predict stock prices for the next 30 days** based on historical data. Users can enter a **stock ticker (e.g., AAPL, TSLA, GOOGL)**, and the app will display a **graph of predicted prices** using **Linear Regression**.

🚀 **Key Features:**
- 📊 **Predicts stock prices for the next 30 days**
- 🔎 **User inputs a stock ticker symbol**
- 📉 **Fetches historical stock data from Yahoo Finance**
- 📈 **Trains a machine learning model (Linear Regression) to predict prices**
- 🎯 **Plots predictions on an interactive graph**
- 🌐 **Deployed online using Streamlit Cloud**

---

## ⚙️ Tech Stack
- **Python** 🐍
- **Streamlit** 🎈 (for the web interface)
- **Yahoo Finance API (`yfinance`)** 📈 (for fetching stock data)
- **Scikit-Learn (`sklearn`)** 🤖 (for training the prediction model)
- **Matplotlib** 📊 (for visualization)
- **GitHub & Streamlit Cloud** ☁️ (for deployment)

---

## 🚀 How to Use the App
### **🔹 Online Version (Easiest Way)**
1. Open the app: **https://stockpredictionbysakshi.streamlit.app/**
2. Enter a stock ticker (e.g., `AAPL`, `TSLA`, `GOOGL`).
3. Click **Predict**.
4. View the **predicted stock prices & graph for the next 30 days**.

---

## 🛠️ How to Run Locally
### **🔹 Step 1: Clone the Repository**
```sh
git clone https://github.com/yourusername/webapp_stockprediction.git
cd webapp_stockprediction
```

### **🔹 Step 2: Create a Virtual Environment**
```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### **🔹 Step 3: Install Dependencies**
```sh
pip install -r requirements.txt
```

### **🔹 Step 4: Run the App**
```sh
streamlit run app.py
```

🔗 The app will open in your browser at: `http://localhost:8501`

---

## 📜 Project Structure
```plaintext
webapp_stockprediction/
│-- app.py                 # Main Streamlit app
│-- requirements.txt       # Required dependencies
│-- README.md              # Project documentation (this file)
│-- .gitignore             # Git ignored files
```

---

## 🔮 Future Improvements
🎯 Enhance the model with **LSTM (Neural Networks) for better predictions**
🎯 Add **more technical indicators (e.g., Moving Averages, RSI)**
🎯 Improve the **UI design & user experience**
🎯 Deploy on **AWS/GCP for high scalability**

---

## 🤝 Contributing
Contributions are welcome! If you have ideas, **feel free to fork the repo, make changes, and submit a pull request**. 🚀

---

## 🔗 Connect
💬 **Author:** Sakshi Kiran Naik
📧 **Email:** sakshi.kiran.naik@gmail.com
🔗 **LinkedIn:** https://www.linkedin.com/in/sakshi-kiran-naik-2313531a5/ 

---

🚀 **Happy Coding & Investing!** 📈🔥

