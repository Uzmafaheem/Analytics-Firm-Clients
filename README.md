# 🧠 Data Science Project Portfolio: NLP · Time Series · Neural Networks
## 📘 Overview
- This project demonstrates three core applications of modern data science and machine learning techniques across different data types:
        - 🗂 Natural Language Processing (NLP): Text classification of forum posts.
        - 📈 Time Series Forecasting: Financial trend analysis and forecasting using S&P 500 data.
        - 🤖 Neural Network Modeling: Classification (image-based) and regression (numeric data) tasks using TensorFlow and PyTorch.
- Each section showcases a realistic business use case, key preprocessing steps, model development, evaluation, and insights.
## 🧩 Part 1: Natural Language Processing (NLP)
### 🎯 Objective
 Automatically categorize community forum posts (similar to the 20 Newsgroups dataset) to help support teams route queries efficiently.
### 🧰 Techniques Used
- Tokenization and text normalization
- Stopword removal
- Lemmatization and stemming
- N-gram extraction
- Word frequency analysis and visualization

### 📊 Key Insights
- Frequent word analysis revealed domain-specific vocabulary (e.g., “space,” “NASA,” “engine,” “graphics”).
- N-gram analysis helped capture contextual patterns (e.g., “space shuttle,” “graphics card”).
- Comprehensive preprocessing significantly improved the clarity of textual patterns and model input quality.

### 🧠 Outcome
Prepared a clean, tokenized corpus suitable for downstream machine learning tasks such as topic classification or sentiment analysis.

## 📈 Part 2: Time Series Forecasting (Financial Data)
### 🎯 Objective
- Analyze and forecast the S&P 500 closing prices to understand market trends and support investment strategy planning.
### 🧰 Techniques Used
- Data exploration and visualization
- Stationarity testing using Augmented Dickey-Fuller (ADF)
- Differencing transformation
- Time series decomposition (trend, seasonal, residual)
- ARIMA modeling and diagnostics
- Forecast visualization

### 🧪 Key Findings

| Step                       | Result                                                         |
| -------------------------- | -------------------------------------------------------------- |
| **ADF Test (Original)**    | Non-stationary (p = 0.74)                                      |
| **ADF Test (Differenced)** | Stationary (p ≈ 0.0)                                           |
| **Best ARIMA Model**       | ARIMA(1,1,0) and ARIMA(1,1,1) compared                         |
| **Model Diagnostics**      | Residuals showed no autocorrelation, confirming good model fit |


### 💡 Insights
- The original price series showed a strong upward trend and non-stationarity.
- After first differencing, the series became stationary and suitable for ARIMA modeling.
- ARIMA(1,1,1) captured short-term dependencies effectively.
- Residual analysis confirmed model adequacy (Ljung–Box test p > 0.05).

### 🧠 Outcome
Built a reliable ARIMA forecasting model capable of predicting near-term market behavior with strong statistical validity.

## 🤖 Part 3: Neural Network Modeling
### 🧩 Tasks
- Classification: Recognizing hand-written digits (MNIST-like dataset).
- Regression (optional extension): Predicting numeric outcomes from structured data.
### ⚙️ Frameworks
- TensorFlow / Keras for high-level deep learning model development.
- PyTorch for low-level experimentation and model training control.

### 🧠 Part 3A — TensorFlow/Keras Implementation
 - Architecture:
            - Input: 64 features (flattened 8×8 digit images)
            - Hidden Layers: 128 → 64 (ReLU activation)
            - Output: 10 classes (softmax activation)
- Results:
       - Test Accuracy: 98.6% (ReLU)
       - Validation curves show excellent generalization.
 Activation Function Experimentation:

| Activation  | Test Accuracy | Observations                                                    |
| ----------- | ------------- | --------------------------------------------------------------- |
| **ReLU**    | **0.9861**    | Best performance; fast convergence, minimal vanishing gradients |
| **Sigmoid** | 0.9667        | Slower training; vanishing gradient effects                     |
| **Tanh**    | 0.9833        | Good alternative; smoother convergence but slightly less stable |



### 📊 Conclusion:
ReLU consistently outperformed other activations, confirming it as the optimal choice for hidden layers in image classification tasks.

### ⚡ Part 3B — PyTorch Implementation
- Architecture:
          - Identical to the Keras model for performance comparison.
          - Implemented custom training loop with Adam optimizer and CrossEntropyLoss.
- Results:
       - Test Accuracy: ~98%
        - Smooth loss curve; confirmed model stability and reproducibility across frameworks.

#### 📊 Conclusion:
Both TensorFlow and PyTorch delivered high-performing models. PyTorch offers more flexibility for experimentation, while TensorFlow excels in ease of deployment.

### 🏁 Overall Conclusions

| Domain              | Key Techniques                       | Core Insight                                                 |
| ------------------- | ------------------------------------ | ------------------------------------------------------------ |
| **NLP**             | Tokenization, Lemmatization, N-grams | Clean text improves feature extraction and topic separation. |
| **Time Series**     | Differencing, ARIMA, Decomposition   | Stationarity is essential for reliable forecasting.          |
| **Neural Networks** | MLPs, Activation Functions           | ReLU leads to faster convergence and higher accuracy.        |

### 📦 Technologies Used
- Languages: Python
- Libraries:
      - Data & Visualization: pandas, numpy, matplotlib, seaborn
      - NLP: nltk, sklearn
      - Time Series: statsmodels, yfinance
      - Deep Learning: TensorFlow/Keras, PyTorch

### 📚 Learning Outcomes
- ✅ Mastered preprocessing pipelines for diverse data types
- ✅ Understood statistical vs deep learning modeling paradigms
- ✅ Implemented, tuned, and evaluated machine learning models across domains
- ✅ Gained experience interpreting model outputs and drawing actionable insights

### 🚀 Future Enhancements
- Integrate transformer-based NLP models (e.g., BERT).
- Explore LSTM or Prophet for time series forecasting.
- Experiment with CNNs for image data and autoencoders for dimensionality reduction.


#### 🧾 Author
Faheemunnisa Syeda
     Data Scientist | Machine Learning Enthusiast
         - 📧 [https://www.linkedin.com/in/faheem-unnisa-s-6270888b/]
         - 📘 [https://github.com/syedafaheem7]
