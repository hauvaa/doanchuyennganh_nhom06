import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

nltk.download('punkt')
nltk.download('wordnet')

# Đọc dữ liệu
sms_spam = pd.read_csv('../SMSSpamCollection', sep='\t', header=None, names=['Label', 'SMS'])

# Tiền xử lý dữ liệu
sms_spam['SMS'] = sms_spam['SMS'].str.replace('[^a-zA-Z\s]', '', regex=True)  # Loại bỏ ký tự đặc biệt và số
sms_spam['SMS'] = sms_spam['SMS'].str.lower()  # Chuyển sang chữ thường
sms_spam['SMS'] = sms_spam['SMS'].apply(lambda x: ' '.join([nltk.WordNetLemmatizer().lemmatize(word) for word in x.split()]))  # Lemmatization

# Tách dữ liệu thành training và testing set
X = sms_spam['SMS']
y = sms_spam['Label']

# Sử dụng TfidfVectorizer thay vì CountVectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X_features = vectorizer.fit_transform(X)

# Tách tập huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Huấn luyện mô hình Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Dự đoán trên tập test
y_pred = nb_model.predict(X_test)

# Đánh giá các chỉ số mô hình
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

# In ra các chỉ số
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Lưu mô hình và vectorizer đã huấn luyện
joblib.dump(nb_model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
