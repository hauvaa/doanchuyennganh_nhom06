import pandas as pd
import tkinter as tk
from tkinter import messagebox
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Tải các mô hình đã huấn luyện và vectorizer
nb_model = joblib.load('naive/naive_bayes_model.pkl')  # Tải mô hình Naive Bayes
svm_model = joblib.load('svm/svm_model.pkl')  # Tải mô hình SVM
vectorizer1 = joblib.load('naive/vectorizer.pkl')  # Tải vectorizer Naive Bayes
vectorizer2 = joblib.load('svm/vectorizer.pkl')  # Tải vectorizer SVM

# Chức năng dự đoán spam/ham cho Naive Bayes
def predict_spam_or_ham_nb(message):
    message_vec = vectorizer1.transform([message])
    prediction = nb_model.predict(message_vec)
    return prediction[0]

# Chức năng dự đoán spam/ham cho SVM
def predict_spam_or_ham_svm(message):
    message_vec = vectorizer2.transform([message])
    prediction = svm_model.predict(message_vec)
    return prediction[0]

# Dữ liệu so sánh thuật toán (cập nhật từ quá trình huấn luyện)
metrics = {
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Naive Bayes": [0.9695, 1.0000, 0.7718, 0.8712],
    "SVM": [0.9874, 1.0000, 0.9060, 0.9507]
}

# Hàm vẽ biểu đồ so sánh
def show_comparison_chart():
    metric_names = metrics["Metric"]
    nb_values = metrics["Naive Bayes"]
    svm_values = metrics["SVM"]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, nb_values, width, label='Naive Bayes', color='skyblue')
    ax.bar(x + width / 2, svm_values, width, label='SVM', color='orange')

    ax.set_xlabel('Metrics')
    ax.set_title('Comparison of Naive Bayes and SVM')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()

    plt.tight_layout()
    plt.show()

# Giao diện đăng nhập
def login_window():
    login_screen = tk.Tk()
    login_screen.title("Đăng nhập")
    login_screen.geometry("400x300")
    login_screen.config(bg="#f0f8ff")

    accounts = {"admin": "12345"}

    def check_login():
        username = username_entry.get()
        password = password_entry.get()

        if username in accounts and accounts[username] == password:
            messagebox.showinfo("Thành công", "Đăng nhập thành công!")
            login_screen.destroy()
            open_main_app()  # Mở giao diện ứng dụng sau khi đăng nhập thành công
        else:
            messagebox.showerror("Thất bại", "Tên đăng nhập hoặc mật khẩu không đúng.")

    tk.Label(login_screen, text="Chào mừng đến với ứng dụng Spam/Ham", font=("Arial", 14, "bold"), bg="#f0f8ff").pack(pady=20)

    tk.Label(login_screen, text="Tên đăng nhập:", font=("Arial", 12), bg="#f0f8ff").pack(pady=5)
    username_entry = tk.Entry(login_screen, font=("Arial", 12), width=25)
    username_entry.pack(pady=5)

    tk.Label(login_screen, text="Mật khẩu:", font=("Arial", 12), bg="#f0f8ff").pack(pady=5)
    password_entry = tk.Entry(login_screen, font=("Arial", 12), show="*", width=25)
    password_entry.pack(pady=5)

    tk.Button(login_screen, text="Đăng nhập", font=("Arial", 12), command=check_login, bg="#4CAF50", fg="white", relief="solid", width=20).pack(pady=20)

    login_screen.mainloop()

# Giao diện chính của ứng dụng sau khi đăng nhập thành công
def open_main_app():
    main_app = tk.Tk()
    main_app.title("Ứng dụng Spam/Ham")
    main_app.geometry("600x550")
    main_app.config(bg="#f0f8ff")

    tk.Label(main_app, text="Ứng dụng Spam/Ham", font=("Arial", 18, "bold"), bg="#f0f8ff").pack(pady=20)

    tk.Label(main_app, text="Nhập tin nhắn:", font=("Arial", 14), bg="#f0f8ff").pack(pady=5)
    message_entry = tk.Entry(main_app, font=("Arial", 12), width=40)
    message_entry.pack(pady=10)

    result_label = tk.Label(main_app, text="Kết quả dự đoán: ", font=("Arial", 14), bg="#f0f8ff")
    result_label.pack(pady=10)

    def check_spam_or_ham():
        message = message_entry.get()
        if algorithm_var.get() == 'Naive Bayes':
            result = predict_spam_or_ham_nb(message)
        elif algorithm_var.get() == 'SVM':
            result = predict_spam_or_ham_svm(message)
        result_label.config(text=f"Kết quả dự đoán: {result}")

    tk.Button(main_app, text="Kiểm tra", font=("Arial", 14), command=check_spam_or_ham, bg="#4CAF50", fg="white", relief="solid", width=20).pack(pady=10)

    # Chọn thuật toán
    algorithm_var = tk.StringVar(value='Naive Bayes')
    tk.Label(main_app, text="Chọn thuật toán:", font=("Arial", 12), bg="#f0f8ff").pack(pady=5)
    tk.Radiobutton(main_app, text="Naive Bayes", variable=algorithm_var, value='Naive Bayes', font=("Arial", 12), bg="#f0f8ff").pack(pady=5)
    tk.Radiobutton(main_app, text="SVM", variable=algorithm_var, value='SVM', font=("Arial", 12), bg="#f0f8ff").pack(pady=5)

    # Nút so sánh thuật toán
    tk.Button(main_app, text="So sánh thuật toán", font=("Arial", 14), command=show_comparison_chart, bg="#4CAF50", fg="white", relief="solid", width=20).pack(pady=20)

    main_app.mainloop()

# Chạy ứng dụng
login_window()
