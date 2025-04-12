import pickle
import pandas as pd
import os
import time

# Load model, scaler dan encoder dengan format pickle
MODEL_PATH = "model_gbr.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "encoder.pkl"

with open(MODEL_PATH, "rb") as file: # Model loading
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file: # Scaler loading
    scaler = pickle.load(file)

with open(ENCODER_PATH, "rb") as file: # Scaler loading
    encoder = pickle.load(file)

# Function untuk membuat prediksi model
def score_predict(model, scaler, encoder, dataframe):
    num_cols = ['Attendance', 'Hours_Studied', 'Previous_Scores', 'Sleep_Hours','Tutoring_Sessions'] # List kolom numerik yang akan discaling
    category_cols = ['Peer_Influence', 'Motivation_Level', 'Teacher_Quality', 'Access_to_Resources'] # List kollom kategorikal yang akan di inverse

    dataframe[num_cols] = scaler.transform(dataframe[num_cols]) # Kode untuk scaling
    result = model.predict(dataframe) # Kode untuk prediksi
    inversed = encoder.inverse_transform(dataframe[category_cols]) # Kode untuk inverse transform

    return result[0], inversed

print("======================================")
print("Nilaiku".center(38))
print("======================================\n")

# iputan untuk kolom yang sudah dipilih
attendance = int(input("1. Masukkan persentase kehadiranmu dikelas (1-100): ")) 
hours_studied = int(input("2. Masukkan jumlah jam belajar per minggu: "))
previous_scores = int(input("3. Masukkan nilai ujian sebelumnya: "))
sleep_hours = int(input("4. Masukkan rata-rata jam tidur per malam: "))
tutoring_sessions = int(input("5. Masukkan jumlah sesi bimbingan belajar yang kamu hadiri per bulan: "))
peer_influence = int(input("6. Pengaruh teman sebaya? (0:Positive, 1:Neutral, 2:Negative): "))
motivation_level = int(input("7. Tingkat motivasi (0:Low, 1:Medium, 2:High): "))
teacher_quality = int(input("8. Kualitas guru (0:Low, 1:Medium, 2:High): "))
access_to_resources = int(input("9. Ketersediaan sumber daya pembelajaran (0:Low, 1:Medium, 2:High): "))

# Membuat data inputan user menjadi dataframe
user_data = pd.DataFrame({
    'Attendance': [attendance],
    'Hours_Studied': [hours_studied],
    'Previous_Scores': [previous_scores],
    'Sleep_Hours': [sleep_hours],
    'Tutoring_Sessions': [tutoring_sessions],
    'Peer_Influence': [peer_influence],
    'Motivation_Level': [motivation_level],
    'Teacher_Quality': [teacher_quality],
    'Access_to_Resources': [access_to_resources]
})


# Membuat prediksi
os.system("cls")
model_prediction, inversed_column = score_predict(model, scaler, encoder, user_data)


# Menampilkan output hasil prediksi
print("=========================================")
print("Ringkasan Input".center(40))
print("=========================================\n")
print(f"* Kehadiran: {attendance}%")
print(f"* Jam belajar: {hours_studied} jam")
print(f"* Nilai ujian sebelumnya: {previous_scores}")
print(f"* Jumlah sesi bimbel: {tutoring_sessions}")
print(f"* Pengaruh teman: {inversed_column[0][0]}")
print(f"* Tingkat motivasi: {inversed_column[0][1]}")
print(f"* Kualitas guru: {inversed_column[0][2]}")
print(f"* Sumber daya pembelajaran: {inversed_column[0][3]}")
print(f"Membuat prediksi....\n")
time.sleep(2)
print("\n=========================================")
print(f"Nilai ujian yang diprediksi model: {model_prediction:.4}")
print("=========================================\n")
