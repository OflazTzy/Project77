import os
import pickle
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta  # digunakan untuk penambahan bulan

from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'secret-key-anda'  # Ganti dengan secret key yang aman

# Load model Random Forest
with open('random_forest.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Custom InputLayer untuk menangani perbedaan konfigurasi (batch_shape vs batch_input_shape)
from tensorflow.keras.layers import InputLayer as _InputLayer
class CustomInputLayer(_InputLayer):
    @classmethod
    def from_config(cls, config):
        if 'batch_shape' in config:
            config['batch_input_shape'] = config.pop('batch_shape')
        return super().from_config(config)

# Coba impor DTypePolicy dari lokasi yang berbeda
try:
    from tensorflow.keras.mixed_precision.policy import DTypePolicy
except ModuleNotFoundError:
    from tensorflow.keras.mixed_precision import global_policy
    DTypePolicy = global_policy().__class__

# Gunakan custom_object_scope untuk mendaftarkan custom objects
from tensorflow.keras.utils import custom_object_scope

with custom_object_scope({'InputLayer': CustomInputLayer, 'DTypePolicy': DTypePolicy}):
    # Muat model LSTM tanpa mengompilasi (compile=False) untuk menghindari error optimizer
    lstm_model = load_model('lstm_model.h5', compile=False)
lstm_model.compile(optimizer='adam', loss='mse')


def generate_predictions(input_data):
    """
    Menghasilkan prediksi 12 bulan ke depan menggunakan kedua model.
    input_data: list dengan 11 fitur: [tahun, bulan, ihk, inflasi_umum,
                pdrb_perkapita_persen, umr, inflasi_pendidikan, inflasi_sd,
                inflasi_menengah, inflasi_tinggi, inflasi_lainnya]
    """
    # Pastikan input berbentuk (1, 11)
    input_array = np.array(input_data).reshape(1, -1)
    
    # Prediksi dengan Random Forest
    rf_pred = rf_model.predict(input_array)[0]
    
    # Pastikan input untuk LSTM berbentuk (1, 11, 1)
    lstm_input = input_array.reshape(1, input_array.shape[1], 1)
    lstm_pred = lstm_model.predict(lstm_input)[0][0]
    
    predictions_rf = []
    predictions_lstm = []
    
    # Buat prediksi untuk 12 bulan ke depan
    for i in range(12):
        predictions_rf.append(rf_pred + 0.1 * i)
        predictions_lstm.append(lstm_pred + 0.1 * i)
    
    # Hasil gabungan (rata-rata) sebagai prediksi final yang "paling mendekati"
    predictions_avg = [(a + b) / 2 for a, b in zip(predictions_rf, predictions_lstm)]
    
    return predictions_rf, predictions_lstm, predictions_avg


def generate_month_labels(tahun, bulan):
    """
    Menghasilkan label bulan selama 12 bulan ke depan, misalnya "Jan 2025", "Feb 2025", dll.
    Menggunakan relativedelta agar pergeseran bulan tepat.
    """
    labels = []
    start_date = datetime(year=tahun, month=bulan, day=1)
    for i in range(12):
        month_date = start_date + relativedelta(months=i)
        labels.append(month_date.strftime('%b %Y'))
    return labels


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    predictions_rf, predictions_lstm, predictions_avg, month_labels = None, None, None, None
    tahun, bulan = None, None  # Inisialisasi agar tidak error di render_template

    if request.method == 'POST':
        try:
            # Ambil input dari form
            tahun = int(request.form.get('tahun'))
            bulan = int(request.form.get('bulan'))
            ihk = float(request.form.get('ihk'))
            inflasi_umum = float(request.form.get('inflasi_umum'))
            pdrb_perkapita_persen = float(request.form.get('pdrb_perkapita_persen'))
            umr = float(request.form.get('umr'))
            inflasi_pendidikan = float(request.form.get('inflasi_pendidikan'))
            inflasi_sd = float(request.form.get('inflasi_sd'))
            inflasi_menengah = float(request.form.get('inflasi_menengah'))
            inflasi_tinggi = float(request.form.get('inflasi_tinggi'))
            inflasi_lainnya = float(request.form.get('inflasi_lainnya'))
            
            input_data = [tahun, bulan, ihk, inflasi_umum, pdrb_perkapita_persen,
                          umr, inflasi_pendidikan, inflasi_sd, inflasi_menengah,
                          inflasi_tinggi, inflasi_lainnya]
            
            # Dapatkan prediksi untuk 12 bulan ke depan
            predictions_rf, predictions_lstm, predictions_avg = generate_predictions(input_data)
            month_labels = generate_month_labels(tahun, bulan)
        
        except Exception as e:
            flash(f"Terjadi kesalahan: {e}", "danger")
            return redirect(url_for('prediksi'))
    
    return render_template('prediksi.html',
                           predictions_rf=predictions_rf,
                           predictions_lstm=predictions_lstm,
                           predictions_avg=predictions_avg,
                           month_labels=month_labels,
                           tahun=tahun,
                           bulan=bulan)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True)
