import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px  # Alat grafik interaktif modern

# 0. Setelan Halaman (Harus ditaruh paling atas)
st.set_page_config(page_title="Smart Inventory", layout="wide", initial_sidebar_state="expanded")

# --- CSS Tambahan untuk Mempercantik Tampilan ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

st.title("📦 Enterprise Smart Inventory & Forecaster")
st.write("Sistem Pemantauan dan Prediksi Rantai Pasok Berbasis AI.")

# 1. Kotak Upload di Sidebar
st.sidebar.header("📂 Masukkan Data Klien")
file_unggahan = st.sidebar.file_uploader("Upload File Data Gudang (Format CSV)", type=["csv"])
st.sidebar.markdown("---")

if file_unggahan is not None:
    tabel_inventory = pd.read_csv(file_unggahan)
    kolom_wajib = ['Nama Barang', 'Stok Saat Ini', 'Batas Aman (ROP)', 'Penggunaan Harian']
    
    if all(kolom in tabel_inventory.columns for kolom in kolom_wajib):
        
        # --- MESIN TEKNIK INDUSTRI ---
        tabel_inventory['Sisa Hari'] = (tabel_inventory['Stok Saat Ini'] / tabel_inventory['Penggunaan Harian']).round(1)
        tabel_inventory['Rekomendasi Waktu'] = np.where(
            tabel_inventory['Sisa Hari'] <= 7, '🚨 KRITIS (Pesan Sekarang)',
            np.where(tabel_inventory['Sisa Hari'] <= 14, '⚠️ WARNING (Siapkan PO)', '✅ AMAN')
        )

        # --- MEMBUAT TABS (HALAMAN MENU) ---
        tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Eksekutif", "📈 Analisis Forecast", "🗄️ Database Mentah"])

        # ==========================================
        # TAB 1: RINGKASAN EKSEKUTIF (KPI & Status)
        # ==========================================
        with tab1:
            st.subheader("Key Performance Indicators (KPI)")
            col1, col2, col3 = st.columns(3)
            
            jumlah_kritis = len(tabel_inventory[tabel_inventory['Sisa Hari'] <= 7])
            rata_rata_hari = tabel_inventory['Sisa Hari'].mean().round(1)

            col1.metric("Total Jenis Barang", len(tabel_inventory), "Item aktif")
            col2.metric("Barang Status Kritis", jumlah_kritis, "- Butuh Perhatian!", delta_color="inverse")
            col3.metric("Rata-rata Ketahanan Stok", f"{rata_rata_hari} Hari")
            
            st.markdown("---")
            st.subheader("Peta Status Persediaan")
            
            # Grafik Plotly: Bar Chart dengan Warna Otomatis berdasarkan Status
            warna_status = {
                '🚨 KRITIS (Pesan Sekarang)': '#ef553b', # Merah
                '⚠️ WARNING (Siapkan PO)': '#feca28',   # Kuning
                '✅ AMAN': '#00cc96'                     # Hijau
            }
            
            fig_bar = px.bar(
                tabel_inventory, 
                x='Nama Barang', 
                y='Stok Saat Ini', 
                color='Rekomendasi Waktu',
                color_discrete_map=warna_status,
                text='Stok Saat Ini',
                title="Perbandingan Stok Barang Berdasarkan Tingkat Urgensi"
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        # ==========================================
        # TAB 2: ANALISIS FORECAST (Prediksi Habis)
        # ==========================================
        with tab2:
            st.subheader("Simulasi Penurunan Stok (30 Hari ke Depan)")
            st.write("Grafik ini memprediksi bagaimana stok akan habis jika kecepatan produksi konstan.")
            
            # Membuat data simulasi 30 hari ke depan
            hari = np.arange(0, 31)
            df_forecast = pd.DataFrame({'Hari Ke-': hari})
            
            for index, row in tabel_inventory.iterrows():
                # Rumus: Stok = Stok Awal - (Penggunaan Harian * Hari). Tidak boleh minus (max 0).
                df_forecast[row['Nama Barang']] = np.maximum(row['Stok Saat Ini'] - (row['Penggunaan Harian'] * hari), 0)
            
            # Merapikan data untuk grafik (Melt)
            df_forecast_melted = df_forecast.melt(id_vars=['Hari Ke-'], var_name='Nama Barang', value_name='Prediksi Stok')
            
            # Grafik Plotly: Line Chart Interaktif
            fig_line = px.line(
                df_forecast_melted, 
                x='Hari Ke-', 
                y='Prediksi Stok', 
                color='Nama Barang',
                markers=True,
                title="Forecast Burn-down Chart"
            )
            # Menambahkan garis batas ROP sebagai referensi
            fig_line.add_hline(y=0, line_dash="solid", line_color="red", annotation_text="Stok Habis (0)")
            st.plotly_chart(fig_line, use_container_width=True)

        # ==========================================
        # TAB 3: DATABASE MENTAH
        # ==========================================
        with tab3:
            st.subheader("Tabel Data Inventaris Interaktif")
            
            # Filter Data
            pilihan_status = st.selectbox("Saring berdasarkan status:", ["Semua Status"] + list(tabel_inventory['Rekomendasi Waktu'].unique()))
            if pilihan_status != "Semua Status":
                tabel_tampil = tabel_inventory[tabel_inventory['Rekomendasi Waktu'] == pilihan_status]
            else:
                tabel_tampil = tabel_inventory
                
            st.dataframe(tabel_tampil, use_container_width=True)

    else:
        st.error("❌ Format file salah. Pastikan kolom sesuai template.")

else:
    st.info("Sistem menunggu unggahan data. Silakan masukkan file CSV di panel kiri.")