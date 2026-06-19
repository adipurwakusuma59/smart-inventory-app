import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 0. Setelan Halaman
st.set_page_config(page_title="Global Smart Inventory", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1f77b4;}
    </style>
""", unsafe_allow_html=True)

st.title("🌍 Global Enterprise Inventory System")
st.write("Unggah data persediaan Anda dalam format apa pun. Sistem kami akan beradaptasi dengan data Anda.")

# 1. Kotak Upload di Sidebar
st.sidebar.header("📂 1. Unggah Data CSV")
file_unggahan = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])
st.sidebar.markdown("---")

if file_unggahan is not None:
    tabel_mentah = pd.read_csv(file_unggahan)
    kolom_csv = tabel_mentah.columns.tolist()

    # 2. FITUR MAPPING: Membiarkan user mencocokkan kolom mereka
    st.sidebar.header("⚙️ 2. Pemetaan Kolom (Mapping)")
    st.sidebar.write("Cocokkan nama kolom di file Anda dengan sistem kami:")
    
    # Trik UI/UX: Mengganti underscore (_) menjadi spasi khusus untuk tampilan dropdown
    format_tampilan = lambda nama: nama.replace('_', ' ')

    # HANYA ADA 4 DROPDOWN INI (Sudah rapi):
    col_nama = st.sidebar.selectbox("Mana kolom 'Nama Barang'?", kolom_csv, index=0, format_func=format_tampilan)
    col_stok = st.sidebar.selectbox("Mana kolom 'Stok Saat Ini'?", kolom_csv, index=min(1, len(kolom_csv)-1), format_func=format_tampilan)
    col_rop = st.sidebar.selectbox("Mana kolom 'Batas Aman (ROP)'?", kolom_csv, index=min(2, len(kolom_csv)-1), format_func=format_tampilan)
    col_penggunaan = st.sidebar.selectbox("Mana kolom 'Penggunaan Harian'?", kolom_csv, index=min(3, len(kolom_csv)-1), format_func=format_tampilan)

    # Tombol untuk memproses data setelah di-mapping
    if st.sidebar.button("🚀 Proses Data Sekarang"):
        
        # Menerjemahkan (Rename) kolom klien menjadi kolom standar sistem kita
        tabel_inventory = tabel_mentah.rename(columns={
            col_nama: 'Nama Barang',
            col_stok: 'Stok Saat Ini',
            col_rop: 'Batas Aman (ROP)',
            col_penggunaan: 'Penggunaan Harian'
        })
        
        # Pastikan kolom angka dibaca sebagai angka (mencegah error dari klien)
        tabel_inventory['Stok Saat Ini'] = pd.to_numeric(tabel_inventory['Stok Saat Ini'], errors='coerce').fillna(0)
        tabel_inventory['Batas Aman (ROP)'] = pd.to_numeric(tabel_inventory['Batas Aman (ROP)'], errors='coerce').fillna(0)
        tabel_inventory['Penggunaan Harian'] = pd.to_numeric(tabel_inventory['Penggunaan Harian'], errors='coerce').fillna(1) # hindari bagi 0

        st.success("✅ Pemetaan Berhasil! Data Anda sedang dianalisis...")

        # --- MESIN TEKNIK INDUSTRI ---
        tabel_inventory['Sisa Hari'] = (tabel_inventory['Stok Saat Ini'] / tabel_inventory['Penggunaan Harian']).round(1)
        tabel_inventory['Rekomendasi Waktu'] = np.where(
            tabel_inventory['Sisa Hari'] <= 7, '🚨 KRITIS (Pesan Sekarang)',
            np.where(tabel_inventory['Sisa Hari'] <= 14, '⚠️ WARNING (Siapkan PO)', '✅ AMAN')
        )

        # --- TAMPILAN DASHBOARD ---
        tab1, tab2, tab3 = st.tabs(["📊 Ringkasan Eksekutif", "📈 Analisis Forecast", "🗄️ Database Terjemahan"])

        with tab1:
            st.subheader("Key Performance Indicators (KPI)")
            col1, col2, col3 = st.columns(3)
            jumlah_kritis = len(tabel_inventory[tabel_inventory['Sisa Hari'] <= 7])
            rata_rata_hari = tabel_inventory['Sisa Hari'].mean().round(1)

            col1.metric("Total Jenis Barang", len(tabel_inventory), "Item aktif")
            col2.metric("Barang Status Kritis", jumlah_kritis, "- Butuh Perhatian!", delta_color="inverse")
            col3.metric("Rata-rata Ketahanan Stok", f"{rata_rata_hari} Hari")
            st.markdown("---")
            
            warna_status = {'🚨 KRITIS (Pesan Sekarang)': '#ef553b', '⚠️ WARNING (Siapkan PO)': '#feca28', '✅ AMAN': '#00cc96'}
            fig_bar = px.bar(tabel_inventory, x='Nama Barang', y='Stok Saat Ini', color='Rekomendasi Waktu', color_discrete_map=warna_status, text='Stok Saat Ini', title="Peta Status Persediaan")
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

        with tab2:
            st.subheader("Simulasi Penurunan Stok (30 Hari ke Depan)")
            hari = np.arange(0, 31)
            df_forecast = pd.DataFrame({'Hari Ke-': hari})
            for index, row in tabel_inventory.iterrows():
                df_forecast[row['Nama Barang']] = np.maximum(row['Stok Saat Ini'] - (row['Penggunaan Harian'] * hari), 0)
            
            df_forecast_melted = df_forecast.melt(id_vars=['Hari Ke-'], var_name='Nama Barang', value_name='Prediksi Stok')
            fig_line = px.line(df_forecast_melted, x='Hari Ke-', y='Prediksi Stok', color='Nama Barang', markers=True, title="Forecast Burn-down Chart")
            fig_line.add_hline(y=0, line_dash="solid", line_color="red")
            st.plotly_chart(fig_line, use_container_width=True)

        with tab3:
            st.subheader("Tabel Data (Setelah Disesuaikan dengan Standar Sistem)")
            st.dataframe(tabel_inventory, use_container_width=True)
            
            # --- FITUR BARU: Tombol Download ---
            st.markdown("---")
            csv_hasil = tabel_inventory.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Analisis (CSV)",
                data=csv_hasil,
                file_name="laporan_inventory_pintar.csv",
                mime="text/csv"
            )


else:
    st.info("Sistem menunggu unggahan data. Silakan masukkan file CSV di panel kiri.")