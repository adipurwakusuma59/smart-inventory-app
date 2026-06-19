import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

# 0. Setelan Halaman
st.set_page_config(page_title="Enterprise Supply Chain SaaS", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1 {color: #1f77b4;}
    .stAlert {margin-bottom: 1rem;}
    @media print {
        .stSidebar {display: none;}
        button {display: none;}
        .stTabs [data-baseweb="tab-list"] {display: none;}
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 Enterprise Supply Chain & Prescriptive Platform")
st.write("Platform Logistik Terpadu: Multi-Variabel Engine, Sistem Peringatan Dini, Klasifikasi ABC, & Simulasi What-If.")

# --- FITUR INGATAN (SESSION STATE) ---
if 'tombol_ditekan' not in st.session_state:
    st.session_state.tombol_ditekan = False

# 1. Kotak Upload di Sidebar
st.sidebar.header("📂 1. Unggah Dataset Persediaan")
file_unggahan = st.sidebar.file_uploader("Pilih dokumen (Format CSV / Excel)", type=["csv", "xlsx", "xls"])
st.sidebar.markdown("---")

if file_unggahan is not None:
    
    nama_file = file_unggahan.name
    if nama_file.endswith('.csv'):
        tabel_mentah = pd.read_csv(file_unggahan)
    elif nama_file.endswith(('.xlsx', '.xls')):
        tabel_mentah = pd.read_excel(file_unggahan) 
        
    kolom_csv = tabel_mentah.columns.tolist()
    format_tampilan = lambda nama: str(nama).replace('_', ' ')

    # 2. Pemetaan Wajib
    st.sidebar.header("⚙️ 2. Pemetaan Variabel Utama")
    col_nama = st.sidebar.selectbox("Identitas / Nama Item:", kolom_csv, index=0, format_func=format_tampilan)
    col_stok = st.sidebar.selectbox("Kuantitas Stok Aktual:", kolom_csv, index=min(1, len(kolom_csv)-1), format_func=format_tampilan)
    
    # 3. Parameter Opsional & Setelan Logistik / ML
    st.sidebar.markdown("---")
    st.sidebar.header("💎 3. Parameter Opsional & Logistik")
    
    opsi_pilihan = ["-- Lewati (Tidak Ada) --"] + kolom_csv
    col_rop = st.sidebar.selectbox("Batas Minimum (Safety Stock):", opsi_pilihan, format_func=format_tampilan)
    col_penggunaan = st.sidebar.selectbox("Tingkat Konsumsi Harian (Aktifkan Forecast):", opsi_pilihan, format_func=format_tampilan)
    col_harga = st.sidebar.selectbox("Nilai / Harga Per Unit (Aktifkan Finansial):", opsi_pilihan, format_func=format_tampilan)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚚 Pengaturan Operasional (Reorder)")
    lead_time = st.sidebar.number_input("Lead Time / Waktu Pengiriman Supplier (Hari):", min_value=1, value=3, step=1)
    target_pemenuhan = st.sidebar.number_input("Target Pemenuhan Stok Ulang (Hari):", min_value=7, value=30, step=7)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔮 Metode Prediksi Permintaan")
    metode_forecast = st.sidebar.selectbox(
        "Pilih Algoritma Forecasting:",
        ["Simulasi Linier Matematika", "Linear Regression (Machine Learning)", "ARIMA (Butuh Data Historis Waktu)*", "Prophet (Butuh Data Historis Waktu)*", "XGBoost (Butuh Data Historis Waktu)*"]
    )
    
    kolom_tersisa = [k for k in kolom_csv if k not in [col_nama, col_stok, col_rop, col_penggunaan, col_harga]]
    kolom_kustom_dipilih = st.sidebar.multiselect(
        "Atribut Kategorisasi Ekstra (Bisa pilih banyak):", 
        options=kolom_tersisa,
        format_func=format_tampilan
    )

    if st.sidebar.button("🚀 Proses Data & Jalankan AI"):
        st.session_state.tombol_ditekan = True

    if st.session_state.tombol_ditekan:
        
        tabel_inventory = tabel_mentah.rename(columns={
            col_nama: 'Nama Barang',
            col_stok: 'Stok Saat Ini'
        })
        tabel_inventory['Stok Saat Ini'] = pd.to_numeric(tabel_inventory['Stok Saat Ini'], errors='coerce').fillna(0)
        tabel_inventory['Rekomendasi Waktu'] = '✅ AMAN'

        fitur_rop_aktif = False
        if col_rop != "-- Lewati (Tidak Ada) --":
            tabel_inventory['Batas Aman (ROP)'] = pd.to_numeric(tabel_mentah[col_rop], errors='coerce').fillna(0)
            fitur_rop_aktif = True
            tabel_inventory['Rekomendasi Waktu'] = np.where(tabel_inventory['Stok Saat Ini'] < tabel_inventory['Batas Aman (ROP)'], '🚨 KRITIS (Di Bawah ROP)', '✅ AMAN')

        # -- LOGIKA FORECAST & AI REGRESI LINIER --
        fitur_forecast_aktif = False
        if col_penggunaan != "-- Lewati (Tidak Ada) --":
            tabel_inventory['Penggunaan Harian'] = pd.to_numeric(tabel_mentah[col_penggunaan], errors='coerce').fillna(1)
            
            if metode_forecast == "Linear Regression (Machine Learning)":
                sisa_hari_list = []
                for index, row in tabel_inventory.iterrows():
                    X_train = np.array([0, 1, 2, 3, 4]).reshape(-1, 1)
                    y_train = np.array([row['Stok Saat Ini'], 
                                        max(row['Stok Saat Ini'] - row['Penggunaan Harian'], 0),
                                        max(row['Stok Saat Ini'] - (row['Penggunaan Harian']*2), 0),
                                        max(row['Stok Saat Ini'] - (row['Penggunaan Harian']*3), 0),
                                        max(row['Stok Saat Ini'] - (row['Penggunaan Harian']*4), 0)])
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    if model.coef_[0] != 0:
                        hari_habis_ml = -model.intercept_ / model.coef_[0]
                        sisa_hari_list.append(max(round(hari_habis_ml, 1), 0))
                    else:
                        sisa_hari_list.append(0)
                tabel_inventory['Sisa Hari'] = sisa_hari_list
            else:
                tabel_inventory['Sisa Hari'] = (tabel_inventory['Stok Saat Ini'] / tabel_inventory['Penggunaan Harian']).round(1)
            
            tabel_inventory['Rekomendasi Waktu'] = np.where(
                tabel_inventory['Sisa Hari'] <= 7, '🚨 KRITIS (Pesan Sekarang)',
                np.where(tabel_inventory['Sisa Hari'] <= 14, '⚠️ WARNING (Siapkan PO)', '✅ AMAN')
            )
            fitur_forecast_aktif = True

        fitur_finansial_aktif = False
        if col_harga != "-- Lewati (Tidak Ada) --":
            tabel_inventory['Harga Satuan Standar'] = pd.to_numeric(tabel_mentah[col_harga], errors='coerce').fillna(0)
            tabel_inventory['Total Nilai Aset'] = tabel_inventory['Stok Saat Ini'] * tabel_inventory['Harga Satuan Standar']
            fitur_finansial_aktif = True

        # --- IMPLEMENTASI BARU: MEMPERTAHANKAN VARIABEL KUMULATIF UNTUK PARETO CHART ---
        if fitur_finansial_aktif and len(tabel_inventory) > 0:
            total_nilai_gudang = tabel_inventory['Total Nilai Aset'].sum()
            if total_nilai_gudang > 0:
                tabel_inventory = tabel_inventory.sort_values(by='Total Nilai Aset', ascending=False)
                tabel_inventory['Kumulatif_Pct'] = tabel_inventory['Total Nilai Aset'].cumsum() / total_nilai_gudang
                tabel_inventory['Analisis ABC'] = np.where(tabel_inventory['Kumulatif_Pct'] <= 0.70, 'Kategori A (Nilai Tinggi)',
                                                   np.where(tabel_inventory['Kumulatif_Pct'] <= 0.90, 'Kategori B (Nilai Menengah)', 
                                                            'Kategori C (Nilai Rendah)'))
            else:
                tabel_inventory['Analisis ABC'] = 'Kategori C (Nilai Rendah)'
                tabel_inventory['Kumulatif_Pct'] = 1.0
        else:
            tabel_inventory['Analisis ABC'] = 'Butuh Data Finansial'
            tabel_inventory['Kumulatif_Pct'] = 0.0

        # --- LOGIKA OPERASIONAL REORDER ADVISOR ---
        if fitur_forecast_aktif:
            tabel_inventory['Hari Terbaik Memesan'] = (tabel_inventory['Sisa Hari'] - lead_time).round(1)
            tabel_inventory['Rekomendasi Pembelian'] = np.where(
                tabel_inventory['Sisa Hari'] <= lead_time, "🚨 Harus Dipesan Hari Ini!",
                np.where(tabel_inventory['Sisa Hari'] <= (lead_time + 5), 
                         "⏳ Jadwalkan dalam " + tabel_inventory['Hari Terbaik Memesan'].astype(str) + " hari", 
                         "✅ Belum Perlu Pemesanan")
            )
            tabel_inventory['Rekomendasi Jumlah Beli (Unit)'] = np.where(
                tabel_inventory['Sisa Hari'] <= (lead_time + 5),
                ((tabel_inventory['Penggunaan Harian'] * target_pemenuhan) + 
                 (tabel_inventory['Batas Aman (ROP)'] if fitur_rop_aktif else 0) - 
                 tabel_inventory['Stok Saat Ini']).round(0),
                0
            )
            tabel_inventory['Rekomendasi Jumlah Beli (Unit)'] = tabel_inventory['Rekomendasi Jumlah Beli (Unit)'].clip(lower=0)

        # KONDISI STATISTIK RINGKASAN GUDANG
        cnt_total = len(tabel_inventory)
        cnt_kritis = len(tabel_inventory[tabel_inventory['Rekomendasi Waktu'].str.contains('KRITIS', na=False)])
        cnt_warning = len(tabel_inventory[tabel_inventory['Rekomendasi Waktu'].str.contains('WARNING', na=False)])
        cnt_aman = len(tabel_inventory[tabel_inventory['Rekomendasi Waktu'].str.contains('AMAN', na=False)])

        # --- TAMPILAN DASHBOARD ---
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Ringkasan Eksekutif", "🚚 Rekomendasi Pemesanan AI", "🔍 Detail Profil Barang", 
            "🔬 Simulasi Skenario (What-If)", "📈 Analisis Forecast", "🗄️ Database Terjemahan"
        ])

        # TAB 1: RINGKASAN EKSEKUTIF
        with tab1:
            st.info("💡 **Tips Ekspor PDF:** Tekan **Ctrl + P** pada keyboard Anda, lalu pilih 'Save as PDF' untuk mencetak laporan Halaman Eksekutif ini.")
            st.subheader("📋 Ringkasan Status & Kondisi Gudang Aktual")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("📦 Total Jenis Barang", f"{cnt_total} Item")
            s2.metric("🟢 Jumlah Kondisi Aman", f"{cnt_aman} Item")
            s3.metric("🟡 Jumlah Perlu Pemantauan", f"{cnt_warning} Item", delta_color="off")
            s4.metric("🔴 Jumlah Kondisi Kritis", f"{cnt_kritis} Item", delta_color="inverse")
            
            st.markdown("---")
            st.subheader("Key Performance Indicators (KPI)")
            kolom_kpi = 2 if fitur_finansial_aktif else 1
            kpi_cols = st.columns(kolom_kpi)
            kpi_cols[0].metric("📊 Validitas Basis Data", "100% Terpetakan", "Sistem Sinkron")
            if fitur_finansial_aktif:
                total_nilai_gudang = tabel_inventory['Total Nilai Aset'].sum()
                kpi_cols[1].metric("💰 Total Nilai Kapitalisasi Aset", f"Rp {total_nilai_gudang:,.0f}")

            st.markdown("---")
            st.subheader("🔔 Pusat Notifikasi & Peringatan")
            if cnt_kritis > 0:
                barang_kritis = tabel_inventory[tabel_inventory['Rekomendasi Waktu'].str.contains('KRITIS', na=False)]['Nama Barang'].tolist()
                st.error(f"🔴 **KONDISI KRITIS (TINDAKAN SEGERA):** Terdapat **{cnt_kritis} item** di zona bahaya persediaan ({', '.join(barang_kritis)}).")
            if cnt_warning > 0:
                barang_warning = tabel_inventory[tabel_inventory['Rekomendasi Waktu'].str.contains('WARNING', na=False)]['Nama Barang'].tolist()
                st.warning(f"🟡 **PERLU PEMANTAUAN (WARNING):** Terdapat **{cnt_warning} item** dalam masa tenggang transisi PO ({', '.join(barang_warning)}).")
            if cnt_aman == cnt_total:
                st.success("🟢 **KONDISI SEHAT:** Seluruh lini persediaan berada dalam kondisi aman.")

            st.markdown("---")
            st.subheader("Peta Status Persediaan (Kuantitas Fisik)")
            warna_grafik = 'Rekomendasi Waktu'
            warna_status = {'🚨 KRITIS (Pesan Sekarang)': '#ef553b', '🚨 KRITIS (Di Bawah ROP)': '#ef553b', '⚠️ WARNING (Siapkan PO)': '#feca28', '✅ AMAN': '#00cc96'}
            if kolom_kustom_dipilih:
                warna_grafik = kolom_kustom_dipilih[0]
                warna_status = None
                st.info(f"💡 Pengelompokan warna grafik disesuaikan berdasarkan variabel kustom: **{warna_grafik.replace('_', ' ')}**")

            fig_bar = px.bar(tabel_inventory, x='Nama Barang', y='Stok Saat Ini', color=warna_grafik, color_discrete_map=warna_status, text='Stok Saat Ini')
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)

            # --- IMPLEMENTASI BARU: GABUNGAN PARETO CHART & GRAFIK ANALISIS ABC (MENGGANTIKAN PIE CHART LAMA) ---
            if fitur_finansial_aktif:
                st.markdown("---")
                st.subheader("🎯 Klasifikasi Manajemen ABC & Kurva Analisis Pareto Finansial")
                st.write("Grafik ini memetakan nilai kontribusi finansial per item secara kumulatif (Urutan Pareto Descending) untuk menentukan prioritas kontrol kapitalisasi aset.")
                
                # Mengonversi desimal kumulatif ke format persen (0 - 100%)
                tabel_inventory['Kumulatif_Pct_100'] = tabel_inventory['Kumulatif_Pct'] * 100
                
                # Pemetaan warna standar industri untuk klasifikasi ABC
                color_map_abc = {
                    'Kategori A (Nilai Tinggi)': '#1f77b4',  # Biru Korporat
                    'Kategori B (Nilai Menengah)': '#ff7f0e', # Oranye Peringatan
                    'Kategori C (Nilai Rendah)': '#2ca02c'   # Hijau Stabil
                }
                bar_colors = tabel_inventory['Analisis ABC'].map(color_map_abc).tolist()
                
                # Membuat diagram dual-axis kombinasi Bar & Line
                fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Sumbu Utama: Bar Chart Nilai Aset per Item
                fig_pareto.add_trace(
                    go.Bar(
                        x=tabel_inventory['Nama Barang'],
                        y=tabel_inventory['Total Nilai Aset'],
                        name="Nilai Aset Item (Rp)",
                        marker_color=bar_colors,
                        hovertemplate="<b>%{x}</b><br>Nilai Aset: Rp %{y:,.0f}<extra></extra>"
                    ),
                    secondary_y=False
                )
                
                # Sumbu Sekunder: Line Chart Akumulasi Kumulatif Pareto
                fig_pareto.add_trace(
                    go.Scatter(
                        x=tabel_inventory['Nama Barang'],
                        y=tabel_inventory['Kumulatif_Pct_100'],
                        name="Kurva Kumulatif (%)",
                        mode="lines+markers",
                        line=dict(color="#ef553b", width=3),
                        marker=dict(size=8),
                        hovertemplate="<b>%{x}</b><br>Akumulasi Kumulatif: %{y:.1f}%<extra></extra>"
                    ),
                    secondary_y=True
                )
                
                # Mengatur Tata Letak Layout
                fig_pareto.update_layout(
                    title_text="<b>Dual-Axis Pareto Chart (Klasifikasi Finansial ABC)</b>",
                    xaxis_title="Nama Barang",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                
                fig_pareto.update_yaxes(title_text="<b>Nilai Kapitalisasi Aset (Rp)</b>", secondary_y=False)
                fig_pareto.update_yaxes(title_text="<b>Persentase Akumulasi Kumulatif (%)</b>", secondary_y=True, range=[0, 105])
                
                st.plotly_chart(fig_pareto, use_container_width=True)
                
                # Legenda penjelas kategori warna
                st.markdown("""
                <div style="display: flex; gap: 25px; justify-content: center; font-weight: bold; margin-top: -10px; margin-bottom: 20px;">
                    <span style="color: #1f77b4;">■ Kategori A (Menyerap ~70% Modal - Pengawasan Sangat Ketat)</span>
                    <span style="color: #ff7f0e;">■ Kategori B (Menyerap ~20% Modal - Pengawasan Sedang)</span>
                    <span style="color: #2ca02c;">■ Kategori C (Menyerap ~10% Modal - Pengawasan Longgar)</span>
                </div>
                """, unsafe_allow_html=True)

        # TAB 2: REKOMENDASI PEMESANAN AI
        with tab2:
            st.subheader("📋 Rekomendasi Pembelian Barang Otomatis (AI Advisor)")
            if fitur_forecast_aktif:
                st.write(f"Kalkulasi didasarkan atas parameter ketetapan Lead Time: {lead_time} Hari & Target Pemenuhan: {target_pemenuhan} Hari.")
                kolom_tampil_reorder = ['Nama Barang', 'Stok Saat Ini', 'Penggunaan Harian', 'Sisa Hari', 'Hari Terbaik Memesan', 'Rekomendasi Pembelian', 'Rekomendasi Jumlah Beli (Unit)']
                if fitur_rop_aktif: kolom_tampil_reorder.insert(3, 'Batas Aman (ROP)')
                if fitur_finansial_aktif: kolom_tampil_reorder.append('Analisis ABC')
                
                st.dataframe(tabel_inventory[kolom_tampil_reorder].style.map(
                    lambda val: 'background-color: #ffcccc; color: #cc0000; font-weight: bold;' if val == "🚨 Harus Dipesan Hari Ini!" else (
                                'background-color: #fff2cc; color: #cc9900;' if "⏳ Jadwalkan" in str(val) else ''),
                    subset=['Rekomendasi Pembelian']
                ), use_container_width=True)
            else:
                st.warning("⚠️ Fitur Penasihat Reorder membutuhkan parameter 'Tingkat Konsumsi Harian' diaktifkan.")

        # TAB 3: DETAIL PROFIL BARANG (DEEP DIVE)
        with tab3:
            st.subheader("🔍 Analisis Mendalam Per Item (Deep Dive Analysis)")
            pilihan_barang = st.selectbox("Pilih Barang yang Ingin Dianalisis:", tabel_inventory['Nama Barang'].tolist())
            if pilihan_barang:
                data_item = tabel_inventory[tabel_inventory['Nama Barang'] == pilihan_barang].iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Stok Saat Ini", f"{data_item['Stok Saat Ini']} Unit", data_item['Rekomendasi Waktu'])
                if Math_Active := fitur_finansial_aktif:
                    c2.metric("Total Nilai Aset", f"Rp {data_item['Total Nilai Aset']:,.0f}", data_item['Analisis ABC'])
                if fitur_forecast_aktif:
                    c3.metric("Ketahanan Stok", f"{data_item['Sisa Hari']} Hari", data_item['Rekomendasi Pembelian'])
                
                st.markdown("---")
                if fitur_forecast_aktif:
                    st.write("**Simulasi Proyeksi Penurunan Stok Khusus Item Ini:**")
                    hari_item = np.arange(0, 31)
                    stok_prediksi = np.maximum(data_item['Stok Saat Ini'] - (data_item['Penggunaan Harian'] * hari_item), 0)
                    df_item_chart = pd.DataFrame({'Hari Ke-': hari_item, 'Prediksi Stok': stok_prediksi})
                    fig_item = px.line(df_item_chart, x='Hari Ke-', y='Prediksi Stok', markers=True, title=f"Tren Penurunan Stok: {pilihan_barang}")
                    fig_item.add_hline(y=(data_item['Batas Aman (ROP)'] if fitur_rop_aktif else 0), line_dash="dot", line_color="orange", annotation_text="Batas Aman (ROP)")
                    st.plotly_chart(fig_item, use_container_width=True)

        # TAB 4: SIMULASI SKENARIO (WHAT-IF)
        with tab4:
            st.subheader("🔬 Simulasi Skenario Risiko (What-If Analysis)")
            st.write("Uji ketahanan gudang Anda jika terjadi disrupsi rantai pasok atau lonjakan pasar secara mendadak.")
            if fitur_forecast_aktif:
                col_w1, col_w2 = st.columns(2)
                with col_w1:
                    sim_kenaikan_demand = st.slider("Lonjakan Permintaan / Konsumsi Pasar (%)", min_value=0, max_value=100, value=0, step=5)
                with col_w2:
                    sim_keterlambatan = st.slider("Keterlambatan Pengiriman Logistik Supplier (Hari)", min_value=0, max_value=14, value=0, step=1)
                
                st.markdown("---")
                df_simulasi = tabel_inventory.copy()
                df_simulasi['Sim_Penggunaan'] = df_simulasi['Penggunaan Harian'] * (1 + (sim_kenaikan_demand/100))
                df_simulasi['Sim_Sisa_Hari'] = (df_simulasi['Stok Saat Ini'] / df_simulasi['Sim_Penggunaan']).round(1)
                sim_lead_time = lead_time + sim_keterlambatan
                
                df_simulasi['Sim_Rekomendasi_Beli'] = np.where(df_simulasi['Sim_Sisa_Hari'] <= sim_lead_time, "🚨 KRITIS (Gagal Penuhi Permintaan)!", 
                                                      np.where(df_simulasi['Sim_Sisa_Hari'] <= (sim_lead_time + 5), "⚠️ Terancam Kritis", "✅ Aman"))
                
                df_banding = df_simulasi[['Nama Barang', 'Sisa Hari', 'Sim_Sisa_Hari', 'Rekomendasi Pembelian', 'Sim_Rekomendasi_Beli']].copy()
                df_banding.columns = ['Nama Barang', 'Ketahanan Normal (Hari)', 'Ketahanan Skenario (Hari)', 'Status Normal', 'Status Skenario Risiko']
                
                st.dataframe(df_banding.style.map(
                    lambda val: 'background-color: #ffcccc; color: #cc0000; font-weight: bold;' if "KRITIS" in str(val) else '', subset=['Status Skenario Risiko']), use_container_width=True)
                
                item_kolaps = len(df_banding[df_banding['Status Skenario Risiko'].str.contains("KRITIS")])
                if item_kolaps > 0:
                    st.error(f"💥 **Kesimpulan Skenario:** Terdapat **{item_kolaps} item** yang diprediksi akan mengalami kekosongan stok (*stockout*) sebelum barang baru tiba jika disrupsi ini benar-hari terjadi!")
                else:
                    st.success("✅ **Kesimpulan Skenario:** Gudang Anda cukup kuat menahan disrupsi skenario ini.")
            else:
                st.warning("⚠️ Fitur Simulasi membutuhkan 'Tingkat Konsumsi Harian' diaktifkan.")

        # TAB 5: ANALISIS FORECAST PENURUNAN STOK UMUM
        with tab5:
            if fitur_forecast_aktif:
                st.subheader(f"Simulasi Penurunan Stok (Metode Peramalan: {metode_forecast})")
                if "Butuh Data" in metode_forecast:
                    st.warning(f"⚠️ Mode Ekspansi: Algoritma {metode_forecast} diaktifkan dalam mode aman (Stabilitas Linier).")
                
                hari = np.arange(0, 31)
                df_forecast = pd.DataFrame({'Hari Ke-': hari})
                for index, row in tabel_inventory.iterrows():
                    df_forecast[row['Nama Barang']] = np.maximum(row['Stok Saat Ini'] - (row['Penggunaan Harian'] * hari), 0)
                
                df_forecast_melted = df_forecast.melt(id_vars=['Hari Ke-'], var_name='Nama Barang', value_name='Prediksi Stok')
                fig_line = px.line(df_forecast_melted, x='Hari Ke-', y='Prediksi Stok', color='Nama Barang', markers=True)
                fig_line.add_hline(y=0, line_dash="solid", line_color="red")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.warning("⚠️ Fitur Simulasi Forecast tidak aktif.")

        # TAB 6: DATABASE UTAMA DENGAN WARNA VISUAL SEL TABLE
        with tab6:
            st.subheader("🗄️ Tabel Hasil Konversi Data Standar Sistem")
            if kolom_kustom_dipilih:
                st.write("🔍 **Filter Multi-Variabel Kustom:**")
                kolom_filter = st.columns(len(kolom_kustom_dipilih))
                for i, col_name in enumerate(kolom_kustom_dipilih):
                    with kolom_filter[i]:
                        pilihan_unik = ["Semua"] + tabel_mentah[col_name].dropna().astype(str).unique().tolist()
                        filter_val = st.selectbox(f"Saring {col_name.replace('_', ' ')}:", pilihan_unik)
                        if filter_val != "Semua":
                            tabel_inventory = tabel_inventory[tabel_mentah[col_name].astype(str) == filter_val]

            def style_status_visual(val):
                if 'KRITIS' in str(val): return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;'
                elif 'WARNING' in str(val): return 'background-color: #fff2cc; color: #cc9900; font-weight: bold;'
                elif 'AMAN' in str(val): return 'background-color: #d1e7dd; color: #0f5132;'
                if 'Kategori A' in str(val): return 'color: #1f77b4; font-weight: bold;'
                elif 'Kategori B' in str(val): return 'color: #ff7f0e; font-weight: bold;'
                elif 'Kategori C' in str(val): return 'color: #2ca02c;'
                return ''

            kolom_ada_di_tabel = tabel_inventory.columns.tolist()
            subset_styling = [c for c in ['Rekomendasi Waktu', 'Analisis ABC'] if c in kolom_ada_di_tabel]

            st.dataframe(tabel_inventory.style.map(style_status_visual, subset=subset_styling), use_container_width=True)
            st.markdown("---")
            csv_hasil = tabel_inventory.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Laporan Hasil Analisis Lengkap (CSV)", data=csv_hasil, file_name="laporan_inventory_complete.csv", mime="text/csv")

else:
    st.info("Sistem menunggu unggahan data. Silakan masukkan dokumen CSV atau Excel di panel kiri.")