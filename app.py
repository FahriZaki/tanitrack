import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load model
model = load_model('modelskripsi_padi.h5')
labels = ['Blast', 'Blight', 'Tungro']

# Fungsi prediksi
def predict(image):
    img = load_img(image, target_size=(100, 100))  
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    return labels[class_idx], pred[0][class_idx]

# Sidebar Menu
st.sidebar.title("📋 Navigasi")
menu = st.sidebar.selectbox("Pilih Halaman", ["Beranda", "Perpustakaan", "Prediksi Penyakit"])

# Halaman BERANDA
if menu == "Beranda":
    st.image("background_TaniTrack.jpg", use_container_width=True)
    st.title("🌾 Selamat Datang di TaniTrack")
    st.markdown("""
    **TaniTrack** adalah aplikasi pintar berbasis *deep learning* yang dirancang untuk membantu petani dalam mendeteksi penyakit daun padi **secara otomatis**.

    Dengan hanya mengunggah gambar daun padi, TaniTrack dapat mengenali tiga jenis penyakit utama pada daun padi:
    - 🌿 **Blast** 
    - 🍂 **Blight** 
    - 🦠 **Tungro** 

    🔍 Klik menu **'Prediksi Penyakit'** di sidebar untuk mulai menganalisis daun padi Anda!  
    📚 Klik menu **'Perpustakaan'** di sidebar untuk mengetahui informasi tentang penyakit pada daun padi.
    """)


# Halaman PERPUSTAKAAN
elif menu == "Perpustakaan":
    st.header("📚 Perpustakaan Penyakit Daun Padi")
    st.markdown("Berikut informasi singkat mengenai penyakit yang menyerang tanaman padi:\n")

    # BLAST
    st.subheader("🌿 Blast")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("foto_blast.jpg", caption="Gejala Blast", width=200)
    with col2:
        st.info("""
        Penyakit blas yang disebabkan oleh Pyricularia oryzae merupakan penyakit penting pada tanaman padi di negara-negara penghasil padi di seluruh dunia. Saat ini terdapat tiga cara utama yang dapat dilakukan untuk mengendalikan penyakit blas, yaitu teknik budidaya, penggunaan fungisida dan varietas tahan.
        """)

    # BLIGHT
    st.subheader("🍂 Blight")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("foto_blight.jpg", caption="Gejala Blight", width=200)
    with col2:
        st.info("""
        Dikenal sebagai hawar daun bakteri, Penyakit disebabkan oleh bakteri Xanthomonas oryzae pv. oryzae (Xoo). Patogen ini dapat mengenfeksi tanaman padi pada semua fase pertumbuhan tanaman dari mulai pesemaian sampai menjelang panen. Penyebab penyakit (patogen) menginfeksi tanaman padi pada bagian daun melalui luka daun atau lobang alami berupa stomata dan merusak klorofil daun. Hal tersebut menyebabkan menurunnya kemampuan tanaman untuk melakukan fotosintesis yang apabila terjadi pada tanaman muda mengakibatkan mati 
        """)

    # TUNGRO
    st.subheader("🦠 Tungro")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("foto_tungro.jpg", caption="Gejala Tungro", width=200)
    with col2:
        st.info("""
        Penyakit virus yang ditularkan oleh wereng hijau, menyebabkan daun berwarna kuning oranye dan pertumbuhan tanaman terhambat. Penyakit tungro disebabkan juga oleh interaksi dua spesies virus yang tidak memiliki hubungan kekerabatan, yaitu Rice tungro bacilliform virus (RTBV) anggota famili Caulimoviridae dan Rice tungro spherical virus (RTSV) anggota famili Sequiviridae. Penyakit tungro berpotensi merugikan tanaman padi di kawasan Asia Tenggara dan Asia Selatan termasuk Indonesia.
        """)

    # BERCAK COKLAT
    st.subheader("🍂 Bercak Coklat")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("foto_bercakcokelat.jpg", caption="Gejala Bercak Coklat", width=200)
    with col2:
        st.info("""
        Penyakit bercak coklat disebabkan oleh jamur *Helminthosporium oryzae* (nama lama) atau *Bipolaris oryzae* (nama baru). Gejalanya berupa bercak-bercak coklat lonjong pada daun yang dapat meluas dan menyebabkan daun mengering. Penyakit ini biasanya menyerang tanaman padi yang lemah karena kekurangan unsur hara atau kondisi lingkungan yang tidak optimal.
        """)

    # BUSUK BATANG
    st.subheader("🌾 Busuk Batang")
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("foto_busukbatang.jpg", caption="Gejala Busuk Batang", width=200)
    with col2:
        st.info("""
        Busuk batang adalah penyakit yang disebabkan oleh jamur *Fusarium* spp. atau *Sclerotium oryzae*. Gejalanya meliputi pelepasan kulit batang yang menjadi lunak, berwarna coklat kehitaman, dan kadang-kadang mengeluarkan bau tidak sedap. Penyakit ini dapat menyebabkan tanaman rebah dan mati sebelum panen jika tidak ditangani.
        """)




# Halaman PREDIKSI PENYAKIT
elif menu == "Prediksi Penyakit":
    st.header("🔍 Prediksi Penyakit Daun Padi")
    st.markdown("Unggah gambar daun padi yang ingin Anda analisis.")
    
    uploaded_file = st.file_uploader("📁 Pilih gambar daun...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="🖼️ Gambar yang diunggah", use_container_width=False, width=200)
        
        with st.spinner("🔎 Menganalisis gambar..."):
            label, confidence = predict(uploaded_file)
        
        st.success("✅ Analisis selesai!")
        st.markdown(f"### 🌱 Hasil Prediksi: **{label}**")
        st.markdown(f"📊 Tingkat Keyakinan: **{confidence:.2%}**")

        if label == "Blast":
            st.info("**Saran Penanganan:**\n\nPengobatan benih dengan thiram efektif melawan penyakit ini. Fungsida yang mengandung azoksistrobin, atau bahan aktif dari triazol atau strobilurin juga dapat disemprotkan pada tahap pembibitan (satu atau dua kali pemberian fungsida pada tahap ini bisa efektif untuk mengendalikan penyakit ini) dan Periksa kelembaban lingkungan.")
        elif label == "Blight":
            st.info("**Saran Penanganan:**\n\nUntuk memerangi blight, perawatan benih dengan antibiotik resmi dengan tambahan tembaga oksiklorida atau tembaga sulfat sangat direkomendasikan dan kontrol irigasi.")
        elif label == "Tungro":
            st.info("**Saran Penanganan:**\n\nPengobatan dapat dilakukan dengan cara  melakukan penyemprotan insektisida berdasarkan buprofezin atau pimetrozin pada 15 dan 30 hari setelah proses penanaman berhasil dilakukan, untuk mengantisipasi penyebaran serangga dapat dilakukan juga penyemprotan insektisida disekitar tanaman yang terpapar penyakit tungro ini.")
