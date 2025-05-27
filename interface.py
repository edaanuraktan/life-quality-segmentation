import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# Sayfa yapısı ve başlık
st.set_page_config(layout="wide")
st.title("Şehirlerde Yaşam Kalitesi Segmentasyonu")
st.markdown("""
Türkiye'deki 81 ilin yaşam kalitesi verilerine göre segmentasyonunu yapabilir,
belirli bir ili arayabilir, filtreleyebilir, sıralayabilir ve benzer şehirleri keşfedebilirsiniz.
""")

# 1. Veri yükleme ve ön işleme
@st.cache_data
def load_and_prepare_data():
    df = pd.read_excel("data/life-quality-index-tuik.xlsx", engine="openpyxl")
    df = df.loc[:, ~df.columns.str.contains("Unnamed")]
    df.columns = df.columns.str.replace("\n", " ", regex=True).str.strip()
    df = df.fillna(df.mean(numeric_only=True))

    iller = df["İl Province"]
    X = df.drop(columns=["İl Province"])
    X_scaled = MinMaxScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
    kume_etiketleri = kmeans.fit_predict(X_scaled)

    df["Küme"] = kume_etiketleri
    kume_isimleri = {
        0: "Gelişmiş Şehirler",
        1: "Gelişmekte Olan Bölgeler",
        2: "Yüksek Yaşam Kalitesine Sahip İller",
        3: "Gelişim Potansiyeli Olan İller"
    }
    df["Küme Adı"] = df["Küme"].map(kume_isimleri)
    return df, X_scaled, X.columns

df, X_scaled, kategoriler = load_and_prepare_data()

# 2. Yan panelde filtreleme
st.sidebar.header("🔎 Filtreleme ve Arama")
secilen_il = st.sidebar.selectbox("Bir il seçin:", sorted(df["İl Province"].unique()))
secilen_kume = st.sidebar.multiselect("Küme seçin:", df["Küme Adı"].unique(), default=df["Küme Adı"].unique())

# 3. Filtrelenmiş iller listesi
st.subheader("📋 Filtrelenmiş Şehirler Listesi")
filtreli_df = df[df["Küme Adı"].isin(secilen_kume)]
st.dataframe(filtreli_df[["İl Province", "Küme Adı"] + list(kategoriler)].reset_index(drop=True), use_container_width=True, hide_index=True)

# 4. Seçilen il için detaylı sıralamalar
st.subheader(f"📍 {secilen_il} - Detaylı Sıralamalar")
detay_df = []
for kolon in kategoriler:
    sirali = df[["İl Province", kolon]].sort_values(by=kolon, ascending=False).reset_index(drop=True)
    sirali.index += 1  # 1'den başlayan sıralama
    satir = sirali[sirali["İl Province"] == secilen_il]
    if not satir.empty:
        detay_df.append((kolon, satir[kolon].values[0], satir.index[0]))
detay_df = pd.DataFrame(detay_df, columns=["Kategori", "Puan", "Sıra"]).sort_values("Sıra").reset_index(drop=True)
st.table(detay_df)

# 5. Seçilen ile benzer şehirler
st.subheader(f"🤝 {secilen_il} ile Benzer İller")
il_index = df[df["İl Province"] == secilen_il].index[0]
benzerlikler = cosine_similarity([X_scaled[il_index]], X_scaled)[0]
df["Benzerlik"] = benzerlikler
benzer_df = df[df["İl Province"] != secilen_il].sort_values("Benzerlik", ascending=False).head(5)
st.dataframe(benzer_df[["İl Province", "Küme Adı", "Benzerlik"]].reset_index(drop=True), use_container_width=True, hide_index=True)

# 6. Kategoriye göre tüm şehir sıralaması
st.subheader("📊 Kategoriye Göre İl Sıralaması")
kategori = st.selectbox("Bir kategori seçin:", kategoriler)
kategori_df = df[["İl Province", kategori]].sort_values(by=kategori, ascending=False).reset_index(drop=True)
kategori_df.index += 1
kategori_df = kategori_df.rename(columns={"İl Province": "İl", kategori: "Puan"})
kategori_df.insert(0, "Sıra", kategori_df.index)
st.dataframe(kategori_df.reset_index(drop=True), use_container_width=True, hide_index=True)
