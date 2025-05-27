import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


st.set_page_config(page_title="Yaşam Kalitesi Bölge Segmentasyonu", layout="wide")
st.title("Şehirlerde Yaşam Kalitesi Segmentasyonu")
st.markdown("""
Türkiye'deki 81 ilin yaşam kalitesi verilerine göre segmentasyonunu yapabilir,
belirli bir ili arayabilir, filtreleyebilir, sıralayabilir ve benzer şehirleri keşfedebilirsiniz.
""")

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
        1: "Gelişmekte Olan Şehirler",
        2: "Yüksek Yaşam Kalitesine Sahip Şehirler",
        3: "Gelişim Potansiyeli Olan Şehirler"
    }
    df["Küme Adı"] = df["Küme"].map(kume_isimleri)
    return df, X_scaled, X.columns

df, X_scaled, kategoriler = load_and_prepare_data()

# Yan panelde filtreleme ekliyorum
st.sidebar.header("🔎 Filtreleme ve Arama")
secilen_il = st.sidebar.selectbox("Bir il seçin:", sorted(df["İl Province"].unique()))
secilen_kume = st.sidebar.multiselect("Küme seçin:", df["Küme Adı"].unique(), default=df["Küme Adı"].unique())

# Filtrelenmiş iller listesi
st.subheader("📋 Filtrelenmiş Şehirler Listesi")
filtreli_df = df[df["Küme Adı"].isin(secilen_kume)]
st.dataframe(filtreli_df[["İl Province", "Küme Adı"] + list(kategoriler)].reset_index(drop=True), use_container_width=True, hide_index=True)

# PCA bileşenlerini hesaplama işlemi
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["İl"] = df["İl Province"]
df_pca["Küme"] = df["Küme"].values
df_pca["Küme Adı"] = df["Küme Adı"].values

st.subheader("PCA Dağılımı: Kümelere Göre Filtrelenebilir Grafik")

secili_kumeler = st.multiselect("Görselleştirmek istediğiniz kümeleri seçin",
                                 options=df_pca["Küme Adı"].unique().tolist(),
                                 default=df_pca["Küme Adı"].unique().tolist())

df_pca_filtered = df_pca[df_pca["Küme Adı"].isin(secili_kumeler)]

fig, ax = plt.subplots(figsize=(10, 6))
for c in sorted(df_pca_filtered["Küme"].unique()):
    subset = df_pca_filtered[df_pca_filtered["Küme"] == c]
    ax.scatter(subset["PC1"], subset["PC2"], label=df_pca[df_pca["Küme"] == c]["Küme Adı"].iloc[0])
    for i in range(len(subset)):
        ax.text(subset["PC1"].iloc[i] + 0.02, subset["PC2"].iloc[i] + 0.02, subset["İl"].iloc[i], fontsize=8)

ax.set_title("Seçilen Kümelere Göre PCA Dağılımı")
ax.set_xlabel("Yaşam Kalitesi Skoru")
ax.set_ylabel("Bölgesel Özellik Skoru")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Seçilen il için detaylı sıralamaları göstermek için
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

# Seçilen ile benzer şehirler
st.subheader(f"🤝 {secilen_il} ile Benzer İller")
il_index = df[df["İl Province"] == secilen_il].index[0]
benzerlikler = cosine_similarity([X_scaled[il_index]], X_scaled)[0]
df["Benzerlik"] = benzerlikler
benzer_df = df[df["İl Province"] != secilen_il].sort_values("Benzerlik", ascending=False).head(5)
st.dataframe(benzer_df[["İl Province", "Küme Adı", "Benzerlik"]].reset_index(drop=True), use_container_width=True, hide_index=True)

# Kategoriye göre tüm şehir sıralaması
st.subheader("📊 Kategoriye Göre İl Sıralaması")
kategori = st.selectbox("Bir kategori seçin:", kategoriler)
kategori_df = df[["İl Province", kategori]].sort_values(by=kategori, ascending=False).reset_index(drop=True)
kategori_df.index += 1
kategori_df = kategori_df.rename(columns={"İl Province": "İl", kategori: "Puan"})
kategori_df.insert(0, "Sıra", kategori_df.index)
st.dataframe(kategori_df.reset_index(drop=True), use_container_width=True, hide_index=True)
