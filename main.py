import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 1. Veri setini yükleme işlemi
df = pd.read_excel("data/life-quality-index-tuik.xlsx", engine="openpyxl")

# 2. Gereksiz sütunları kaldırıyoruz
df = df.loc[:, ~df.columns.str.contains("Unnamed")]

# 3. Sütun isimlerini düzenliyoruz
df.columns = df.columns.str.replace("\n", " ", regex=True).str.strip()

# 4. Eksik değerleri kontrol edip, ortalama ile dolduruyoruz
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.mean(numeric_only=True))

# 5. İl bilgisini ayırıp veriyi ölçekliyoruz
iller = df["İl Province"]
X = df.drop(columns=["İl Province"])
X_scaled = MinMaxScaler().fit_transform(X)

# 6. KMeans ile kümeleme işlemi (k=4)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
kmeans_labels = kmeans.fit_predict(X_scaled)

# 7. DBSCAN ve Agglomerative ile alternatif kümelemeler
dbscan_labels = DBSCAN(eps=0.5, min_samples=3).fit_predict(X_scaled)
agglom_labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_scaled)

# 8. Küme adlandırmaları
kume_isimleri = {
    0: "Gelişmiş Şehirler",
    1: "Gelişmekte Olan Bölgeler",
    2: "Yüksek Yaşam Kalitesine Sahip İller",
    3: "Gelişim Potansiyeli Olan İller"
}

# 9. Etiketleri ana veri çerçevesine ekliyoruz
df["KMeans_Küme"] = kmeans_labels
df["Küme Adı"] = df["KMeans_Küme"].map(kume_isimleri)
df["DBSCAN_Küme"] = dbscan_labels
df["Agg_Küme"] = agglom_labels

# 10. Küme özet istatistikleri
kume_ozet = df.groupby("Küme Adı").mean(numeric_only=True)

# 11. PCA ile boyut indirgeme ve görselleştirme
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["İl"] = iller
df_pca["Küme"] = kmeans_labels

plt.figure(figsize=(10, 6))
for c in sorted(df_pca["Küme"].unique()):
    subset = df_pca[df_pca["Küme"] == c]
    plt.scatter(subset["PC1"], subset["PC2"], label=f"Küme {c}")
for i in range(len(df_pca)):
    plt.text(df_pca["PC1"][i]+0.02, df_pca["PC2"][i]+0.02, df_pca["İl"][i], fontsize=7)
plt.title("Şehirlerin Küme Dağılımı (PCA)")
plt.xlabel("Bileşen 1")
plt.ylabel("Bileşen 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 12. Random Forest ile öznitelik önem dereceleri
rf = RandomForestClassifier(random_state=42)
rf.fit(X_scaled, kmeans_labels)
feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# 13. Korelasyon matrisi
plt.figure(figsize=(10, 8))
sns.heatmap(pd.DataFrame(X).corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Kategoriler Arası Korelasyon Matrisi")
plt.tight_layout()
plt.show()

# 14. İl bazlı kategori sıraları
kategori_siralari = {}
for kolon in X.columns:
    sirali_df = df[["İl Province", kolon]].sort_values(by=kolon, ascending=False).reset_index(drop=True)
    sirali_df.index += 1
    sirali_df["Sıra"] = sirali_df.index
    kategori_siralari[kolon] = sirali_df

def il_sirali_ozet(il_adi):
    sonuc = []
    for kolon in X.columns:
        sira_df = kategori_siralari[kolon]
        satir = sira_df[sira_df["İl Province"] == il_adi]
        if not satir.empty:
            puan = satir[kolon].values[0]
            sira = satir["Sıra"].values[0]
            sonuc.append((kolon, puan, sira))
    return pd.DataFrame(sonuc, columns=["Kategori", "Puan", "Sıra"])

# 15. Benzer il öneri sistemi oluşturmak için kosinüs benzerliğinden yararlanacağım
X_df = pd.DataFrame(X_scaled, index=iller)
benzerlik_matrisi = pd.DataFrame(cosine_similarity(X_df), index=iller, columns=iller)

def benzer_iller(il_adi, n=5):
    if il_adi in benzerlik_matrisi:
        return benzerlik_matrisi[il_adi].sort_values(ascending=False)[1:n+1]
    else:
        return pd.Series(dtype=float)
