import pandas as pd

# Dosyayı oku
df = pd.read_excel('veri/illerde-yasam-endeksi-tuik.xls', header=[0,1,2])

print(df.columns)
