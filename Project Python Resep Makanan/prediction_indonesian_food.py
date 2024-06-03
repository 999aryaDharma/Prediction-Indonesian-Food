import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

# Baca dataset resep masakan
df_ayam = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-ayam.csv")
df_ikan = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-ikan.csv")
df_kambing = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-kambing.csv")
df_sapi = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-sapi.csv")
df_tahu = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-tahu.csv")
df_telur = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-telur.csv")
df_tempe = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-tempe.csv")
df_udang = pd.read_csv("C:/Data Science/Project Python Resep Makanan/datasets/dataset-udang.csv")

# Gabungkan DataFrame secara vertikal
df_resep = pd.concat(
    [df_ayam, df_ikan, df_kambing, df_sapi, df_tahu, df_telur, df_tempe, df_udang],
    ignore_index=True,
)

# Hapus baris yang mengandung nilai NaN
df_resep.dropna(subset=["Ingredients"], inplace=True)

# Daftar stop words dalam bahasa Indonesia
stop_words_indonesian = [
    "dan",
    "atau",
    "di",
    "dari",
    "yang",
]  # Tambahkan stop words yang relevan

# Misalnya, gunakan TF-IDF untuk mengubah teks bahan-bahan menjadi representasi vektor
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_indonesian)
tfidf_matrix = tfidf_vectorizer.fit_transform(df_resep["Ingredients"].astype(str))


# Contoh fungsi untuk merekomendasikan resep berdasarkan bahan yang dimiliki pengguna
def recommend_recipes(user_ingredients, num_recommendations=5):
    # Ubah input pengguna menjadi representasi vektor menggunakan TF-IDF
    user_tfidf = tfidf_vectorizer.transform([user_ingredients])
    # Hitung similaritas kosinus antara input pengguna dan semua resep
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    # Ambil indeks resep yang memiliki similaritas tertinggi
    similar_indices = cosine_similarities.argsort().flatten()[-num_recommendations:][
        ::-1
    ]
    # Ambil resep-resep yang direkomendasikan
    recommended_recipes = df_resep.iloc[similar_indices]
    return recommended_recipes


# Contoh fungsi untuk mencetak rekomendasi resep dengan format yang baik
def print_recommendations(df):
    recommendations = []
    for index, row in df.iterrows():
        recommendation = {
            "Title": row["Title"],
            "Ingredients": row["Ingredients"].split("--"),
            "Steps": row["Steps"].split("--"),
            "URL": row["URL"],
        }
        recommendations.append(recommendation)
    return recommendations


if __name__ == "__main__":
    # Interaksi pengguna
    user_input = input("Masukkan bahan-bahan yang Anda miliki (pisahkan dengan koma): ")
    recommended_recipes = recommend_recipes(user_input)

    if recommended_recipes is not None:
        data = recommended_recipes[["Title", "Ingredients", "Steps", "URL"]]
        df = pd.DataFrame(data)
        print_recommendations(df)
    else:
        print("Maaf, tidak ada resep yang sesuai dengan bahan yang Anda miliki.")
