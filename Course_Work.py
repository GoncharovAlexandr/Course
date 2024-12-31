import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import streamlit as st
from googletrans import Translator


# Функция для загрузки и объединения данных из нескольких файлов
def load_and_combine_data(files):
    data_frames = []
    for file in files:
        df = pd.read_csv(file)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df


# Список файлов
files = ['business_data.csv','education_data.csv', 'entertainment_data.csv', 'sports_data.csv', 'technology_data.csv']

# Загрузка и объединение данных
df = load_and_combine_data(files)


# Очистка текста
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Удаление ссылок
    text = re.sub(r'\W', ' ', text)  # Удаление всего, кроме букв и цифр
    text = text.lower()  # Приведение к нижнему регистру
    text = text.strip()  # Удаление пробелов в начале и в конце
    return text


# Функция перевода текста с русского на английский с обработкой ошибок
def translate_to_english(text):
    try:
        if not text:
            return ""  # Возвращаем пустую строку, если текст пустой

        translator = Translator()
        translated = translator.translate(text, src='ru', dest='en')

        # Проверка наличия текста в ответе
        if translated and translated.text:
            return translated.text
        else:
            return text  # Если перевод не удался, возвращаем исходный текст
    except Exception as e:
        st.error(f"Ошибка при переводе: {e}")
        return text  # Возвращаем исходный текст в случае ошибки


# Применение очистки и перевода к колонкам данных
df['cleaned_content'] = df['content'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')
df['cleaned_headlines'] = df['headlines'].apply(lambda x: clean_text(x) if isinstance(x, str) else '')

# Кодирование категорий
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['category'])

# Разделение на обучающую и тестовую выборки
X = df['cleaned_content']
y = df['category_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Преобразование текста в векторы
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Обучение модели
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Оценка модели
y_pred = model.predict(X_test_vect)
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Streamlit интерфейс
st.title("Классификатор новостных статей")

# Ввод текста новости
news_input = st.text_area("Введите текст новости:")

if st.button('Классифицировать'):
    # Проверяем, если введенный текст на русском, переводим
    if news_input:
        translated_input = translate_to_english(news_input)
        cleaned_input = clean_text(translated_input)

        # Преобразование текста в вектор
        news_vect = vectorizer.transform([cleaned_input])

        # Предсказание категории
        prediction = model.predict(news_vect)

        # Отображение результата
        predicted_category = label_encoder.inverse_transform(prediction)
        st.write(f"Категория новости: {predicted_category[0]}")
    else:
        st.error("Пожалуйста, введите текст новости.")
