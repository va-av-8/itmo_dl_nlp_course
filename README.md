# ITMO DL NLP Course

Домашние задания по курсу Deep Learning + NLP.

---

## HW1: Классификация текстов по топикам

**Задача:** Многоклассовая классификация новостных статей Lenta.ru по тематическим категориям.

**Датасет:** [lenta-ru-news](https://github.com/yutkin/Lenta.Ru-News-Dataset) (~800k статей, 14 категорий после фильтрации)

### Пайплайн

1. **Предобработка:** лемматизация (pymorphy3), удаление стоп-слов, URL, чисел
2. **Векторизация:** TF-IDF с биграммами
3. **Модель:** Logistic Regression
4. **Подбор гиперпараметров:** GridSearchCV с 3-fold стратифицированной кросс-валидацией

### Результаты

| Модель | Accuracy | F1 Macro | F1 Weighted |
|--------|----------|----------|-------------|
| DummyClassifier (most_frequent) | 0.208 | 0.025 | 0.072 |
| CountVectorizer + LR | 0.785 | 0.743 | 0.785 |
| TfidfVectorizer + LR | 0.816 | 0.743 | 0.811 |
| **Best Model (tuned)** | **0.826** | **0.793** | **0.827** |

### Лучшие гиперпараметры

```python
{
    'vectorizer__max_features': 20000,
    'vectorizer__ngram_range': (1, 2),
    'vectorizer__min_df': 5,
    'vectorizer__sublinear_tf': True,
    'classifier__C': 10.0,
    'classifier__class_weight': 'balanced'
}
```

---

## HW2: Word2Vec для классификации текстов

**Задача:** Многоклассовая классификация новостных статей Lenta.ru по тематическим категориям с использованием word embeddings.

**Датасет:** [lenta-ru-news](https://github.com/yutkin/Lenta.Ru-News-Dataset) (~663k статей для обучения Word2Vec, 100k выборка для классификации, 14 категорий)

### Пайплайн

1. **Предобработка:** лемматизация (pymorphy3), удаление стоп-слов, URL, чисел (из HW1)
2. **Обучение Word2Vec:** Skip-gram, 300d, window=10, min_count=5, 10 эпох на полном датасете
3. **Предобученные эмбеддинги:** Navec (худлит, 12B токенов), RusVectores (Taiga, 4.8B токенов)
4. **Векторизация:** усреднение эмбеддингов слов текста
5. **Модель:** Logistic Regression с class_weight='balanced'
6. **TF-IDF взвешивание:** взвешенное усреднение эмбеддингов по TF-IDF коэффициентам

### Результаты на тестовой выборке

| Модель | Accuracy | F1 Macro | F1 Weighted |
|--------|----------|----------|-------------|
| **Word2Vec (наш)** | **0.766** | **0.713** | **0.778** |
| Word2Vec + TF-IDF | 0.734 | 0.678 | 0.747 |
| RusVectores (Taiga) | 0.726 | 0.670 | 0.740 |
| Navec (худлит) | 0.719 | 0.664 | 0.733 |

### Сравнение с HW1 (TF-IDF baseline)

| Метрика | HW1 (TF-IDF) | HW2 (Word2Vec) | Разница |
|---------|--------------|----------------|---------|
| Accuracy | 0.826 | 0.766 | -0.060 |
| F1 Macro | 0.793 | 0.713 | -0.080 |

### Основные выводы

1. **Собственный Word2Vec превосходит предобученные модели** (+4-5% F1 Macro) — domain adaptation важен
2. **TF-IDF взвешивание ухудшило результаты** (-3.5% F1 Macro) из-за конфликта целей: TF-IDF усиливает редкие слова, у которых менее качественные эмбеддинги
3. **TF-IDF из HW1 превосходит Word2Vec** — для тематической классификации важно *какие* слова встречаются, а не их семантика; усреднение в 300d вектор теряет дискриминативную информацию

### Гиперпараметры Word2Vec

```python
{
    'vector_size': 300,
    'window': 10,
    'min_count': 5,
    'sg': 1,  # Skip-gram
    'epochs': 10,
    'negative': 5
}
```

---

## HW4: Тематическое моделирование с BERTopic

**Задача:** Автоматическое выделение тем из корпуса новостей без учителя с использованием BERTopic.

**Датасет:** [lenta-ru-news](https://github.com/yutkin/Lenta.Ru-News-Dataset) (30k статей — стратифицированная выборка, 14 категорий)

### Пайплайн

1. **Предобработка:** удаление URL, email, чисел; лемматизация (pymorphy3) для c-TF-IDF
2. **Эмбеддинги:** `paraphrase-multilingual-MiniLM-L12-v2` (384d)
3. **Снижение размерности:** UMAP (n_components=5, metric='cosine')
4. **Кластеризация:** HDBSCAN (min_cluster_size=150)
5. **Ключевые слова:** c-TF-IDF + MaximalMarginalRelevance (diversity=0.3)
6. **Постобработка:** reduce_outliers для переназначения выбросов

### Результаты

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| **Topic Diversity** | 0.89 | Высокое — 89% слов уникальны |
| **UMass Coherence** | -2.31 | Отличное — темы когерентны |
| **Adjusted Rand Index** | 0.25 | Умеренное совпадение с рубриками |
| **Normalized Mutual Info** | 0.40 | Умеренное совпадение с рубриками |
| **Найдено тем** | 22 | vs 14 рубрик Lenta.ru |
| **Outliers** | 0% | После reduce_outliers |

### Примеры найденных тем

| Тема | Ключевые слова | Соответствие Lenta.ru |
|------|----------------|----------------------|
| 1 | матч, чемпионат, сборная, игрок, футбол | Спорт (97%) |
| 2 | процент, доллар, миллиард, рынок, евро | Экономика (87%) |
| 4 | квадратный метр, недвижимость, жильё | Дом (67%) |
| 5 | учёный, животное, исследователь, птица | Наука и техника |
| 7 | самолёт, авиакомпания, пилот, рейс | — (новая тема) |

### Основные выводы

1. **Лемматизация критична** для русского языка — без неё словоформы (`ученые`, `ученых`) занимают несколько слотов в топ-словах
2. **BERTopic нашёл более детальную структуру** (22 темы vs 14 рубрик) — разделил широкие категории на подтемы
3. **Высокая когерентность** (UMass = -2.31) подтверждает качество тем
4. **Умеренный ARI/NMI** объясняется разным числом категорий и тем, что BERTopic группирует по семантике, а не редакционной политике

### Гиперпараметры

```python
# UMAP
{'n_neighbors': 15, 'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine'}

# HDBSCAN
{'min_cluster_size': 150, 'metric': 'euclidean', 'cluster_selection_method': 'eom'}

# CountVectorizer
{'tokenizer': lemmatize_tokenizer, 'ngram_range': (1, 2), 'min_df': 5}
```

---

## Структура проекта

```
├── hw1_text_classification.ipynb  # HW1: TF-IDF классификация
├── hw2_word2vec.ipynb             # HW2: Word2Vec классификация
├── hw4_topic_modeling.ipynb       # HW4: BERTopic тематическое моделирование
├── results/                       # Артефакты HW4
│   ├── embeddings_30k.npy         # Кэш эмбеддингов
│   ├── topic_barchart.html        # Визуализация топ-слов тем
│   ├── topic_documents_2d.html    # 2D карта документов
│   └── topic_heatmap.html         # Тепловая карта сходства тем
├── pyproject.toml                 # Зависимости проекта
└── uv.lock                        # Локфайл зависимостей
```

## Запуск

```bash
# Установка зависимостей
uv sync

# Скачивание данных
wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2

# Скачивание предобученных эмбеддингов (для HW2)
# Navec
wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar

# RusVectores (Taiga)
wget http://vectors.nlpl.eu/repository/20/185.zip
unzip 185.zip

# Запуск Jupyter
uv run jupyter lab
```
