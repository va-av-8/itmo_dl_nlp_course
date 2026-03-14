# ITMO DL NLP Course

Домашние задания по курсу Deep Learning + NLP.

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

### Структура

```
├── hw1_text_classification.ipynb  # Ноутбук с решением
├── pyproject.toml                 # Зависимости проекта
└── uv.lock                        # Локфайл зависимостей
```

### Запуск

```bash
# Установка зависимостей
uv sync

# Скачивание данных
wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2

# Запуск Jupyter
uv run jupyter lab
```
