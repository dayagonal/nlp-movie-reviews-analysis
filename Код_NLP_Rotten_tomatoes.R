library(tidyverse)
library(dplyr)
library(data.table)
library(quanteda)
library(quanteda.textmodels)
library(tidytext)
library(textstem) 
library(textdata)
library(tokenizers)
library(topicmodels)
library(ggplot2)
library(wordcloud)
library(glmnet)
library(caret)
library(stopwords)
library(patchwork)
library(quanteda.textstats)
library(SentimentAnalysis)
library(tm)  
library(caret)
# Загрузка данных
reviews <- fread("rotten_tomatoes_critic_reviews.csv")
movies <- fread("rotten_tomatoes_movies.csv")

# Создание уникального ID для отзывов
reviews <- reviews %>%
  mutate(review_id = row_number())

# Объединение данных
combined_data <- reviews %>%
  inner_join(movies, by = "rotten_tomatoes_link") %>%
  select(review_id, movie_title, review_type, review_content, 
         genres, tomatometer_rating, audience_rating, top_critic) %>% 
  sample_n(10000) 

# Проверка структуры данных
glimpse(combined_data)

# Анализ распределения типов отзывов
review_type_distribution <- combined_data %>%
  dplyr::count(review_type) %>%
  mutate(percent = n / sum(n) * 100)

print(review_type_distribution)
# генерация стоп слов
generate_custom_stopwords <- function(data) {
  # Базовый список стоп-слов
  base_stopwords <- stopwords("en")
  
  # Кино-специфичные термины
  movie_terms <- c("film", "movie", "can", "may", "will", "just", "make", 
                   "even", "one", "see", "also", "still", "much", "way",
                   "get", "say", "character", "scene", "time",
                   "film", "movie", "cinema", "picture", "feature", 
                   "flick", "review", "director", "directed", "writer", 
                   "performance", "actor", "actress", "character", 
                   "audience", "scene", "screen", "story", "plot",
                   "like", "good", "find", "seem", "work", "know", "come", "man", 
                   "something", "end", "take", "give", "make", "look", "really", "see", "get",
                   "new", "never", "enough", "great", "last", "need", "keep", "right",
                   "become", "two", "many", "think", "little", "leave", "woman", "bring",
                   "though", "full", "year", "another", "try", "thing", "lead", "want",
                   "along", "big", "ever", "long", "perhaps", "first", "second",
                   "yet", "nothing", "day", "often", "whose", "without", "far", "kind",
                   "sometimes", "truly", "part", "start", "back", "present", "live",
                   "idea", "promise", "special",
                   "show", "play", "offer", "tell", "use", "together", "rather", "almost",
                   "every", "anything", "instead", "despite", "quite", "around", "provide",
                   "create", "turn", "run", "less", "easy", "hope",
                   "bad")
  
  # Предварительная очистка текста
  cleaned_text <- data %>%
    mutate(
      clean_content = str_to_lower(review_content),
      clean_content = str_remove_all(clean_content, "[[:punct:]]|\\d"),
      clean_content = str_remove_all(clean_content, "\\b\\w{1,2}\\b")
    )
  
  # Выявление высокочастотных слов
  word_stats <- cleaned_text %>%
    unnest_tokens(word, clean_content) %>%
    anti_join(stop_words, by = "word") %>%
    filter(nchar(word) > 2) %>%
    dplyr::count(word, sort = TRUE) %>%
    mutate(
      rank = row_number(),
      freq_ratio = n / lag(n)
    )
  
  # Отбор слов с высокой частотой и низкой информативностью
  high_freq_words <- word_stats %>%
    filter(rank <= 30 | (freq_ratio > 0.85 & rank <= 100)) %>%
    pull(word)
  
  # Формирование итогового списка
  custom_stops <- unique(c(base_stopwords, movie_terms, high_freq_words))
  
  # Очистка списка
  custom_stops %>%
    str_replace_all("[[:punct:]]", "") %>%
    str_trim() %>%
    unique()
}

# Генерация стоп-слов
custom_stopwords <- generate_custom_stopwords(combined_data)
cat("Кастомные стоп-слова (первые 50):\n")
head(custom_stopwords, 50) %>% paste(collapse = ", ") %>% cat()

# Улучшенная функция очистки текста
clean_text <- function(text) {
  text %>%
    str_to_lower() %>%
    str_remove_all("\\n|http\\S+") %>%       # URL и переносы строк
    str_remove_all("[[:punct:]]") %>%        # Пунктуация
    str_remove_all("\\d+") %>%               # Числа
    str_remove_all("\\b\\w{1,2}\\b") %>%     # Короткие слова
    str_squish()                             # Лишние пробелы
}

# Оптимизированная обработка данных
tokenized_data <- combined_data %>%
  # Базовая очистка текста
  mutate(clean_content = clean_text(review_content)) %>%
  # Токенизация и фильтрация
  unnest_tokens(word, clean_content) %>%
  filter(
    !word %in% custom_stopwords,
    nchar(word) > 2
  ) %>%
  # Лемматизация
  mutate(word = lemmatize_words(word)) %>%
  # Фильтрация редких слов
  add_count(word) %>%
  filter(n >= 10) %>%
  select(-n)
tokenized_data <- tokenized_data %>%
  filter(!tolower(word) %in% custom_stopwords)
tokenized_data <- tokenized_data %>%
  filter(word != "go")

processed_data <- tokenized_data %>%
  # Сборка обратно в текст
  group_by(review_id, movie_title, review_type, genres, 
           tomatometer_rating, audience_rating) %>%
  summarise(clean_content = paste(word, collapse = " "), .groups = "drop") %>%
  # Фильтрация коротких текстов
  filter(nchar(clean_content) > 50)
# Анализ униграмм
unigram_freq <- tokenized_data %>%
  dplyr::count(word, sort = TRUE) 

# Визуализация топ-20 слов
p1 <- unigram_freq %>%
  head(20) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(n, word)) +
  geom_col(fill = "steelblue", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = n), hjust = -0.1, size = 3.5, color = "darkblue") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Топ-20 наиболее частых слов", 
       subtitle = "После очистки и удаления стоп-слов",
       x = "Частота", y = "Слово") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.major.y = element_blank()) + 
  scale_y_discrete(expand = expansion(add = c(0, 1)))  
print(unigram_freq)
# Анализ биграмм
bigrams <- processed_data %>%
  unnest_tokens(bigram, clean_content, token = "ngrams", n = 2) %>%
  separate(bigram, into = c("word1", "word2"), sep = " ") %>%
  filter(
    !word1 %in% custom_stopwords,
    !word2 %in% custom_stopwords,
    nchar(word1) > 2,
    nchar(word2) > 2
  ) %>%
  unite("bigram", word1, word2, sep = " ") %>%
  dplyr::count(bigram, sort = TRUE) 

# Визуализация топ-15 биграмм
p2 <- bigrams %>%
  head(15) %>%
  mutate(bigram = reorder(bigram, n)) %>%
  ggplot(aes(n, bigram)) +
  geom_col(fill = "coral", alpha = 0.8, width = 0.7) +
  geom_text(aes(label = n), hjust = -0.1, size = 3.5, color = "darkred") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Топ-15 наиболее частых биграмм", 
       subtitle = "После очистки и удаления стоп-слов",
       x = "Частота", y = "Биграмма") +
  theme_minimal(base_size = 14) +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(size=12))

# Компоновка графиков униграмм и биграмм
combined_plot <- p1 / p2 +
  plot_annotation(title = "Частотный анализ слов и словосочетаний",
                  subtitle = "Униграммы и биграммы в отзывах кинокритиков",
                  theme = theme(plot.title = element_text(size = 16, face = "bold"),
                                plot.subtitle = element_text(size = 14),
                                axis.text.y = element_text( size=10)))

print(combined_plot)

# TF-IDF анализ
tf_idf <- tokenized_data %>%
  dplyr::count(review_id, word) %>%  # review_id как идентификатор документа
  bind_tf_idf(word, review_id, n)

# Визуализация TF-IDF
tf_idf_agg <- tf_idf %>%
  left_join(select(tokenized_data, review_id, review_type), by = "review_id") %>%
  group_by(review_type, word) %>%
  summarise(mean_tf_idf = mean(tf_idf), .groups = "drop") %>%
  group_by(review_type) %>%
  slice_max(mean_tf_idf, n = 10) %>%
  ungroup()

# Визуализация 
tf_idf_plot <- tf_idf_agg %>%
  mutate(word = reorder_within(word, mean_tf_idf, review_type)) %>%
  ggplot(aes(mean_tf_idf, word, fill = review_type)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  geom_text(aes(label = round(mean_tf_idf, 4)), 
            hjust = -0.1, size = 3, color = "black") +
  scale_x_continuous(
    expand = expansion(mult = c(0, 0.3)),  # Больше места справа
    labels = scales::number_format(accuracy = 0.0001)  # Формат чисел
  ) +
  facet_wrap(~review_type, scales = "free", ncol = 2) +
  scale_y_reordered() +
  labs(title = "Среднее значение TF-IDF по типу отзыва", 
       x = "TF-IDF (среднее значение)", y = NULL) +
  theme_bw() +
  theme(strip.text = element_text(face = "bold"))

print(tf_idf_plot)
library(tidyverse)
library(tidytext)
library(topicmodels)


extra_stopwords <- c("like", "good", "find", "seem", "work", "know", "come", "man", 
                     "something", "end", "take", "give", "make", "look", "really", "see", "get",
                     "new", "never", "enough", "great", "last", "need", "keep", "right",
                     "become", "two", "many", "think", "little", "leave", "woman", "bring",
                     "though", "full", "year", "another", "try", "thing", "lead", "want",
                     "along", "big", "ever", "long", "perhaps", "first", "second",
                     "yet", "nothing", "day", "often", "whose", "without", "far", "kind",
                     "sometimes", "truly", "part", "start", "back", "present", "live",
                     "idea", "promise", "special",
                     "show", "play", "offer", "tell", "use", "together", "rather", "almost",
                     "every", "anything", "instead", "despite", "quite", "around", "provide",
                     "create", "turn", "run", "less", "easy", "hope",
                     "bad")

all_stopwords <- unique(c(custom_stopwords, extra_stopwords, stopwords("en")))

tokens_df <- processed_data %>%
  unnest_tokens(word, clean_content) %>%
  filter(!word %in% all_stopwords,
         nchar(word) > 2) %>%
  group_by(word) %>%
  filter(n() > 5) %>%  
  ungroup()



# DTM 
dtm <- tokens_df %>%
  count(review_id, word) %>%
  cast_dtm(review_id, word, n)

# LDA 
set.seed(123)
num_topics <- 6 
lda_model <- LDA(dtm, k = num_topics, control = list(seed = 123))

# Топ-слова по темам
topics <- tidy(lda_model, matrix = "beta")

top_terms <- topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)

top_words_by_topic <- top_terms %>%
  group_by(topic) %>%
  summarise(
    terms = paste(term, collapse = ", "),
    .groups = "drop"
  )

print(top_words_by_topic)

# Визуализация 
lda_plot <- top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = FALSE, alpha = 0.8) +
  facet_wrap(~ topic, scales = "free_y", ncol = 2) +
  scale_y_reordered() +
  labs(
    title = paste("Топ-10 слов по темам (LDA с", num_topics, "темами)"),
    subtitle = "Тематическое моделирование отзывов кинокритиков",
    x = "Вероятность (beta)",
    y = NULL
  ) +
  theme_minimal() +
  theme(strip.text = element_text(face = "bold"))

print(lda_plot)

# Загружаем словари
afinn <- get_sentiments("afinn")
bing <- get_sentiments("bing") %>%
  mutate(value = ifelse(sentiment == "positive", 1, -1)) %>% 
  select(word, value)


# соединяем словари
combined_dict <- bind_rows(
  afinn  %>% select(word, value),
  bing
) %>%
  group_by(word) %>%
  summarise(sentiment_score = mean(value, na.rm = TRUE), .groups = "drop")

# Считаем тональность по отзывам
sentiment_scores <- tokenized_data %>%
  left_join(combined_dict, by = "word") %>%
  group_by(review_id, review_type, top_critic) %>%
  summarise(sentiment = mean(sentiment_score, na.rm = TRUE), .groups = "drop") %>%
  mutate(sentiment = ifelse(is.nan(sentiment), 0, sentiment))


# График распределения по отзывам
dist_p <- ggplot(sentiment_scores, aes(x = sentiment, fill = review_type)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("fresh" = "#1f77b4", "rotten" = "#d62728")) +
  labs(title = "Распределение тональности (2 словаря)",
       x = "Средняя тональность", y = "Плотность") +
  theme_minimal()
print(dist_p)
# Топ-слов по тональности
word_sentiment <- tokenized_data %>%
  left_join(combined_dict, by = "word") %>%
  filter(!is.na(sentiment_score)) %>%
  count(word, sentiment_score, sort = TRUE)

top_pos <- word_sentiment %>% filter(sentiment_score > 0) %>% slice_max(n, n = 10)
top_neg <- word_sentiment %>% filter(sentiment_score < 0) %>% slice_max(n, n = 10)

positive_p <- ggplot(top_pos, aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "#1f77b4") +
  coord_flip() +
  labs(title = "Топ позитивных слов", x = NULL, y = "Частота") +
  theme_minimal()

negative_p <- ggplot(top_neg, aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "#d62728") +
  coord_flip() +
  labs(title = "Топ негативных слов", x = NULL, y = "Частота") +
  theme_minimal()

print(positive_p)
print(negative_p)

summary(sentiment_scores$sentiment)
sentiment_scores %>%
  group_by(review_type) %>%
  summarise(
    mean_sentiment = mean(sentiment),
    median_sentiment = median(sentiment),
    sd_sentiment = sd(sentiment),
    .groups = "drop"
  )
dist_critic <- ggplot(sentiment_scores, aes(x = sentiment, fill = as.factor(top_critic))) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("TRUE" = "#1f77b4", "FALSE" = "#d62728"),
                    name = "Top Critic") +
  labs(title = "Распределение тональности по статусу критика",
       x = "Средняя тональность", y = "Плотность") +
  theme_minimal()
print(dist_critic)
# 3. Вычисление средней тональности по группам
sentiment_by_group <- sentiment_scores %>%
  group_by(top_critic) %>%
  summarise(
    mean_sentiment = mean(sentiment, na.rm = TRUE),
    sd_sentiment = sd(sentiment, na.rm = TRUE),
    count = n()
  )
print(sentiment_by_group)

# 4. t-тест для проверки статистической значимости разницы
t_test_result <- t.test(sentiment ~ top_critic, data = sentiment_scores)
print(t_test_result)

#   Преобразуем текст в матрицу TF-IDF
revies_sample <- reviews %>% slice_sample(n=80000)
corpus <- VCorpus(VectorSource(revies_sample$review_content))
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, custom_stopwords)
corpus <- tm_map(corpus, stripWhitespace)


dtm <- DocumentTermMatrix(corpus, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.99) 

#  Приводим к data.frame
data_matrix <- as.data.frame(as.matrix(dtm))
data_matrix$rating <- as.factor(revies_sample$review_type)

trainIndex <- createDataPartition(data_matrix$rating, p = 0.8, list = FALSE)
trainData <- data_matrix[trainIndex, ]
testData  <- data_matrix[-trainIndex, ]


# Логистическая регрессия
model <- glm(rating ~ ., data = trainData, family = "binomial")

# Предсказания
pred_probs <- predict(model, newdata = testData, type = "response")
pred_labels <- ifelse(pred_probs > 0.5, "Fresh", "Rotten")

# Оценка точности
confusionMatrix(factor(pred_labels, levels = c("Rotten", "Fresh")),
                testData$rating)
