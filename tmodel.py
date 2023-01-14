import nltk
import matplotlib.colors as mcolors
from wordcloud import WordCloud, STOPWORDS
import pickle
from razdel import sentenize
from matplotlib import pyplot as plt
from gensim.utils import simple_preprocess
from pprint import pprint
import warnings
from base64 import encode
# import numpy as np
# import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
import pyLDAvis  # pip install pyLDAvis==2.1.2
import pyLDAvis.gensim
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    perplexity_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)
        model_list.append(model)
        perplexity_values.append(model.log_perplexity(corpus))
        coherencemodel = CoherenceModel(
            model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values, perplexity_values


def prepare_data(text):
    # Разбить на предложения
    from razdel import sentenize
    text = list(sentenize(text))
    text_mas = []
    for item in text:
        text_mas.append(item.text)

    # print(text_mas)

    def sent_to_words(sentences):
        for sentence in sentences:
            # deacc=True removes punctuations
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))

    data_words = list(sent_to_words(text_mas))
    # print(data_words[:1])
    # Создание биграмм и триграмм
    # Build the bigram and trigram models
    # higher threshold fewer phrases.
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
    # Faster way to get a sentence clubbed as a trigram/bigram
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # # See trigram example
    # print(trigram_mod[bigram_mod[data_words[1]]])

    def make_bigrams(texts):
        return [bigram_mod[doc] for doc in texts]

    def make_trigrams(texts):
        return [trigram_mod[bigram_mod[doc]] for doc in texts]

    # # # Подготовим стоп-слова
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = stopwords.words('russian')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def remove_stopwords(texts):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    data_words_nostops = remove_stopwords(data_words)
    # print(data_words_nostops)

    data_words_bigrams = make_bigrams(data_words_nostops)

    # print(data_words_bigrams )

    # Лемматизация
    def lemmatization(texts):
        texts_out = []
        import pymorphy2
        morph = pymorphy2.MorphAnalyzer()
        allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']

        for item in texts:
            ltexts = []
            for word in item:
                # print(word)
                if len(str(word)) > 2:
                    tag = str(morph.parse(word)[0].tag).split(',')[0]
                    if tag in allowed_postags:
                        ltexts.append(morph.parse(word)[0].normal_form)
            texts_out.append(ltexts)
        return texts_out

    data_lemmatized = lemmatization(data_words_bigrams, )
    # print(data_lemmatized[:1])

    # # # Создадим словарь и корпус.
    # Create Dictionary
    id2word = corpora.Dictionary(data_lemmatized)
    # Create Corpus
    texts = data_lemmatized
    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]
    # # View
    # print(corpus[:1])
    #
    # # Human readable format of corpus (term-frequency)
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])
    return texts, id2word, corpus, data_lemmatized


def tmodel(text, topic_num):
    # загрузка данных из файла

    if len(text) < 1:
        with open('text.txt', encoding="utf-8") as fp:
            text = fp.read()
    # print(text)
    texts, id2word, corpus, data_lemmatized = prepare_data(text)

    # Построим тематическую модель для графиков
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=topic_num,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    # # Visualize the topics
    # # pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, "static/vis.html")
    # pyLDAvis.save_json(vis, "static/vis.json")
    # ls=pyLDAvis.utils.NumPyEncoder(skipkeys=False, ensure_ascii=False, check_circular=True, allow_nan=True, sort_keys=False, indent=None, separators=None,  default=None) #encoding='UTF-8',
    # print(ls)
    # import webbrowser
    # webbrowser.open_new("vis.html")

    # Рисуем слова
    # 1. Wordcloud of Top N words in each topic
    def drawWords():

        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = stopwords.words('russian')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        # more colors: 'mcolors.XKCD_COLORS'
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
        cloud = WordCloud(stopwords=stop_words,
                          background_color='white',
                          width=2500,
                          height=1800,
                          max_words=10,
                          colormap='tab10',
                          color_func=lambda *args, **kwargs: cols[i],
                          prefer_horizontal=1.0)
        topics = lda_model.show_topics(formatted=False)
        fig, axes = plt.subplots(2, 2, figsize=(
            10, 10), sharex=True, sharey=True)
        for i, ax in enumerate(axes.flatten()):
            fig.add_subplot(ax)
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words, max_font_size=300)
            plt.gca().imshow(cloud)
            plt.gca().set_title('Тема ' + str(i+1), fontdict=dict(size=16))
            plt.gca().axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.axis('off')
        plt.margins(x=0, y=0)
        plt.tight_layout()
        # plt.show()
        plt.savefig('./static/wc.png')
    drawWords()


if __name__ == '__main__':
    tmodel(text="", topic_num=5)
