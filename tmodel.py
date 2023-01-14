from base64 import encode
import numpy as np
import pandas as pd
# Gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaModel
# Plotting tools
# %matplotlib inline
# Enable logging for gensim - optional
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
from pprint import pprint

from gensim.utils import simple_preprocess
from matplotlib import pyplot as plt

from razdel import sentenize

import pickle

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
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
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values, perplexity_values


def prepare_data(text):
    # Разбить на предложения
    from razdel import sentenize
    text = list(sentenize(text))
    # print(text)

    text_mas = []
    for item in text:
        text_mas.append(item.text)

    # print(text_mas)

    def sent_to_words(sentences):
        for sentence in sentences:
            yield (gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

    data_words = list(sent_to_words(text_mas))
    # print(data_words[:1])

    # Создание биграмм и триграмм
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
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
    import nltk;
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
    
    if len(text)<1:
        with open('text.txt', encoding="utf-8") as fp:
            text = fp.read()
    # print(text)
    texts, id2word, corpus, data_lemmatized = prepare_data(text)
    
    #Построим тематическую модель для графиков
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=topic_num,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
    # #pip install pyLDAvis==2.1.2
    import pyLDAvis
    import pyLDAvis.gensim

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
        from matplotlib import pyplot as plt
        from wordcloud import WordCloud, STOPWORDS
        import matplotlib.colors as mcolors
        import nltk;
        nltk.download('stopwords')
        from nltk.corpus import stopwords
        stop_words = stopwords.words('russian')
        stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
        cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'
        cloud = WordCloud(stopwords=stop_words,
                        background_color='white',
                        width=2500,
                        height=1800,
                        max_words=10,
                        colormap='tab10',
                        color_func=lambda *args, **kwargs: cols[i],
                        prefer_horizontal=1.0)
        topics = lda_model.show_topics(formatted=False)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
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






    # def drawDiagram1():
    #     import matplotlib.colors as mcolors
    #     from collections import Counter
    #     topics = lda_model.show_topics(formatted=False)
    #     data_flat = [w for w_list in data_lemmatized for w in w_list]
    #     counter = Counter(data_flat)
    #     out = []
    #     for i, topic in topics:
    #         for word, weight in topic:
    #             out.append([word, i, weight, counter[word]])
    #     df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])
    #     # Plot Word Count and Weights of Topic Keywords
    #     fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True, dpi=160)
    #     cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    #     for i, ax in enumerate(axes.flatten()):
    #         ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
    #             label='Word Count')
    #         ax_twin = ax.twinx()
    #         ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
    #                     label='Weights')
    #         ax.set_ylabel('Word Count', color=cols[i])
    #         # ax_twin.set_ylim(0, 0.050);
    #         # ax.set_ylim(0, 9000)
    #         ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    #         ax.tick_params(axis='y', left=False)
    #         ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=30, horizontalalignment='right')
    #         ax.legend(loc='upper left');
    #         ax_twin.legend(loc='upper right')
    #     fig.tight_layout(w_pad=2)
    #     fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)
    #     plt.show()
    # drawDiagram1()

    # # Sentence Coloring of N Sentences
    # def  drawDiagram2():
    #     import matplotlib.colors as mcolors
    #     from matplotlib.patches import Rectangle
    #     def sentences_chart(lda_model=lda_model, corpus=corpus, start=0, end=13):
    #         corp = corpus[start:end]
    #         mycolors = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    #         fig, axes = plt.subplots(end - start, 1, figsize=(20, (end - start) * 0.95), dpi=160)
    #         axes[0].axis('off')
    #         for i, ax in enumerate(axes):
    #             if i > 0:
    #                 corp_cur = corp[i - 1]
    #                 topic_percs, wordid_topics, wordid_phivalues = lda_model[corp_cur]
    #                 word_dominanttopic = [(lda_model.id2word[wd], topic[0]) for wd, topic in wordid_topics]
    #                 ax.text(0.01, 0.5, "Doc " + str(i - 1) + ": ", verticalalignment='center',
    #                         fontsize=16, color='black', transform=ax.transAxes, fontweight=700)
    #                 # Draw Rectange
    #                 topic_percs_sorted = sorted(topic_percs, key=lambda x: (x[1]), reverse=True)
    #                 ax.add_patch(Rectangle((0.0, 0.05), 0.99, 0.90, fill=None, alpha=1,
    #                                        color=mycolors[topic_percs_sorted[0][0]], linewidth=2))
    #                 word_pos = 0.06
    #                 for j, (word, topics) in enumerate(word_dominanttopic):
    #                     if j < 14:
    #                         ax.text(word_pos, 0.5, word,
    #                                 horizontalalignment='left',
    #                                 verticalalignment='center',
    #                                 fontsize=16, color=mycolors[topics],
    #                                 transform=ax.transAxes, fontweight=700)
    #                         word_pos += .009 * len(word)  # to move the word for the next iter
    #                         ax.axis('off')
    #                 ax.text(word_pos, 0.5, '. . .',
    #                         horizontalalignment='left',
    #                         verticalalignment='center',
    #                         fontsize=16, color='black',
    #                         transform=ax.transAxes)
    #         plt.subplots_adjust(wspace=0, hspace=0)
    #         plt.suptitle('Sentence Topic Coloring for Documents: ' + str(start) + ' to ' + str(end - 2), fontsize=22,
    #                      y=0.95, fontweight=700)
    #         plt.tight_layout()
    #         plt.show()
    #     sentences_chart()
    # # drawDiagram2()


    # # Sentence Coloring of N Sentences
    # def drawDiagram3():
    #     def topics_per_document(model, corpus, start=0, end=1):
    #         corpus_sel = corpus[start:end]
    #         dominant_topics = []
    #         topic_percentages = []
    #         for i, corp in enumerate(corpus_sel):
    #             topic_percs, wordid_topics, wordid_phivalues = model[corp]
    #             dominant_topic = sorted(topic_percs, key=lambda x: x[1], reverse=True)[0][0]
    #             dominant_topics.append((i, dominant_topic))
    #             topic_percentages.append(topic_percs)
    #         return (dominant_topics, topic_percentages)
    #     dominant_topics, topic_percentages = topics_per_document(model=lda_model, corpus=corpus, end=-1)
    #     # Distribution of Dominant Topics in Each Document
    #     df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    #     dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    #     df_dominant_topic_in_each_doc = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    #     # Total Topic Distribution by actual weight
    #     topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    #     df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()
    #     # Top 3 Keywords for each Topic
    #     topic_top3words = [(i, topic) for i, topics in lda_model.show_topics(formatted=False)
    #                        for j, (topic, wt) in enumerate(topics) if j < 3]
    #     df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    #     df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    #     df_top3words.reset_index(level=0, inplace=True)
    #     from matplotlib.ticker import FuncFormatter

    #     # Plot
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=120, sharey=True)
    #     # Topic Distribution by Dominant Topics
    #     ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic_in_each_doc, width=.5, color='firebrick')
    #     ax1.set_xticks(range(df_dominant_topic_in_each_doc.Dominant_Topic.unique().__len__()))
    #     tick_formatter = FuncFormatter(
    #         lambda x, pos: 'Topic ' + str(x) + '\n' + df_top3words.loc[df_top3words.topic_id == x, 'words'].values[0])
    #     ax1.xaxis.set_major_formatter(tick_formatter)
    #     ax1.set_title('Number of Documents by Dominant Topic', fontdict=dict(size=2))
    #     ax1.set_ylabel('Number of Documents')
    #     # ax1.set_ylim(0, 1000)
    #     # Topic Distribution by Topic Weights
    #     ax2.bar(x='index', height='count', data=df_topic_weightage_by_doc, width=.5, color='steelblue')
    #     ax2.set_xticks(range(df_topic_weightage_by_doc.index.unique().__len__()))
    #     ax2.xaxis.set_major_formatter(tick_formatter)
    #     ax2.set_title('Number of Documents by Topic Weightage', fontdict=dict(size=2))
    #     plt.show()
    # # drawDiagram3()


    # def drawTSNE():
    #     # Get topic weights and dominant topics ------------
    #     import matplotlib.colors as mcolors
    #     from sklearn.manifold import TSNE
    #     from bokeh.plotting import figure, output_file, show
    #     from bokeh.models import Label
    #     from bokeh.io import output_notebook

    #     # Get topic weights
    #     topic_weights = []
    #     for i, row_list in enumerate(lda_model[corpus]):
    #         topic_weights.append([w for i, w in row_list[0]])
    #     # Array of topic weights
    #     arr = pd.DataFrame(topic_weights).fillna(0).values
    #     # Keep the well separated points (optional)
    #     arr = arr[np.amax(arr, axis=1) > 0.35]
    #     # Dominant topic number in each doc
    #     topic_num = np.argmax(arr, axis=1)
    #     # tSNE Dimension Reduction
    #     tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    #     tsne_lda = tsne_model.fit_transform(arr)
    #     # Plot the Topic Clusters using Bokeh

    #     n_topics = 4
    #     mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    #     plot = figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics),
    #                   plot_width=900, plot_height=700)
    #     plot.scatter(x=tsne_lda[:, 0], y=tsne_lda[:, 1], color=mycolors[topic_num])
    #     show(plot)
    # # drawTSNE()
