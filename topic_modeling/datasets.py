import string
import pymystem3
import pickle
import time
from nltk.tokenize import sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer
import razdel
import json
from nltk.util import ngrams
from collections import Counter
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import tqdm
tqdm.tqdm.pandas()

pd.options.display.max_colwidth = 1000
from headline_generation import add_oracle_summary_to_records


class LentaDataset(object):
    def __init__(self, input_path, transformed_path, n_rows=None):
        self.df = None
        self.n_rows = n_rows
        self.transformed_df = None
        self.input_path = input_path
        self.transformed_path = transformed_path

    def read_data(self):
        if self.n_rows is not None:
            df = pd.read_csv(self.input_path, nrows=self.n_rows)
        else:
            df = pd.read_csv(self.input_path)
        df = df[~df['text'].isnull()]
        df = df[~df['date'].isnull()]
        df = df[~df['topic'].isnull()]
        self.df = df

    def process_save(self, preprocessor, df_to_process, i):
        df_lenta_out = preprocessor.process_file_data(df_to_process)
        # modalities
        df_lenta_out['date'] = pd.to_datetime(df_lenta_out['date'])
        df_lenta_out['month'] = df_lenta_out['date'].dt.month.astype(str)
        # df_lenta_out['ym'] = df_lenta_out['date'].dt.year.astype(str) + "_" + df_lenta_out['month']
        # print(df_lenta_out['ym'].value_counts().sort_values())
        df_lenta_out = df_lenta_out[df_lenta_out['date'] >= '01-01-1999']
        min_date = df_lenta_out['date'].min()
        df_lenta_out['time'] = (df_lenta_out['date'].dt.year - min_date.year) * 12 + \
                               df_lenta_out['date'].dt.month - min_date.month
        f = open(self.transformed_path.replace(".txt", "_" + str(i) + ".txt"), 'a')
        # vowpal wabbit format: doc1 Alpha Bravo:10 Charlie:5 |author Ola_Nordmann
        for doc_idx, row in df_lenta_out.iterrows():
            if len(row['words']) > 0:
                tokens_counts = " ".join(
                    ["{}:{}".format(word, freq) for word, freq in Counter(row['words']).items() if freq > 1])
                targets = " |month {} |time {} |topic {}".format(row['month'], row['time'], row['topic'])
                out_str = "doc{} ".format(doc_idx) + tokens_counts + targets
                f.write(out_str + '\n')
        f.close()

    def transform(self):
        preprocessor = TextPreprocessor()
        n_parts = 20
        n_rows_in_batch = self.df.shape[0] // (n_parts - 1)
        for i in range(n_parts - 1):
            df_to_process = self.df[(n_rows_in_batch*i):(n_rows_in_batch*(i+1))]
            self.process_save(preprocessor, df_to_process, i)
        self.process_save(preprocessor, self.df[n_rows_in_batch*(i+1):], i + 1)

    def save_preprocessed_data_for_artm(self):
        self.read_data()
        self.transform()

    def split(self, input_path_mask, train_path, valid_path):
        f_train = open(train_path, 'a')
        f_valid = open(valid_path, 'a')
        n_parts = 20
        for j in range(n_parts):
            with open(input_path_mask.replace(".txt", "_" + str(j) + ".txt"), 'r') as in_data:
                for i, line in enumerate(in_data.readlines()):
                    if (i + 1) % 100 == 0:
                        f_valid.write(line)
                    else:
                        f_train.write(line)
        f_train.close()
        f_valid.close()


class RiaDataset(object):
    def __init__(self, input_path, transformed_path):
        self.df = None
        self.transformed_df = None
        self.input_path = input_path
        self.transformed_path = transformed_path
        self.records = None

    @staticmethod
    def process_save(data_to_process, i):
        result_data = []
        for dict_data in tqdm.tqdm(data_to_process):
            if 'text' in dict_data.keys() and 'title' in dict_data.keys():
                if len(dict_data['text']) > 100 and len(dict_data['title']) > 15:
                    # clean html around text
                    input_text = (
                        dict_data['text']
                            .replace("<b>", "")
                            .replace("</b>", "")
                            .replace("\n", "")
                            .replace("<br />", "")
                            .replace("&quot;", "")
                    )
                    soup = BeautifulSoup(input_text, features="html.parser")
                    # replace \n, capitalize
                    paragraphs = soup.find_all('p')
                    if paragraphs == list():
                        sent_tokenizer = PunktSentenceTokenizer(input_text)
                        all_sents = [x.capitalize().replace("\xa0", " ") for x in sent_tokenizer.tokenize(input_text)]
                        if all_sents == list():
                            all_sents = [x.capitalize().replace("\xa0", " ") for x in sent_tokenize(input_text)]
                    else:
                        all_sents = []
                        for paragraph in paragraphs:
                            text = paragraph.get_text()
                            try:
                                sent_tokenizer = PunktSentenceTokenizer(text)
                                sents = [x.capitalize().replace("\xa0", " ") for x in sent_tokenizer.tokenize(text)]
                                if sents == list():
                                    sents = [x.capitalize().replace("\xa0", " ") for x in sent_tokenize(text)]
                                paragraph.string = "\n".join(sents)
                                all_sents.extend(sents)
                            except:
                                print(text)
                    try:
                        del all_sents[0]
                        dict_data.update({
                            "text": " ".join(all_sents),
                            "sentences": all_sents,
                            "title": dict_data['title'].capitalize()
                        })
                        if len(all_sents) > 0:
                            result_data.append(dict_data)
                    except:
                        print("del all_sents[0] warning")
        new_records = add_oracle_summary_to_records(result_data, max_sentences=30, lower=True, nrows=10000000)
        train_file_name = "data/ria/ria_train_{}.json".format(i)
        test_file_name = "data/ria/ria_test_{}.json".format(i)
        with open(train_file_name, "w") as w_train:
            with open(test_file_name, "w") as w_test:
                for i, record in enumerate(new_records):
                    record["oracle_sentences"] = list(record["oracle_sentences"])
                    if i % 50 == 0:
                        w_test.write(json.dumps(record, ensure_ascii=False).strip() + "\n")
                    else:
                        w_train.write(json.dumps(record, ensure_ascii=False).strip() + "\n")

    def save_preprocessed_data_for_headline_generation(self):
        all_data = []
        print("read data...")
        with open(self.input_path, 'r') as f:
            for num_doc, line in tqdm.tqdm(enumerate(f.readlines())):
                dict_data = json.loads(line)
                all_data.append(dict_data)
        n_parts = 10
        n_rows_in_batch = len(all_data) // (n_parts - 1)
        for i in tqdm.tqdm(range(n_parts - 1)):
            data_to_process = all_data[(n_rows_in_batch * i):(n_rows_in_batch * (i + 1))]
            self.process_save(data_to_process, i)
        self.process_save(data_to_process[n_rows_in_batch * (i + 1):], i + 1)


class TextPreprocessor(object):
    def __init__(self):
        # lemmatizer
        self.lemmatizer = pymystem3.Mystem()
        # nltk stopwords
        self.stopwords = stopwords.words('russian')
        self.stopwords.extend(['', ' ', '\n', '«', '»'])
        self.stopwords.extend([p for p in string.punctuation])

    def process_file_data(self, data):
        data_out = data.copy(deep=True)
        data_out['words'] = data_out['text'].progress_apply(self.preprocess_text)
        return data_out

    def preprocess_text(self, text):
        try:
            # split on sentences
            sents = [sent for (start_pos, end_pos, sent) in razdel.sentenize(text)]
            # split on words, transform to lower case and lemmatize
            words = [word.lower().strip() for sent in sents for word in self.lemmatizer.lemmatize(sent)]
            # remove stop words
            words = [word for word in words if word not in self.stopwords and ":" not in word]
        except:
            words = []
        return words

    @staticmethod
    def collect_bigrams(text_words):
        # for a single text
        bigrams = ngrams(text_words, 2)
        bigrams = ["_".join([first_word, sec_word]) for (first_word, sec_word) in bigrams]
        top_bigrams = Counter(bigrams).most_common(500)
        print(top_bigrams)


if __name__ == '__main__':
    # PREPARE LENTA DATA FOR TOPIC MODELING
    lenta_data = LentaDataset(
         input_path='data/lenta-ru-news.csv',
         transformed_path='data/lenta_transformed_tm.txt',
         n_rows=None
     )
    lenta_data.save_preprocessed_data_for_artm()
    lenta_data.split(
         input_path_mask='data/lenta_transformed_tm.txt',
         train_path='data/lenta_transformed_tm_train.txt',
         valid_path='data/lenta_transformed_tm_valid.txt'
    )

    # PREPARE RIA DATA FOR HEADLINE GENERATION
    ria_data = RiaDataset(
        input_path='data/ria.json',
        transformed_path='data/ria_preprocessed.txt'
    )
    st_time = time.time()
    ria_data.save_preprocessed_data_for_headline_generation()
    print(time.time() - st_time)

    # apply baseline
    # predictions = ria_data.transformed_df['sents'].apply(baseline_headline_generation).values.tolist()
    # print(calc_scores(ria_data.transformed_df['title'].values.tolist(), predictions))

    # 20,000 random articles to form the test set
