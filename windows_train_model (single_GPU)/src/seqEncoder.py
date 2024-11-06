from transformers import BertTokenizerFast, CLIPProcessor, AutoProcessor
import json


def _get_token(tokenIn):
    token = tokenIn.lower()
    return token


class SeqEncoder:
    def __init__(self, _config, JSONFile, textTokenizer=None):
        self.MAX_ANSWERS = _config["MAX_ANSWERS"]
        self.LEN_QUESTION = _config["LEN_QUESTION"]
        self.encoder_type = "answer"
        self.tokenizerName = textTokenizer
        self.textModel = _config["textModelPath"]
        self.clipList = _config["clipList"]
        if self.tokenizerName in self.clipList:
            self.tokenizer = CLIPProcessor.from_pretrained(self.textModel)
        elif self.tokenizerName in ["siglip-512"]:
            self.tokenizer = AutoProcessor.from_pretrained(self.textModel)
        elif self.tokenizerName in ["bert_base_uncased"]:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.textModel)
        Q_words = {}

        with open(JSONFile) as json_data:
            self.data = json.load(json_data)["questions"]

        for i in range(len(self.data)):
            if self.data[i]["active"]:
                sentence = self.data[i]["question"]
                if sentence[-1] == "?" or sentence[-1] == ".":
                    sentence = sentence[:-1]
                tokens = sentence.split()
                for token in tokens:
                    token = _get_token(token)
                    if token not in Q_words:
                        Q_words[token] = 1
                    else:
                        Q_words[token] += 1

        self.question_list_words = []
        self.question_words = {}

        sorted_words = sorted(Q_words.items(), key=lambda kv: kv[1], reverse=True)
        if self.tokenizerName in ["skipthoughts", "2lstm", "lstm"]:
            self.question_words = {"<EOS>": 0}
            self.question_list_words = ["<EOS>"]
            for i, (word, _) in enumerate(sorted_words):
                self.question_words[word] = i
                self.question_list_words.append(word)
        elif self.tokenizerName in ["siglip-512"]:
            for i, (word, _) in enumerate(sorted_words):
                self.question_words[word] = self.tokenizer(
                    text=word, return_tensors="np"
                )["input_ids"][0][0]
                self.question_list_words.append(word)
        else:  # clip
            for i, (word, _) in enumerate(sorted_words):
                self.question_words[word] = self.tokenizer(text=word)["input_ids"][1]
                self.question_list_words.append(word)

    def encode(self, sentence, question=True):
        if sentence[-1] == "?" or sentence[-1] == ".":
            sentence = sentence[:-1]
        res = ''
        if self.tokenizerName in self.clipList or self.tokenizerName in [
            "bert_base_uncased"
        ]:
            if question:
                res = self.tokenizer(
                    text=sentence, padding="max_length", max_length=self.LEN_QUESTION
                )
                return res

        elif self.tokenizerName in ["siglip-512"]:
            if question:
                res = self.tokenizer(
                    text=sentence,
                    padding="max_length",
                    max_length=self.LEN_QUESTION,
                    return_tensors="np",
                )
                return res

        elif self.tokenizerName in ["skipthoughts", "2lstm", "lstm"]:
            res = []
            if sentence[-1] == "?" or sentence[-1] == ".":
                sentence = sentence[:-1]

            if question:
                tokens = sentence.split()
                res.append(self.question_words["<EOS>"])
                while len(res) < self.LEN_QUESTION:
                    res.append(self.question_words["<EOS>"])
                res = res[: self.LEN_QUESTION]
        else:
            res = "unexpected wrong"
        return res

    def getVocab(self, question=True):
        if question:
            return self.question_list_words
