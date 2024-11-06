import json


class answerNumber:
    def __init__(self, _config, JSONFile):
        self.config = _config
        with open(JSONFile) as json_data:
            self.answers = json.load(json_data)['answers']


    def encode(self, qType, answer):
        output = self.answers[int(qType)-1]['answer'][answer]
        return output

