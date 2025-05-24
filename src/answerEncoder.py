import json


class answerNumber:
    def __init__(self, _config, JSONFile):
        self.config = _config
        self.reverse = []
        temp_dict = {}
        with open(JSONFile) as json_data:
            self.answers = json.load(json_data)['answers']
        for answer in self.answers:
            temp_dict['type'] = answer['type']
            temp_dict['answer'] = {value: key for key, value in answer['answer'].items()}
            self.reverse.append(temp_dict.copy())

    def encode(self, qType, answer):
        output = self.answers[int(qType)-1]['answer'][answer]
        return output

    def decode(self, qType, answer):
        output = self.reverse[int(qType)-1]['answer'][answer]
        return output
