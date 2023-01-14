import numpy as np


class AutoExample:
    def __init__(self, tokenizer, tokenizable_fields, max_len, **kwargs):
        self.tokenizer = tokenizer
        self.tokenizable_fields = tokenizable_fields
        self.max_len = max_len
        self.skip = False
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def preprocess(self):
        tokenized_fields = {field_name: self.tokenizer.encode(" ".join(str(getattr(self, field_name)).split()))
                            for field_name in self.tokenizable_fields}
        for field_name, tokenized_field in tokenized_fields.items():
            input_ids = tokenized_field
            token_type_ids = [1] * len(tokenized_field)
            attention_mask = [1] * len(input_ids)
            padding_length = self.max_len - len(input_ids)
            if padding_length > 0:
                input_ids = input_ids + ([0] * padding_length)
                attention_mask = attention_mask + ([0] * padding_length)
                token_type_ids = token_type_ids + ([0] * padding_length)
            elif padding_length < 0:
                self.skip = True
                return
            setattr(self, f"{field_name}_input_ids", input_ids)
            setattr(self, f"{field_name}_token_type_ids", token_type_ids)
            setattr(self, f"{field_name}_attention_mask", attention_mask)
        for field_name in self.tokenizable_fields:
            delattr(self, field_name)

    def get_fields(self):
        fields = list(vars(self))
        skip_fields = ["tokenizer", "tokenizable_fields", "max_len"]
        return [field for field in fields if field not in skip_fields]


class PreprocessDataFrame:
    def __init__(self, df, tokenizable_fields, max_len, tokenizer):
        self.df = df
        self.tokenizable_fields = tokenizable_fields
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.examples = []

    def generate_examples(self):
        df = self.df
        keys = df.columns.tolist()
        for i in df.index:
            kwargs = {}
            for key in keys:
                kwargs[key] = df[key][i]
            eg = AutoExample(self.tokenizer, self.tokenizable_fields, self.max_len, **kwargs)
            eg.preprocess()
            self.examples.append(eg)

    def generate_dataset_dict(self):
        fields = self.examples[0].get_fields()
        dataset_dict = {field: [] for field in fields}
        for item in self.examples:
            for field in fields:
                dataset_dict[field].append(getattr(item, field))
        for field in fields:
            dataset_dict[field] = np.array(dataset_dict[field])
        return dataset_dict
