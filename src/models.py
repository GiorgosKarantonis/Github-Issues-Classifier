class PretrainedModels:
    def __init__(self, meta):
        self.tokenizer = None
        self.model = None
        self.meta = meta

        self.get_pretrained_model()
    
    
    def get_model(self):    
        if self.meta['name'] == 'gpt2':
            from transformers import GPT2Tokenizer, TFGPT2Model
            
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.meta['model'])
            self.model = TFGPT2Model.from_pretrained(self.meta['model'])
            self.tokenizer.pad_token=self.tokenizer.eos_token
        elif self.meta['name'] == 'electra':
            from transformers import ElectraTokenizer, TFElectraModel
            
            self.tokenizer = ElectraTokenizer.from_pretrained(self.meta['model'])
            self.model = TFElectraModel.from_pretrained(self.meta['model'])
        elif self.meta['name'] == 'distilbert':
            from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
            
            self.tokenizer = DistilBertTokenizer.from_pretrained(meta['model'])
            self.model = TFDistilBertForSequenceClassification.from_pretrained(self.meta['model'])
        elif self.meta['name'] == 'bert':
            from transformers import BertTokenizer, TFBertForSequenceClassification
            
            self.tokenizer = BertTokenizer.from_pretrained(meta['model'])
            self.model = TFBertForSequenceClassification.from_pretrained(self.meta['model'])
        else:
            raise NotImplementedError('Invalid model name...')