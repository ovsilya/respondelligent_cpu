#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from spacy.tokens import Doc

class WhitespaceTokenizer:
    """simple whitespace tokenizer to be used when
    input is pretokenised (e.g. output of sentiment
    analysis system)"""
    
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)

def add_special_tokens_to_tokenizer(nlp):
    nlp.tokenizer.add_special_case("---SEP---", [{"ORTH": "---SEP---"}])
    nlp.tokenizer.add_special_case("<endtitle>", [{"ORTH": "<endtitle>"}])
    nlp.tokenizer.add_special_case("<URL>", [{"ORTH": "<URL>"}])
    nlp.tokenizer.add_special_case("<DIGIT>", [{"ORTH": "<DIGIT>"}])
    nlp.tokenizer.add_special_case("<EMAIL>", [{"ORTH": "<EMAIL>"}])
    nlp.tokenizer.add_special_case("<NAME>", [{"ORTH": "<NAME>"}])
    nlp.tokenizer.add_special_case("<LOC>", [{"ORTH": "<LOC>"}])
    nlp.tokenizer.add_special_case("<GPE>", [{"ORTH": "<GPE>"}])
    nlp.tokenizer.add_special_case("[sep]", [{"ORTH": "[sep]"}])
    nlp.tokenizer.add_special_case("[cls]", [{"ORTH": "[cls]"}])
    nlp.tokenizer.add_special_case("<GREETING>", [{"ORTH": "<GREETING>"}])
    nlp.tokenizer.add_special_case("<SALUTATION>", [{"ORTH": "<SALUTATION>"}])
    print('Updated special tokens in spacy tokenizer.')


if __name__ == "__main__":
    pass