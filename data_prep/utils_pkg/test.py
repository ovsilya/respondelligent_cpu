#!/usr/bin/python3
# -*- coding: utf-8 -*-

import GerVADER.vaderSentimentGER as de_vader

de_sentiment = de_vader.SentimentIntensityAnalyzer()

print(de_sentiment.polarity_scores('Ich bin da.'))
