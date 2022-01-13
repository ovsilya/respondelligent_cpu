#!/usr/bin/python3
# -*- coding: utf-8 -*-

# import spacy
# import matplotlib as mlp
# from spacy_langdetect import LanguageDetector
import re
# import os
import pandas as pd
import copy
from tqdm.notebook import tqdm

# updated April 22, 2020
rst_grpids = {515, 516, 517, 518, 519, 8, 520, 10, 522, 14, 526, 16, 527, 18, 528, 534, 23, 537, 538, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 95, 98, 102, 106, 107, 108, 111, 112, 128, 129, 130, 131, 132, 139, 140, 141, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 188, 189, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 216, 217, 218, 221, 224, 225, 233, 234, 235, 236, 237, 249, 250, 251,
              252, 253, 261, 262, 263, 264, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 324, 325, 327, 374, 375, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 392, 395, 397, 398, 399, 400, 401, 402, 404, 405, 406, 407, 408, 409, 410, 411, 412, 414, 419, 420, 421, 429, 435, 440, 454, 455, 456, 457, 460, 462, 465, 468, 469, 472, 473, 474, 475, 476, 477, 480, 481, 483, 484, 485, 486, 487, 488, 489, 499}

htl_grpids = {260, 524, 535, 24, 536, 453, 458, 459, 461,
              464, 467, 478, 479, 96, 100, 493, 494, 116, 372, 255}


# rst_grpids = [8, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 88, 95, 97, 98, 102, 106, 107, 108, 111, 112, 128, 129, 130, 131, 132, 139, 140, 141, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 188, 189, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 216, 217, 218, 221, 224, 225, 233, 234, 235, 236, 237, 249, 250, 251, 252, 253, 261, 262, 263, 264, 267, 275, 277, 283, 284, 285, 286, 287, 288,
#               289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 321, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 374, 375, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 390, 392, 395, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 409, 410, 411, 412, 414, 419, 420, 421, 435, 440, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 454, 455, 456, 457, 460]

# htl_grpids = [24, 96, 100, 116, 232, 255, 260,
#               276, 278, 279, 372, 391, 426, 453, 458, 459]

sf_guard_group_cols = {
    "id": "Int64",
    "name": "string",
    "description": "string",
    "culture": "string",
    "timezone": "string",
    "expirationdate": "string",
    "reminderdays": "Int64",
    "maxuser": "Int64",
    "country": "string",
    "token": "string",
    "company": "string",
    "customercomment": "string",
    "logo": "string",
    "created_at": str,
    "updated_at": str,
    "city": "string",
    "phoneb": "string",
    "emailb": "string",
    "address": "string",
    "answerstart": "string",
    "greetingtype": "string",
    "goodbyetext": "string",
    "keywordstouse": "string",
    "keywordcomment": "string",
    "google_access_token": "string",
    "facebook_access_token": "string",
    "recategoryid": "float64",
    "rechannel_id": "float64",
    "guestfeedbackemail": "string"
}

review_dtypes = {
    "id": "Int64",
    "groupid": "Int64",
    "platformid": "Int64",
    "reviewid": "string",
    "reviewdate": str,
    "author": "string",
    "rating": "Int64",
    "url": "string",
    "reviewtitle": "string",
    "reviewtext": "string",
    "created_at": str,
    "updated_at": str,
    "reviewstatus": "Int64",
    "deleted_at": str,
    "imported_at": str,
    "assigneduserid": "Int64",
    "reviewcreated_at": str,
    "platformrating": float,
    "reviewlang": "string",
    "assigned_at": str
}

review_dates = ["reviewdate", "created_at", "updated_at",
                "deleted_at", "imported_at", "reviewcreated_at", "assigned_at"]

answer_dtypes = {
    "id": "Int64",
    "groupid": "Int64",
    "userid": "Int64",
    "reviewid": "Int64",
    "platformid": "Int64",
    "answerdate": str,
    "answerauthor": "string",
    "answertext": "string",
    "status": "string",
    "created_at": str,
    "updated_at": str
}

answer_dates = ["answerdate", "created_at", "updated_at"]


if __name__ == "__main__":
    pass
