#!/usr/bin/env python3

"""
Helper function for loading in data from Fairseq generation
output. The function extends the function provided in the
VizSeq module.
"""

import json
from collections import Counter
from typing import List, Union


def get_data_from_jsonl_generation_files(log_path_or_paths: Union[str, List[str]], nbest: int = 1):
    r"""
    Extension of vizseq _get_data().

    - Handles src factor lines (prefixed by 'F-\d+')
    - Sorts src, ref, hyps by ids to match input line files.

    """
    if isinstance(log_path_or_paths, str):
        log_path_or_paths = [log_path_or_paths]
    ids, src, ref, hypo = None, None, None, {}
    names = Counter()
    for k, log_path in enumerate(log_path_or_paths):
        # assert op.isfile(log_path)
        cur_ids, cur_src, cur_ref, cur_hypo = [], [], [], []
        with open(log_path) as f:
            for i, raw in enumerate(f):
                cur_ids.append(i)
                line = json.loads(raw.strip())
                cur_src.append(line['src'])
                cur_ref.append(line['ref'])
                cur_hypo.append([hyp_dict['hyp'] for hyp_dict in line['hyps']]) # skip over scores
                
        if k == 0:
            ids, src, ref = cur_ids, cur_src, cur_ref
        else:
            assert set(ids) == set(cur_ids), f"IDs in {log_path} do not match IDs from other log files!"
            # for truncated sources, checking for exact
            # matches leads to probelms
            # since Huggingface decodes without reording,
            # it's safe to assume to matches!
            # assert set(src) == set(cur_src), f"src texts in {log_path} do not match src texts from other log files!"
            assert set(ref) == set(cur_ref), f"ref texts in {log_path} do not match src texts from other log files!"

        name = log_path # use full path as hypotheses id (key in dict)
        names.update([name])
        if names[name] > 1:
            name += f'.{names[name]}'
        # hypo[name] = [cur_hypo[i] for i in cur_ids]
        if nbest == 1:
            # ensure that hypo = {'name': ['hyp1', 'hyp2',
            # etc]}
            hypo[name] = [cur_hypo[i][0] for i in cur_ids]
        else:
            hypo[name] = [cur_hypo[i][:nbest] for i in cur_ids]

    return {'0': src}, {'0': ref}, hypo, ids