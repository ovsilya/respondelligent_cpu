#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Helper function for loading in data from Fairseq generation
output. The function extends the function provided in the
VizSeq module.
"""

from typing import List, Dict, Union
from collections import Counter
import json

def get_data_from_jsonl_generation_files(log_path_or_paths: Union[str, List[str]], nbest: int = 1):
    """
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
        cur_ids, cur_src ,cur_ref, cur_hypo = [], [], [], []
        with open(log_path) as f:
            for i, l in enumerate(f):
                cur_ids.append(i)
                line = json.loads(l.strip())
                # if src and line['src'] != src[i]:
                #     breakpoint()
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


# def get_data_sorted_by_ids(log_path_or_paths: Union[str, List[str]], nbest: int = 1):
#     """
#     Extension of vizseq _get_data().
    
#     - Handles src factor lines (prefixed by 'F-\d+') 
#     - Sorts src, ref, hyps by ids to match input line files. 

#     """
#     if isinstance(log_path_or_paths, str):
#         log_path_or_paths = [log_path_or_paths]
#     ids, src, src_factor, ref, hypo = None, None, None, None, {}
#     names = Counter()
#     for k, log_path in enumerate(log_path_or_paths):
#         # assert op.isfile(log_path)
#         cur_src, cur_src_factor,cur_ref, cur_hypo = {}, {}, {}, {}
#         with open(log_path) as f:
#             for l in f:
#                 line = l.strip()
#                 if line.startswith('H-'):
#                     _id, _, sent = line.split('\t', 2)
#                     # cur_hypo[_id[2:]] = sent
#                     # collect multiple hypotheses for each ID (nbest > 1)
#                     if not _id[2:] in cur_hypo:
#                         cur_hypo[_id[2:]] = [sent]
#                     else:
#                         cur_hypo[_id[2:]].append(sent)
#                 elif line.startswith('T-'):
#                     _id, sent = line.split('\t', 1)
#                     cur_ref[_id[2:]] = sent
#                 elif line.startswith('S-'):
#                     _id, sent = line.split('\t', 1)
#                     cur_src[_id[2:]] = sent
#                 elif line.startswith('F-'):
#                     _id, sent = line.split('\t', 1)
#                     cur_src_factor[_id[2:]] = sent
        
#         # import pdb; pdb.set_trace()
        
#         cur_ids = sorted(list(cur_src.keys()), key=int)
#         # cur_ids = sorted((int(id) for id in cur_src.keys()))

#         # breakpoint()

#         assert set(cur_ids) == set(cur_ref.keys()) == set(cur_hypo.keys())
#         cur_src = [cur_src[i] for i in cur_ids]
#         if cur_src_factor:
#             cur_src_factor = [cur_src_factor[i] for i in cur_ids]
#         cur_ref = [cur_ref[i] for i in cur_ids]
        
#         if k == 0:
#             ids, src, src_factor, ref = cur_ids, cur_src, cur_src_factor, cur_ref
#         else:
#             assert set(ids) == set(cur_ids), f"IDs in {log_path} do not match IDs from other log files!"
#             # assert set(src) == set(cur_src)
#             # assert set(ref) == set(cur_ref)
#             # assert set(src_factor) == set(cur_src_factor)
#         # name = op.splitext(op.basename(log_path))[0]
#         name = log_path # use full path as hypotheses id (key in dict)
#         names.update([name])
#         if names[name] > 1:
#             name += f'.{names[name]}'
#         # hypo[name] = [cur_hypo[i] for i in cur_ids]
#         if nbest == 1:
#             # ensure that hypo = {'name': ['hyp1', 'hyp2',
#             # etc]}
#             hypo[name] = [cur_hypo[i][:nbest][0] for i in cur_ids]
#         else:
#             # hypo will be {'name': [['hyp1.1', 'hyp1.2', 'hyp1.3', ...], ['hyp2.1', 'hyp2.2', 'hyp2.3', ...], [etc]]}
#             # other functions will need to be extended for compatibility
#             hypo[name] = [cur_hypo[i][:nbest] for i in cur_ids]

#     # breakpoint()
    
#     return {'0': src}, {'0': ref}, hypo, ids

if __name__ == "__main__":
    pass