import sys
import re
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

# from tqdm import tqdm
from typing import List, Dict, Tuple
# from vizseq.ipynb import fairseq_viz as fs
from read_jsonl_generation_log import *
# from rerank_hypotheses import *

SEED = 42
random.seed(SEED)

"""

Takes JSONL-style output files (as produced by mBART implementation) from *MULTIPLE* models and
produces randomised JSONL dataset for human evaluation in Prodigy.

    python generate_data_for_prodigy_from_jsonl_outputs.py -m ./model_outputs_de/ft100src_rg.greedy.txt ./model_outputs_de/ft100src_rg.nbest5_topk10.txt -o rrgen_human_eval_v1/rrgen_human_eval.de.jsonl -n 10

NOTE: test sets must be the same! Will not work with the
outputs from rrgen_up_down_sampling and rrgen as different
items are skipped during testing and validation.

"""

def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_outputs', nargs='*', required=True, help='set of model output files produced by longmBART (simplify generation script).')
    ap.add_argument('-r', '--reference_inputs', nargs='*', required=False, help='set of reference input files for test set.')
    ap.add_argument('-o', '--outfile', required=True, help='filepath to output JSONL file that is read in by Prodigy')
    ap.add_argument('-n', type=int, default=0, help='number of items in jsonl dataset. Note: total annotation items in JSONL dataset is N * num_models')
    ap.add_argument('--include_tgt', action='store_true', help='whether or not to include ground truth in output JSON')
    ap.add_argument('--rerank', action='store_true', help='set flag to rerank an nbest list using lexical overlap measure')
    ap.add_argument('--truecase', type=str, help='path to trained truecase model')
    return ap.parse_args()

def add_line_breaks(text: str) -> str:
    """
    replace '---SEP---' with clear linebreaks to improve readability
    NB. in later versions, ---SEP--- was replaces with the tag '<endtitle>'
    """
    text = re.sub(r'\s?---SEP---\s?', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'\s?<endtitle>\s?', '\n\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<GREETING>\s?([\.\?\!\,\-]?)', '<GREETING>\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<SALUTATION>', '\n<SALUTATION>', text, flags=re.IGNORECASE)
    return text

def collect_input_meta(reference_files):
    meta = {}
    for ref_file in reference_files:
        file_type = ref_file.split('.')[-1]
        with open(ref_file, 'r', encoding='utf8') as f:
            meta[file_type] = f.read().splitlines()
    return meta

if __name__ == "__main__":
    args = set_args()

    # if len(args.model_outputs) > 2:
    #     sys.exit("\n[!] Script only expects two model output files! :(\n")
    if args.reference_inputs:
        eval_set_meta = collect_input_meta(args.reference_inputs)
    else:
        eval_set_meta = None

    # breakpoint()

    if args.rerank:
        srcs, refs, hyps, ids = get_data_from_jsonl_generation_files(args.model_outputs, nbest=10)
        for hyp_file, nbest_hyps in hyps.items():
            hyps[hyp_file] = [lexical_overlap_rerank(src, nbest_list) for src, nbest_list in zip(srcs['0'], nbest_hyps)]
    else:
        srcs, refs, hyps, ids = get_data_from_jsonl_generation_files(args.model_outputs, nbest=1)
    
    # NOTE data structures:
    # srcs = {'0': [...]}
    # refs = {'0': [...]}
    # hyps = {'file1': [...], 'file2': [...]}
    # ids = ['id_a', 'id_b', 'id_c']

    #breakpoint()

    if not args.n:
        selection = random.sample(list(enumerate(ids)), len(ids))
    else:
        selection = random.sample(list(enumerate(ids)), args.n)

    # seen_pairs = set()

    with open(args.outfile, 'w', encoding = 'utf8') as outf:
        
        for item in selection:
            batch = []
            idx = item[0]
            id = int(item[1])
        
            src_text = srcs['0'][idx]
            tgt_text = refs['0'][idx]

            # select meta data for patricular item
            item_meta = {}
            item_meta['eval_id'] = idx
            if eval_set_meta:
                for k in eval_set_meta:
                    item_meta[k] = eval_set_meta[k][idx]

            anno_fields = {"fluency": None, "repetition": None, "specif": None, "approp": None, "sent_acc": None, "dom_acc": None}

            if args.truecase:
                src_text = add_line_breaks(' '.join(caser.get_true_case_from_tokens(src_text.split(), out_of_vocabulary_token_option="as-is")))
                tgt_text = add_line_breaks(' '.join(caser.get_true_case_from_tokens(tgt_text.split(), out_of_vocabulary_token_option="as-is")))
            else:
                src_text = add_line_breaks(src_text)
                tgt_text = add_line_breaks(tgt_text)

            tgt_item = {
                        "text": tgt_text,
                        "src": src_text,
                        "model_name": "tgt",
                        "meta": item_meta,
                        "anno": anno_fields
                    }

            if args.include_tgt:
                batch.append(tgt_item)

            for model_name in hyps.keys():
                
                hyp_text = hyps[model_name][idx]

                if args.truecase:
                    hyp_text = add_line_breaks(' '.join(caser.get_true_case_from_tokens(hyp_text.split(), out_of_vocabulary_token_option="as-is")))
                else:
                    hyp_text = add_line_breaks(hyp_text)

                # hashed_pair = hash(src_text + ' ' + hyp_text)

                # if hashed_pair not in seen_pairs:
                    
                #     seen = False
                #     seen_pairs.add(hashed_pair)
                # else:
                #     seen = True

                model_item = {
                        "text": hyp_text,
                        "src": src_text,
                        "model_name": model_name,
                        "meta": item_meta,
                        "anno": anno_fields
                    }
                batch.append(model_item)

            random.shuffle(batch)

            for entry in batch:
                json_line = json.dumps(entry, ensure_ascii=False)
                outf.write(json_line + '\n')


    # print(f'unique pair count: {len(seen_pairs)}')
    print(f'Output JSONL file written to {args.outfile}')
