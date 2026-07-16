"""Build a randomised JSONL dataset for human evaluation in Prodigy.

Takes JSONL-style output files (as produced by the mBART implementation) from
*MULTIPLE* models and produces a randomised JSONL dataset for human evaluation
in Prodigy.

    python -m readvisor.evaluation.build_prodigy_dataset \
        -m ./model_outputs_de/ft100src_rg.greedy.txt ./model_outputs_de/ft100src_rg.nbest5_topk10.txt \
        -o rrgen_human_eval_v1/rrgen_human_eval.de.jsonl -n 10

NOTE: test sets must be the same! Will not work with the outputs from
rrgen_up_down_sampling and rrgen as different items are skipped during testing
and validation.
"""

import argparse
import json
import logging
import random
import re

from readvisor.evaluation.jsonl_loader import get_data_from_jsonl_generation_files
from readvisor.evaluation.rerank import lexical_overlap_rerank

logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)


def set_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model_outputs', nargs='*', required=True, help='set of model output files produced by longmBART (simplify generation script).')
    ap.add_argument('-r', '--reference_inputs', nargs='*', required=False, help='set of reference input files for test set.')
    ap.add_argument('-o', '--outfile', required=True, help='filepath to output JSONL file that is read in by Prodigy')
    ap.add_argument('-n', type=int, default=0, help='number of items in jsonl dataset. Note: total annotation items in JSONL dataset is N * num_models')
    ap.add_argument('--include_tgt', action='store_true', help='whether or not to include ground truth in output JSON')
    ap.add_argument('--rerank', action='store_true', help='set flag to rerank an nbest list using lexical overlap measure')
    return ap.parse_args()

def add_line_breaks(text: str) -> str:
    """Replace ``---SEP---`` (or ``<endtitle>``) with clear linebreaks.

    Improves readability of the source/response texts shown in Prodigy.
    NB. in later versions, ---SEP--- was replaced with the tag '<endtitle>'.
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
        with open(ref_file, encoding='utf8') as f:
            meta[file_type] = f.read().splitlines()
    return meta

if __name__ == "__main__":
    args = set_args()

    if args.reference_inputs:
        eval_set_meta = collect_input_meta(args.reference_inputs)
    else:
        eval_set_meta = None

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

    if not args.n:
        selection = random.sample(list(enumerate(ids)), len(ids))
    else:
        selection = random.sample(list(enumerate(ids)), args.n)

    with open(args.outfile, 'w', encoding='utf8') as outf:

        for item in selection:
            batch = []
            idx = item[0]

            src_text = add_line_breaks(srcs['0'][idx])
            tgt_text = add_line_breaks(refs['0'][idx])

            # select meta data for particular item
            item_meta = {}
            item_meta['eval_id'] = idx
            if eval_set_meta:
                for k in eval_set_meta:
                    item_meta[k] = eval_set_meta[k][idx]

            anno_fields = {"fluency": None, "repetition": None, "specif": None, "approp": None, "sent_acc": None, "dom_acc": None}

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

                hyp_text = add_line_breaks(hyps[model_name][idx])

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

    logger.info('Output JSONL file written to %s', args.outfile)
