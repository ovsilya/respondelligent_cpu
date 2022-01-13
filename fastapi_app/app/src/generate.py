#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Note, this code is adapted from Annette Rios' implementation for seq2seq simplification with LongmBART.
Models called 'simplify/simplification*' are simply encoder-decoder models for mBART.

"""

from typing import Dict, List, Tuple
import logging
import torch
from .mbart_hospo_respo.inference import *
from .mbart_hospo_respo.train import prepare_input
from memory_profiler import profile

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@profile
def load_model_from_checkpoint(args):
    checkpoint_path=os.path.join(args.model_path, args.checkpoint_name)
    response_generator = InferenceSimplifier(args)
    cp = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    response_generator.model = MBartForConditionalGeneration.from_pretrained(args.model_path)
    response_generator.load_state_dict(cp["state_dict"])
    logging.info('model loaded successfully')
    return response_generator

def prepare_input_for_mbart(response_generation_input: Dict, cfg=None) -> Dict:
    """
    Prepend relevant labels and language tag to review text and set
    target response language tag.

    Labels for domain, rating, review title are appended if
    available in the response_generation_input object.

    Args:
        response_generation_input (dict): See example in input_egs.json
    
    Returns:
        obj (dict) to be consumed 

    """
    lang_tags_map = {
        "en": "en_XX",
        "de": "de_DE",
    }
    
    review = response_generation_input["review"]

    lang = response_generation_input["meta"].get("lang")
    lang_tag = lang_tags_map[lang]
    
    title = response_generation_input["meta"].get("title", None)
    if title is not None:
        review = title + ' <endtitle> ' + review

    if cfg and cfg.response_generator.type in ['mBART_DR', 'mBART_DER']:
        # expected format (DR): <domain> <rating> <review>
        rating = response_generation_input["meta"].get("rating", None)
        if rating is not None:
            if rating < 1:
                rating = 1 # NOTE: model expects review ratings between [1,5]
            rating_label = '<' + str(rating) + '>'
            review = rating_label + ' ' + review

        if cfg.response_generator.type == 'mBART_DER':
            # expected format (DER): <domain> <company> <rating> <review>
            company_label = response_generation_input["meta"].get("company_label", '<est_0>')
            review = company_label + ' ' + review

        domain = response_generation_input["meta"].get("domain", None)
        if domain is not None:
            domain_label = '<' + str(domain) + '>'
            review = domain_label + ' ' + review
            
    # expected format: <lang_tag> [<domain> <company> <rating>] <review>
    review = lang_tag + ' ' + review
    
    return {
        'review': review,
        'tgt_tag': lang_tag,
    }
  

def batchify(model, response_generation_inputs: List[Dict]) -> Tuple[torch.Tensor, List, List]:
    """
    implicilty yields a batch of size 1 
    
    NOTE: If memory consumption / generation time is not an
    issue, this may be altered to return batch size > 1. 
    
    However, for response generation with stochastic
    decoding strategies, it's probably best to increase
    the beam size to get more varied hypotheses in return.
    """

    for input_dict in response_generation_inputs:
        
        sample = model.tokenizer.prepare_seq2seq_batch(
            src_texts=input_dict['review'],
            tags_included=True,
            max_length=model.max_input_len,
            max_target_length=model.max_output_len,
            truncation=True,
            padding=False,
            return_tensors="pt"
            )

        # Reorder language tag  to the end of the source
        # sequence (doing this here avoids it being
        # truncated for long sequences)
        input_ids = sample['input_ids'].squeeze()
        input_ids = torch.cat([input_ids[1:], input_ids[:1]])
        input_ids = input_ids.unsqueeze(0)

        batch = (input_ids, input_dict.get('reference', [None]), [input_dict.get('tgt_tag')])

        yield batch

@profile
def generate(model, args, batch):
    
    logging.info('Generating for target tags: {}'.format(str(batch[-1])))

    input_ids, refs, tags = batch
    input_ids, attention_mask = prepare_input(input_ids, model.tokenizer.pad_token_id)
    assert (refs[0] is not None or tags[0] is not None), "Need either reference with target labels or list of target labels!"
    if refs[0] is not None:
        tgt_ids = [model.tokenizer.lang_code_to_id[sample.split(' ')[0]]  for sample in refs] # first token
    elif tags[0] is not None:
        # get decoder_start_token_ids from file in target_tags
        tgt_ids = [model.tokenizer.lang_code_to_id[sample.split(' ')[0]]  for sample in tags]

    decoder_start_token_ids = torch.tensor(tgt_ids, dtype=input_ids.dtype, device=input_ids.device).unsqueeze(1)
    
    generated_ids = model.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                        use_cache=True, max_length=args.max_output_len,
                                        num_beams=args.beam_size, pad_token_id=model.tokenizer.pad_token_id, decoder_start_token_ids=decoder_start_token_ids,
                                        do_sample=args.do_sample,
                                        temperature=args.temperature,
                                        top_k=args.top_k,
                                        top_p=args.top_p,
                                        repetition_penalty=args.repetition_penalty,
                                        length_penalty=args.length_penalty,
                                        num_return_sequences=args.num_return_sequences,
                                        output_scores=True,
                                        return_dict_in_generate=True)

    hyp_strs = model.tokenizer.batch_decode(generated_ids.sequences.tolist(), skip_special_tokens=True)
    hyp_scores = generated_ids.sequences_scores.tolist()
    # source_strs = model.tokenizer.batch_decode(input_ids.tolist(), skip_special_tokens=True)

    return [(score, hyp) for score, hyp in zip(hyp_scores, hyp_strs)]


def main(args):
    model = load_model_from_checkpoint(args)
    
    # minimal example
    inputs = [
        {
        "review": "Hatten ein Wochenende in Basel verbracht und da wir auch Karten für das Musical Theater hatten verbrachten wir die Nacht im Hotel du Commerce. Wunder schöne grosse Zimmer und ein reichhaltiges Frühstück versüssten uns das tolle Wochenende in Basel.",
        "meta": {
            "lang": "de",
            "title": "Schönes Erlebnis in Basel!",
            "rating": 5,
            "domain": "hotel"}
        },
        {
        "review": "Elana the staff member was amazing. One of the best hospitality workers we have ever encountered. Friendly, engaging and knowledgeable when it came to the food and beverages being served. We enjoyed some great swiss wine and beer with the deserts we ordered.",
        "meta": {
            "lang": "en",
            "title": "Amazing",
            "rating": 5,
            "domain": "restaurant"}
        },
    ]

    for inp in inputs: 
        model_input = prepare_input_for_mbart(inp)
        for batch in batchify(model, [model_input]):
            print(batch)
            responses = generate(model, args, batch)
            print(responses)
        

if __name__ == '__main__':
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = InferenceSimplifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
