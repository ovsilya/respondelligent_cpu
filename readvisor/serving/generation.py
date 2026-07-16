#!/usr/bin/env python3

"""mBART response-generation helpers for the FastAPI serving app.

Loads a fine-tuned mBART checkpoint and runs (optionally stochastic) beam
decoding to produce candidate responses for a review.

Note: this code is adapted from Annette Rios' implementation for seq2seq
simplification with LongmBART. Models called ``simplify``/``simplification*``
are simply encoder-decoder models for mBART.

Run a self-contained smoke test (needs a real checkpoint) with::

    python -m readvisor.serving.generation --model_path ... --checkpoint ...
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from readvisor.model.inference import (
    InferenceSimplifier,
    MBartForConditionalGeneration,
)
from readvisor.model.train import prepare_input

logger = logging.getLogger(__name__)

# Demo payload used by the ``main()`` smoke test (externalised from source).
DEMO_REVIEWS_PATH = Path(__file__).resolve().parent / "egs" / "demo_reviews.json"


def load_model_from_checkpoint(args: argparse.Namespace) -> InferenceSimplifier:
    """Instantiate an :class:`InferenceSimplifier` and load its checkpoint weights.

    Args:
        args: Parsed model arguments (must expose ``model_path`` and
            ``checkpoint_name``).

    Returns:
        The response-generation model with weights loaded (on CPU).
    """
    checkpoint_path = os.path.join(args.model_path, args.checkpoint_name)
    response_generator = InferenceSimplifier(args)
    cp = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    response_generator.model = MBartForConditionalGeneration.from_pretrained(args.model_path)
    response_generator.load_state_dict(cp["state_dict"])
    logger.info("model loaded successfully")
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
        review = title + " <endtitle> " + review

    if cfg and cfg.response_generator.type in ["mBART_DR", "mBART_DER"]:
        # expected format (DR): <domain> <rating> <review>
        rating = response_generation_input["meta"].get("rating", None)
        if rating is not None:
            if rating < 1:
                rating = 1  # NOTE: model expects review ratings between [1,5]
            rating_label = "<" + str(rating) + ">"
            review = rating_label + " " + review

        if cfg.response_generator.type == "mBART_DER":
            # expected format (DER): <domain> <company> <rating> <review>
            company_label = response_generation_input["meta"].get("company_label", "<est_0>")
            review = company_label + " " + review

        domain = response_generation_input["meta"].get("domain", None)
        if domain is not None:
            domain_label = "<" + str(domain) + ">"
            review = domain_label + " " + review

    # expected format: <lang_tag> [<domain> <company> <rating>] <review>
    review = lang_tag + " " + review

    return {
        "review": review,
        "tgt_tag": lang_tag,
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
            src_texts=input_dict["review"],
            tags_included=True,
            max_length=model.max_input_len,
            max_target_length=model.max_output_len,
            truncation=True,
            padding=False,
            return_tensors="pt",
        )

        # Reorder language tag  to the end of the source
        # sequence (doing this here avoids it being
        # truncated for long sequences)
        input_ids = sample["input_ids"].squeeze()
        input_ids = torch.cat([input_ids[1:], input_ids[:1]])
        input_ids = input_ids.unsqueeze(0)

        batch = (input_ids, input_dict.get("reference", [None]), [input_dict.get("tgt_tag")])

        yield batch


def generate(model, args: argparse.Namespace, batch) -> List[Tuple[float, str]]:
    """Run decoding for a single batch and return ``(score, hypothesis)`` pairs."""

    logger.info("Generating for target tags: %s", str(batch[-1]))

    input_ids, refs, tags = batch
    input_ids, attention_mask = prepare_input(input_ids, model.tokenizer.pad_token_id)
    assert (
        refs[0] is not None or tags[0] is not None
    ), "Need either reference with target labels or list of target labels!"
    if refs[0] is not None:
        tgt_ids = [model.tokenizer.lang_code_to_id[sample.split(" ")[0]] for sample in refs]  # first token
    elif tags[0] is not None:
        # get decoder_start_token_ids from file in target_tags
        tgt_ids = [model.tokenizer.lang_code_to_id[sample.split(" ")[0]] for sample in tags]

    decoder_start_token_ids = torch.tensor(
        tgt_ids, dtype=input_ids.dtype, device=input_ids.device
    ).unsqueeze(1)

    generated_ids = model.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
        max_length=args.max_output_len,
        num_beams=args.beam_size,
        pad_token_id=model.tokenizer.pad_token_id,
        decoder_start_token_ids=decoder_start_token_ids,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        num_return_sequences=args.num_return_sequences,
        output_scores=True,
        return_dict_in_generate=True,
    )

    hyp_strs = model.tokenizer.batch_decode(generated_ids.sequences.tolist(), skip_special_tokens=True)
    hyp_scores = generated_ids.sequences_scores.tolist()

    return [(score, hyp) for score, hyp in zip(hyp_scores, hyp_strs)]


def main(args: argparse.Namespace) -> None:
    """Smoke test: load the model and generate for the packaged demo reviews."""
    model = load_model_from_checkpoint(args)

    with open(DEMO_REVIEWS_PATH, encoding="utf8") as f:
        inputs = json.load(f)

    for inp in inputs:
        model_input = prepare_input_for_mbart(inp)
        for batch in batchify(model, [model_input]):
            logger.info("batch: %s", batch)
            responses = generate(model, args, batch)
            logger.info("responses: %s", responses)


if __name__ == "__main__":
    main_arg_parser = argparse.ArgumentParser(description="simplification")
    parser = InferenceSimplifier.add_model_specific_args(main_arg_parser, os.getcwd())
    args = parser.parse_args()
    main(args)
