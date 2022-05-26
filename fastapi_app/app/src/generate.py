#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Note, this code is adapted from Annette Rios' implementation for seq2seq simplification with LongmBART.
Models called 'simplify/simplification*' are simply encoder-decoder models for mBART.

"""

from typing import Dict, List, Tuple
import logging
import torch
from mbart_hospo_respo.inference import *
from mbart_hospo_respo.train import prepare_input
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
        "review": "Great location and morning espressos were a huge plus! Bed wasn’t that comfortable and room was tiny (+) Coffee machine was amazing! (-) The size of the room was pretty small and the bed was fairly hard.,",
        "meta": {
            "lang": "en",
            "title": "",
            "rating": 5,
            "domain": "hotel"}
        },
        {
        "review": "Pleasant, well decorated hotel in central location (+) Very central location in a lively part of the city. Despite it's location it was not noisy. It's the size of hotel, that makes you forget it's a hotel, especially as the staff was very friendly. The design is modern and welcoming and the rooms was comfortable if smallish.",
        "meta": {
            "lang": "en",
            "title": "",
            "rating": 3,
            "domain": "hotel"}
        },
        {
        "review": "Wenn man es nicht ruhig will... (+) Super Lage, allerdings für eine heiße Sommernacht zu laut in der Gegend. (-) Super Lage, allerdings für eine heiße Sommernacht zu laut in der Gegend.",
        "meta": {
            "lang": "de",
            "title": "",
            "rating": 3,
            "domain": "hotel"}
        },
        {
        "review": "Nicht buchen!!! Liebe Leute. Wenn ihr dieses Hotel buchen wollt, spart euch das Geld, und geht in ein anderes Hotel. DEFINITIV NICHT ZU EMPFEHLEN. Der Schein trügt!!! 1. Steht auf der Homepage: Klimaanlage vorhanden in den Zimmern. Wir hatten 3 Doppelzimmer. 2 Davon hatten keine. Es war viel zu heiss. 2. Die Rezeption deckt sich gegenseitig wenn sie einen Fehler machen. Sie lügen euch wirklich gerade aus ins Gesicht. Wir haben extra beim Check-In nachgefragt wann das Check-out wäre. Die Dame sagte uns : ihr könnt bis 14:00 bleiben. Ganz erstaunt haben wir uns alle 3 angeschaut und nochmals nachgefragt. Wir haben es also alle 3 gehört. Beim Checkout dann die Überraschung: einen Aufpreis von 60.- pro Zimmer. Die Dame an der Rezeption sagte uns das sei ganz klar so, und überall so ausgeschrieben. Die Dame vom Check in war auch noch da und wir haben sie nochmals gefragt, und sie sagte uns: nein sie habe check in gesagt. Sie haben laut eigener Aussage auch behauptet, sie wären bei allen 3 Zimmern klopfen gekommen. Wir waren alle 3 wach...also bitte Das ist einfach eine Lüge. Die Rezeptionistin hatte auch noch die Dreistigkeit, mit uns zu reden, als müsste sie uns eines besseren belehren. Einfach nur eine. Unverschämtheit. Wir waren schon öfters in Zürich. Aber so etwas haben wir noch nicht erlebt. Naja die Holiday Check Rezessionen sprechen ebenfalls für sich. Liebe Leute, macht nicht den Gleichen Fehler wie wir und zahlt so viel Geld für solch einen Service. Einfach nur kindisch, peinlich, bemitleidenswert!",
        "meta": {
            "lang": "de",
            "title": "",
            "rating": 1,
            "domain": "hotel"}
        },
        {
        "review": "Hervorragendes Preis-Leistungsverhältnis, super Lage, sehr modern & sauber! (+) Die Lage war sehr gut, da in Zürich alles wunderbar fußläufig erreichbar ist. Sehr gut war die kurze Entfernung zu Restaurants und zur zentralen Station der Tram. Die Zimmer entsprechen absolut den hier eingestellten Bildern der Unterkunft. (-) Irritierend war das Bild der Unterkunft einer Dachterrasse, die jedoch entgegen unserer Erwartung nicht allgemein für alle Hotelgäste zugänglich war sondern zu der Suite gehört. Dies sollte deutlicher dargestellt werden.",
        "meta": {
            "lang": "de",
            "title": "",
            "rating": 4,
            "domain": "hotel"}
        },
        {
        "review": "Sehr gut (+) Everything else besides lack of pillows and coffee in the room (-) Pillows - too soft. Did not sleep comfortable. Extra pilows in the room would be of great help. Missed coffee in the room. We are coffee lovers so would have loved some coffee while relaxing in the room",
        "meta": {
            "lang": "en",
            "title": "",
            "rating": 3,
            "domain": "hotel"}
        },
        {
        "review": "Hervorragend (+) Clean and quiet",
        "meta": {
            "lang": "en",
            "title": "",
            "rating": 4,
            "domain": "hotel"}
        },
        {
        "review": "Nice clean boutique hotel a short walk from the train station and all spots in Zurich. (+) Nice hotel a 10 minute walk from the Zurich main train station and close to lots of bars/restaurants. Rooms are cozy for 2 people but they also have a loft that is very large and gives you roof access.",
        "meta": {
            "lang": "en",
            "title": "",
            "rating": 4,
            "domain": "hotel"}
        },
            {
        "review": "TOP Hotel und freundliche Mitarbeiter:innen",
        "meta": {
            "lang": "de",
            "title": "",
            "rating": 5,
            "domain": "hotel"}
        },
        {
        "review": "Sehr gut (+) Zentrale Lage. Modern und Sauber. (-) Das erste Zimmer hatte einen unangenehmen Geruch. Es wurde aber ohne Stress ein anderes Zimmer bereitgestellt, bei dem war alles in Ordnung. Zu teuer für die Zimmerkategorie.",
        "meta": {
            "lang": "de",
            "title": "",
            "rating": 3,
            "domain": "hotel"}
        },
        {
        "review": "Service fehl am Platz... Bin seit etlichen Jahren spontaner Gast beim Sternengrill, denn die Wurst (mit dem Bürli und scharfen Senf) ist und bleibt einfach lecker. Standort ideal. Preis ok. Aber von Service kann hier gar nicht die Rede sein. Dies möchte ich hier klar festhalten. War heute seit langem wieder dort: Lang gezogenes Gesicht beim Personal am Grill - Ich bestelle die Spezialwurst mit Sauerkraut - die Augen werden verdreht \"keine Wurst, 15 Minuten\" in gebrochenem Deutsch. - ok dann halt eine normale Bratwurst, kein Problem. Das Geld wird kommentarlos eingesackt. Blick seitwärts. Man fühlt sich fast schon schuldig das Personal so sehr belastet zu haben.Bei jeder Dönerbude wird man feundlicher bedient. Von aussen währen dem verspeisen der Wurst, habe ich dann das Personal etwas weiter beobachtet: bei gewissen weiblichen Gästen war die Bedienung etwas positiver eingestellt - der abschliessende Blick auf den Hintern des Gastes - nun ja, dies kann man auf dem Bau erwarten... @Felix Zehnder: hast Du mal noch Zeit für einen Brief?Anyway, ein klein wenig anständiger Umgang mit den Gästen würde nicht schaden...",
        "meta": {
            "lang": "de",
            "title": "",
            "rating": 2,
            "domain": "hotel"}
        },
        {
        "review": "Great location and morning espressos were a huge plus! Bed wasn’t that comfortable and room was tiny (+) Coffee machine was amazing! (-) The size of the room was pretty small and the bed was fairly hard.",
        "meta": {
            "lang": "en",
            "title": "",
            "rating": 3,
            "domain": "hotel"}
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
