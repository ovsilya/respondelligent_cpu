# ReAdvisor: BART-style response generation model deployment

This directory contains all code required to build and run a
dockerized module with REST-API access.

In order to run successfully, make sure that:

- all underlying models and data are located in
  `./app/models/`
- relative paths in `config.json` match the location of the
  models and datafiles.

## Docker

To build the docker image, from this directory, run:

```
docker build -m 14g -t respogen .
```

where `respogen` is the name of the container
Once successfully built, run  the container with:

```
docker run -p 9530:8000 respogen
```

Then go to `http://0.0.0.0:8000/docs` to test interface.

### Docker save/load

If launching from a saved image, run:

```
docker load < respogen.tar.gz
docker run -p 8000:8000 rrgen_model1_de
```

## App

The entry point to the app is `fastapi_app/main.py` and can
be launched using

```
python fastapi_app/main.py fastapi_app/config.json
```

## Model inference

{
	"review": "Hatten ein Wochenende in Basel verbracht und da wir auch Karten für das Musical Theater hatten verbrachten wir die Nacht im Hotel du Commerce. Wunder schöne grosse Zimmer und ein reichhaltiges Frühstück versüssten uns das tolle Wochenende in Basel.",
	"meta": {
		"lang": "de",
		"title": "Schönes Erlebnis in Basel!",
		"domain": "hotel",
		"rating": 5,
		"author": "Hans Heissen",
		"gpe": "Basel",
		"loc": "",
		"email": "service@commercehotel.ch",
		"phone": "+41 XXXXXXXX",
		"url": "www.commercehotel.ch",
		"name": "hotel-du-commerce-basel",
		"company": "Hotel du Commerce",
		"greetings": ["Liebe*r NAMEPLACEHOLDER,", "Guten Tag NAMEPLACEHOLDER,"],
		"salutations": ["Besten Dank, Hanspeter, Team Leader.", "Mit freundlichen Grüssen, Hotel du Commerce."]
	}
}
To demo model inference, use `generate.py`, for example:
```
python app/src/generate.py \
    --model_path /srv/scratch6/kew/mbart/hospo_respo/ml_hosp_re_unmasked_untok/2021-04-30_12-40-05_w128-2021 \
    --checkpoint "/srv/scratch6/kew/mbart/hospo_respo/ml_hosp_re_unmasked_untok/2021-04-30_12-40-05_w128-2021/checkpointepoch=19_vloss=3.54154.ckpt" \
    --tokenizer /srv/scratch6/kew/mbart/hospo_respo/ml_hosp_re_unmasked_untok/2021-04-30_12-40-05_w128-2021/ \
    --tags_included \
    --max_output_len 512 \
    --max_input_len 512 \
    --batch_size 1 \
    --beam_size 4 \
    --num_return_sequences 4 \
    --do_sample --top_k 10 --temperature=1.2
```

If successful, the return object should look something like this:

```

{
    'review': 'Hatten ein Wochenende in Basel verbracht und da wir auch Karten für das Musical Theater hatten verbrachten wir die Nacht im Hotel du Commerce. Wunder schöne grosse Zimmer und ein reichhaltiges Frühstück versüssten uns das tolle Wochenende in Basel.',
    'meta': {'lang': 'de', 'title': 'Schönes Erlebnis in Basel!', 'domain': 'hotel', 'rating': 5, 'author': 'Hans Heissen', 'gpe': 'Basel', 'loc': '', 'email': 'service@commercehotel.ch', 'phone': '+41 XXXXXXXX', 'url': 'www.commercehotel.ch', 'name': 'hotel-du-commerce-basel', 'company': 'Hotel du Commerce', 'greetings': ['Liebe*r NAMEPLACEHOLDER,', 'Guten Tag NAMEPLACEHOLDER,'], 'salutations': ['Besten Dank, Hanspeter, Team Leader.', 'Mit freundlichen Grüssen, Hotel du Commerce.']},
    'responses': [
        'Guten Tag Hans Heissen, Vielen Dank, dass Sie unser Hotel für Ihr Wochenende gewählt haben und sich die Zeit für diese tolle Rückmeldung genommen haben. Wir freuen uns sehr zu hören, dass Sie sich bei uns wohl gefühlt haben und unser reichhaltiges Frühstück geniessen konnten. Wir hoffen entsprechend Sie schon bald wieder bei uns zu begrüssen, sei es wieder für ein Wochenende in Basel oder auch einmal für ein verlängertes Wochenende in der Stadt. Mit freundlichen Grüssen, Hotel du Commerce.',
        'Liebe*r Hans Heissen, Vielen Dank, dass Sie unser Hotel für Ihr Wochenende ausgewählt haben und sich die Zeit für ein Feedback genommen haben. Wir freuen uns sehr zu hören, dass Sie sich bei uns rundum wohl gefühlt haben und vor allem unser reichhaltiges Frühstück geniessen konnten. Wir hoffen Sie bei Ihrem nächsten Besuch in Basel wieder bei uns begrüssen zu dürfen. Wir sind übrigens täglich mit durchgehend warmer Küche für Sie da. Mit freundlichen Grüssen, Hotel du Commerce.',
        'Liebe*r Hans Heissen, Vielen Dank, dass Sie unser Hotel für Ihr Wochenende ausgewählt haben und sich die Zeit für diese tolle Rückmeldung genommen haben. Wir freuen uns sehr zu hören, dass Sie sich bei uns rundum wohl gefühlt haben und unser reichhaltiges Frühstück so richtig geniessen konnten. Wir hoffen entsprechend Sie schon bald wieder bei uns zu begrüssen, sei es wieder einmal für ein verlängertes Wochenende oder auch einmal für eine längere Auszeit vom Alltag. Mit freundlichen Grüssen, Hotel du Commerce.',
        'Guten Tag Hans Heissen, Vielen Dank, dass Sie sich für unser Hotel entschieden haben und sich die Zeit für diese tolle Bewertung genommen haben. Wir freuen uns sehr zu hören, dass Sie sich bei uns wohl gefühlt haben und vor allem das reichhaltige Frühstück so richtig geniessen konnten. Wir hoffen entsprechend Sie bei Ihrem nächsten Aufenthalt in Basel wieder bei uns zu begrüssen. Wir sind übrigens täglich ab dem Frühstück für Sie da. Mit freundlichen Grüssen, Hotel du Commerce.'
    ]
}

```

## Dependencies

- `spaCy`: is used for postprocessing model outputs (e.g.
  identifying and replacing named entities)
    - `en_core_web_md-2.3.1`
    - `de_core_news_md-2.3.0`
    - **NOTE** the spacy models shipped with this software
      have been fine-tuned using Prodigy's NER online
      training by Alex and Natalia (Feb 2021), however there
      is room for considerable customization and
      improvements on top of this.
    
- `SentenceTransformers` is used for rescoring model outputs
  by computing semantic textual similarity between the
  source text and generated hypotheses.
    - **NOTE** we do not provide the model files as they will be downloaded automatically when launching
      the docker container.
    - We found good results with moth
      `xx_paraphrase_xlm_r_multilingual_v1`
      and `xx_distiluse_base_multilingual_cased`

- `company_gazetteer` is a list of all restaurant and hotel
  names that appear in re:spondelligent's database AND the
  TripAdvisor data used in the ReAdvisor project. These
  names are added to both `spaCy` models using the
  `PhraseMatcher` (https://spacy.io/api/phrasematcher) which
  is an efficient method of adding gazetteer entries to a
  pretrained `spaCy` NER model.
  **NOTE** these names should be controlled to ensure
  ambiguous terms like 'EAT' do not introduce problems for the
  generation module's postprocessing component.
- `company_label_mapping` is used for models that are
  trained with company-specific labels (e.g. `<est_74>`).
  The mapping provided here MUST be the same as the one used
  for training the model.

## TODOs

- [ ] code comments / type hints
- [ ] adjust paths in config file for dockerization
- [ ] api access to gazetteer (live updating) (optional)

