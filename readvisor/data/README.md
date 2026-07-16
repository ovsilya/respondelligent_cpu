# Data Processing Scripts

This directory contains various scripts used for preparing review-response data for model training.

The formated expected by our customised mBART model is:

Review (src):
    ```
    en_XX <restaurant> <est_1> <4> Best Fondue <endtitle> The waiters ... good.
    ```

Response (tgt):
    ```
    en_XX <GREETING> We are honoured ... glass. <SALUTATION>
    ```

In order to transform review-response pairs from
re:spondelligent's database, we provide scripts to: 
- parse extracted JSON files from the 2020_04 and 2021_01
  database dumps and convert these to well-formed CSV (see
  `convert_json_exports_to_well_formed_csv.py` and `merge_rechannel_and_sf_guard_group.py`)
- clean texts in CSV files and convert to pandas dataframe
  (see `clean_respondelligent_data_from_extracted_csv.py`)
- generate an up-to-date mapping between establishment names
  in the dataset and labels for models (see `collect_establishment_occurence_freq_counts_from_respondelligent_data.py`)
- generate line-aligned src/tgt/meta input files for model
  (see `generate_model_files_for_mBART.ipynb`)
- prepend relevant label tokens to src and tgt texts (see `prepend_labels_to_review_texts_for_mbart.sh`)

## Steps to reproduce

From JSON extracted tables:

```
python convert_json_exports_to_well_formed_csv.py \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2020_04/exported_from_mysql/json \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2020_04/exported_from_mysql/csv
```

```
python merge_rechannel_and_sf_guard_group.py \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2020_04/exported_from_mysql/json \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2020_04/exported_from_mysql/csv
```

From csv to pandas DF with cleaning:
```
python clean_respondelligent_data_from_extracted_csv.py \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv/respo_data.pkl
```

Get establishment labels:
```
python collect_establishment_occurence_freq_counts_from_respondelligent_data.py \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv/respo_data.pkl \
/mnt/storage/clfiles/projects/readvisor/respondelligent/2021_01/exported_from_mysql/csv/est_labels.pkl
```
**NOTE** This step is only necessary if training an
establishment-aware model. The generated output file needs
to be specified in `generate_model_files_for_mBART.ipynb`
and also given to the final model for performing inference
(see `config.json` in `fastapi-app`).

From pandas DF to model input files:
```
generate_model_files_for_mBART.ipynb
```

Prepend tags to inputs for mBART:
```
bash prepend_labels_to_review_texts_for_mbart.sh
```

