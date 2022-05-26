#!/usr/bin/env bash
# -*- coding: utf-8 -*-

dir=/home/ovsyannikovilyavl/respondelligent/rg/data/latest_training_files_mbart/

paste -d' ' $dir/train.lang_tags $dir/train.domain $dir/train.est_label $dir/train.rating $dir/train.review > $dir/train.review_tagged

paste -d' ' $dir/valid.lang_tags $dir/valid.domain $dir/valid.est_label $dir/valid.rating $dir/valid.review > $dir/valid.review_tagged

paste -d' ' $dir/test.lang_tags $dir/test.domain $dir/test.est_label $dir/test.rating $dir/test.review > $dir/test.review_tagged


paste -d' ' $dir/train.lang_tags $dir/train.response > $dir/train.response_tagged

paste -d' ' $dir/valid.lang_tags $dir/valid.response > $dir/valid.response_tagged

paste -d' ' $dir/test.lang_tags $dir/test.response > $dir/test.response_tagged

echo "done!"
