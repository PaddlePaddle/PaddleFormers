# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

MODE=$1

if [ ${MODE} = "lite_train_infer" ]; then
    cd ../examples/machine_translation/transformer/
    # The whole procedure of lite_train_infer should be less than 15min.
    # Hence, set maximum output length is 16. 
    sed -i "s/^max_out_len.*/max_out_len: 16/g" configs/transformer.base.yaml
    sed -i "s/^batch_size.*/batch_size: 3072/g" configs/transformer.base.yaml
    sed -i "s/^max_out_len.*/max_out_len: 16/g" configs/transformer.big.yaml
    sed -i "s/^batch_size.*/batch_size: 3072/g" configs/transformer.big.yaml

    sed -i "s/^random_seed:.*/random_seed: 128/g" configs/transformer.base.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: False/g" configs/transformer.base.yaml
    sed -i "s/^shuffle:.*/shuffle: False/g" configs/transformer.base.yaml

    sed -i "s/^random_seed:.*/random_seed: 128/g" configs/transformer.big.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: False/g" configs/transformer.big.yaml
    sed -i "s/^shuffle:.*/shuffle: False/g" configs/transformer.big.yaml
    # Data set prepared. 
    if [ ! -f WMT14.en-de.partial.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.partial.tar.gz
        tar -zxf WMT14.en-de.partial.tar.gz
    fi
    # Set soft link.
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    rm -f vocab_all.bpe.33712
    rm -f vocab_all.bpe.33708
    # Vocab
    cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
    cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
    # Train
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.en train.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.de train.de
    # Dev
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.en dev.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.de dev.de
    #Test
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/test.tok.bpe.en test.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/test.tok.bpe.de test.de
    cd -
elif [ ${MODE} = "whole_infer" ]; then
    cd ../examples/machine_translation/transformer/
    sed -i "s/^max_out_len.*/max_out_len: 256/g" configs/transformer.base.yaml
    sed -i "s/^batch_size.*/batch_size: 4096/g" configs/transformer.base.yaml
    sed -i "s/^max_out_len.*/max_out_len: 1024/g" configs/transformer.big.yaml
    sed -i "s/^batch_size.*/batch_size: 4096/g" configs/transformer.big.yaml

    sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.base.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.base.yaml
    sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.base.yaml

    sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.big.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.big.yaml
    sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.big.yaml

    # Trained transformer base model checkpoint. 
    # For infer. 
    if [ ! -f transformer-base-wmt_ende_bpe.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz
        tar -zxf transformer-base-wmt_ende_bpe.tar.gz
        mv base_trained_models/ trained_models/
    fi
    # For train. 
    if [ ! -f WMT14.en-de.partial.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.partial.tar.gz
        tar -zxf WMT14.en-de.partial.tar.gz
    fi
    # Whole data set prepared. 
    if [ ! -f WMT14.en-de.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz
        tar -zxf WMT14.en-de.tar.gz
    fi
    # Set soft link.
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    rm -f vocab_all.bpe.33712
    rm -f vocab_all.bpe.33708
    # Vocab
    cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
    cp -f WMT14.en-de.partial/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
    # Train with partial data. 
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.en train.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/train.tok.clean.bpe.de train.de
    # Dev with partial data. 
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.en dev.en
    ln -s WMT14.en-de.partial/wmt14_ende_data_bpe/dev.tok.bpe.de dev.de
    # Test with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
    cd -
elif [ ${MODE} = "whole_train_infer" ]; then
    cd ../examples/machine_translation/transformer/
    sed -i "s/^max_out_len.*/max_out_len: 256/g" configs/transformer.base.yaml
    sed -i "s/^batch_size.*/batch_size: 4096/g" configs/transformer.base.yaml
    sed -i "s/^max_out_len.*/max_out_len: 1024/g" configs/transformer.big.yaml
    sed -i "s/^batch_size.*/batch_size: 4096/g" configs/transformer.big.yaml

    sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.base.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.base.yaml
    sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.base.yaml

    sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.big.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.big.yaml
    sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.big.yaml

    # Whole data set prepared. 
    if [ ! -f WMT14.en-de.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz
        tar -zxf WMT14.en-de.tar.gz
    fi
    # Set soft link. 
    if [ -f train.en ]; then
        rm -f train.en
    fi
    if [ -f train.de ]; then
        rm -f train.de
    fi
    if [ -f dev.en ]; then
        rm -f dev.en
    fi
    if [ -f dev.de ]; then
        rm -f dev.de
    fi
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    rm -f vocab_all.bpe.33712
    rm -f vocab_all.bpe.33708
    # Vocab
    cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
    cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
    # Train with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/train.tok.clean.bpe.33708.en train.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/train.tok.clean.bpe.33708.de train.de
    # Dev with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2013.tok.bpe.33708.en dev.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2013.tok.bpe.33708.de dev.de
    # Test with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
    cd -
else # infer
    cd ../examples/machine_translation/transformer/
    sed -i "s/^max_out_len.*/max_out_len: 256/g" configs/transformer.base.yaml
    sed -i "s/^batch_size.*/batch_size: 4096/g" configs/transformer.base.yaml
    sed -i "s/^max_out_len.*/max_out_len: 1024/g" configs/transformer.big.yaml
    sed -i "s/^batch_size.*/batch_size: 4096/g" configs/transformer.big.yaml

    sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.base.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.base.yaml
    sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.base.yaml

    sed -i "s/^random_seed:.*/random_seed: None/g" configs/transformer.big.yaml
    sed -i "s/^shuffle_batch:.*/shuffle_batch: True/g" configs/transformer.big.yaml
    sed -i "s/^shuffle:.*/shuffle: True/g" configs/transformer.big.yaml

    # Trained transformer base model checkpoint. 
    if [ ! -f transformer-base-wmt_ende_bpe.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/models/transformers/transformer/transformer-base-wmt_ende_bpe.tar.gz
        tar -zxf transformer-base-wmt_ende_bpe.tar.gz
        mv base_trained_models/ trained_models/
    fi
    # Whole data set prepared. 
    if [ ! -f WMT14.en-de.tar.gz ]; then
        wget https://bj.bcebos.com/paddlenlp/datasets/WMT14.en-de.tar.gz
        tar -zxf WMT14.en-de.tar.gz
    fi
    # Set soft link.
    if [ -f test.en ]; then
        rm -f test.en
    fi
    if [ -f test.de ]; then
        rm -f test.de
    fi
    rm -f vocab_all.bpe.33712
    rm -f vocab_all.bpe.33708
    # Vocab
    cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33712 ./
    cp -f WMT14.en-de/wmt14_ende_data_bpe/vocab_all.bpe.33708 ./
    # Test with whole data. 
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.en test.en
    ln -s WMT14.en-de/wmt14_ende_data_bpe/newstest2014.tok.bpe.33708.de test.de
    cd -
fi
