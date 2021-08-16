cd ../..

python distributed.py -c checkpoints/waveglow_ko/ketts-ketts2-kss-nc_tf_mel_pretrained/config.json --prj_name waveglow_ko --run_name ketts-ketts2-kss-nc_tf_mel_pretrained --visible_gpus 1,2,3
