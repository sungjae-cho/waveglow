cd ../..

#python distributed.py -c config.json --prj_name waveglow_ko --run_name ketts-ketts2-kss-nc-kad_pretrained_waveglow_rm-silence --visible_gpus 1,2,3

python distributed.py -c config.json --prj_name waveglow_ko --run_name ketts-ketts2-kss-nc-kad_pretrained_waveglow_rm-silence_reduced-lr --visible_gpus 4,5,6
