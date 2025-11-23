cd /data2/tianang/projects/Synergy/DataPrepare/MDLM
NUM_CORES=108
echo "NUM_CORES: ${NUM_CORES}"
python tokenize_SELFIES_descriptors_hf.py -s "120" -c $NUM_CORES
python tokenize_SELFIES_descriptors_hf.py -s "119" -c $NUM_CORES
python tokenize_SELFIES_descriptors_hf.py -s "118" -c $NUM_CORES


