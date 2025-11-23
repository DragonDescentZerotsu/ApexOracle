cd /data1/tianang/Projects/Synergy/DataPrepare/MDLM
NUM_CORES=128
echo "NUM_CORES: ${NUM_CORES}"
python tokenize_SELFIES_descriptors_hf.py -s "115" -c $NUM_CORES
python tokenize_SELFIES_descriptors_hf.py -s "116" -c $NUM_CORES
python tokenize_SELFIES_descriptors_hf.py -s "117" -c $NUM_CORES