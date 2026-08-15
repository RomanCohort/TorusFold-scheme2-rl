@echo off
cd /d C:\Users\颜子壹\TorusFold-scheme2-rl
set CIRCBASE_FASTA=C:/Users/颜子壹/Documents/circbase_seqs.fa.gz
set DATA_OUT=C:/tmp/test_isrna/rhofold_data
C:\ana\envs\comfyui\python.exe -u generate_data_rhofold.py --n-workers 4 --n-samples 113539 --max-len 2000 --min-len 50 --n-anneal 300 --resume
