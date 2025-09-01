#!/bin/bash
#PBS -N netseg
#PBS -l select=1:ncpus=16:ngpus=1:mem=32gb:gpu_mem=16gb:scratch_local=128gb
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -m abe
#PBS -M preislet@natur.cuni.cz

# Cesta k vašemu domovskému adresáři na MetaCentru
HOMEDIR="/storage/plzen1/home/preislet/NetSeg"

echo "Scratch directory: $SCRATCHDIR"
# Načtení modulu s Pythonem
module add python/3.11.11-gcc-10.2.1-555dlyc

# Přesun do scratch adresáře
export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR || exit 1
# Vytvoření virtuálního prostředí
python -m venv venv
source venv/bin/activate


cp $HOMEDIR/download_data.py .
cp $HOMEDIR/preprocessing.py .
cp $HOMEDIR/Model_training.py .
cp $HOMEDIR/test_inference.py .
cp $HOMEDIR/config.py .
cp -r $HOMEDIR/utils .


# Instalace potřebných knihoven
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy tqdm scikit-image requests

python download_data.py 
python preprocessing.py
python Model_training.py --threads 16 --epochs 20 --batch_size 10

# Kopírování výsledků zpět do HOMEDIR (model, logy, výstupy)
rsync -av --exclude='venv' --exclude='__pycache__' $SCRATCHDIR/ $HOMEDIR/
