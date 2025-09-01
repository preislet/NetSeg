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


cp $HOMEDIR/download_data.py $SCRATCHDIR
cp $HOMEDIR/preprocessing.py $SCRATCHDIR
cp $HOMEDIR/Model_training.py $SCRATCHDIR
cp $HOMEDIR/test_inference.py $SCRATCHDIR
cp $HOMEDIR/config.py $SCRATCHDIR
cp -r $HOMEDIR/utils $SCRATCHDIR


# Instalace potřebných knihoven
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy tqdm scikit-image requests

python download_data.py 
python preprocessing.py
echo "Training with U-Net model"
python Model_training.py --model unet
echo "Training with ResUNet model"
python Model_training.py --model resunet

# Kopírování výsledků zpět do HOMEDIR (model, logy, výstupy)
cp -r checkpoints $HOMEDIR/
