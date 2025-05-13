ssh ag52peve@login01/02.sc.uni-leipzig.de
dev ordner machen mkdir dev
ssh keygen machen
in github den noch hinzufügen
ordner mit git machen (mathml, git clone)
scripts ordner vonb julian 
vim rp.sh

clara hat v100, paula a30

im ordner einmal:
module load Python/3.12
dann erst python -m venv .venv

danach:
ausfürhrne mit: sbatch scripts/rp.sh

job id steht da

squeue -u ag52peve
PD = pending
R = running
