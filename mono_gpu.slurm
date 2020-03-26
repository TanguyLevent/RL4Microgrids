#!/bin/bash
#SBATCH --job-name=gpu_mono         # nom du job
#SBATCH --ntasks=1                  # nombre de tâche (un unique processus ici)
#SBATCH --gres=gpu:1                # nombre de GPU à réserver (un unique GPU ici)
#SBATCH --qos=qos_gpu-t4            #specifier le qos à executer
#SBATCH --cpus-per-task=10          # nombre de coeurs à réserver (un quart du noeud)
# /!\ Attention, la ligne suivante est trompeuse mais dans le vocabulaire
# de Slurm "multithread" fait bien référence à l'hyperthreading.
#SBATCH --hint=nomultithread        # on réserve des coeurs physiques et non logiques
#SBATCH --time=60:00:00             # temps exécution maximum demande (HH:MM:SS)
#SBATCH --output=gpu_mono%A_%a.out     # nom du fichier de sortie
#SBATCH --error=gpu_mono%A_%a.out      # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0 ###-1
#######%5 c'est pour la ligne du dessus 
# echo des commandes lancées
set -x

# on se place dans le répertoire de soumission
cd ${SLURM_SUBMIT_DIR}

#nettoyage

module purge

#chargement des modules

module load python/2.7.16
module load tensorflow-gpu/py3/2.0.0-beta1
module load pandas

# exécution du code
srun python Main_X.py
###mv *.out  ${SLURM_SUBMIT_DIR}/results_gpu/
###mv *.csv  ${SLURM_SUBMIT_DIR}/results/
