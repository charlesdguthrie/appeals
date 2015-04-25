#!/bin/bash

## A qsub job that builds the ML dataset

#PBS -l nodes=1:ppn=1
#PBS -l walltime=1:00:00
#PBS -l mem=8GB
#PBS -N build_opinion_data
#PBS -j oe
#PBS -M alex.pine@nyu.edu

module purge

cd /home/akp258/appeals

ipython scripts/join_data.py
