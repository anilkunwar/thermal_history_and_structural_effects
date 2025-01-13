#!/bin/bash

a=4.0486
#########FCC structure#######################################
atomsk --create fcc $a Al aluminum_unit.xsf cfg
atomsk aluminum_unit.xsf -duplicate 3 3 3 aluminum_super.xsf cfg
atomsk aluminum_super.xsf -select random 1 Al -substitute Al Mg almg_super.xsf cfg
atomsk almg_super.xsf -select random 1 Al -substitute Al Zr almgzr_super.xsf cfg
atomsk almgzr_super.xsf -select random 1 Al -substitute Al Si almgzrsi_super.xsf cfg
#atomsk almgzr_super.xsf -select random 2.0% Al  -substitute Al Fe almg1zr1fe2_super.xsf cfg
#atomsk almgzrsi_super.xsf lammps
##########Tetragonal Al3Zr##################################################
atomsk Al3Zr.cif -duplicate 3 3 1 al3zr_super.xsf cfg
atomsk al3zr_super.xsf -rotate Z 10 al3zr_super_rotZ10.xsf cfg
atomsk al3zr_super_rotZ10.xsf -rotate Y 9 al3zr_super_rotZ10Y9.xsf cfg
##########Merge without rotation############################################
atomsk --merge X 2 almgzrsi_super.xsf al3zr_super.xsf bimodal_unrotated.xsf cfg
atomsk --merge X 2 almgzrsi_super.xsf al3zr_super_rotZ10.xsf bimodal_10degreez.xsf cfg
atomsk --merge X 2 almgzrsi_super.xsf al3zr_super_rotZ10Y9.xsf bimodal_10degreez9y.xsf cfg
