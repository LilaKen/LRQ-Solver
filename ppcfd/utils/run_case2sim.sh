#!/bin/bash

export CDLMD_LICENSE_FILE=/raid/ppcfd/Siemens/18.04.008/license.dat

/raid/ppcfd/Siemens/18.04.008/STAR-CCM+18.04.008/star/bin/starccm+ \
    -np 4 /raid/test/data_trpnoz/ES39_DG2_P15_R01_K65_New_Data_P15_KW_20241205@06000.sim\
    -batch ./sim_to_case_0222.java
