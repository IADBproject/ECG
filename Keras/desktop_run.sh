#!/bin/bash

###########################################################################
#                                                                         #
# enerGyPU for monitoring performance and power consumption on Multi-GPU  #
#                                                                         #
###########################################################################

# enerGyPU_run.sh
# Execution steps and the application libraries are declared.
###########################################################################


# Global workload parameters
# For the sample matrixMul the dimensions of A & B matrices must be equal.
DIM=2048
nGPU=1

# Location of the power consumption measures
Dir=./../enerGyPU/testbed/
HOST=$(hostname)
APP="keras"
DATA=`date +%Y%m%d%H%M`
ARGV=$HOST-$APP-$DIM-$nGPU-$DATA
mkdir $Dir/$ARGV
mkdir output

# Executes the enerGyPU_record.sh
./../enerGyPU/dataCapture/enerGyPU_record.sh $Dir $ARGV &

# Add path of the application and libraries necessaries
python3.6 desktop.py

mv output $Dir/$ARGV/

kill %1
