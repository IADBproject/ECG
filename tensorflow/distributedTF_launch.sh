#!/bin/bash
#################################################################################
## Launcher: 
## Declare the execution steps for the application, libraries and the monitor
#################################################################################

## Application workload parameters
APP="distri-tf"
NUM_HOST=3

## Location of the power consumption tracks 
Dir=../enerGyPU/testbed/
DATA=`date +%Y%m%d%H%M`
ARGV=$APP-$NUM_HOST-$DATA
mkdir $Dir/$ARGV
## Global parameters for distributed computing
Dir_remote=cloud/ECG/enerGyPU/testbed/

## Read the ip-host list for cluster
c=0; for line in `cat h-workers  | awk '{ print $1 }'`; do
       IP_hosts[$c]=$line
       ((c++))
done

c=0; for line in `cat h-workers  | awk '{ print $2 }'`; do
       SP_hosts[$c]=$line
       ((c++))
done

c=0; for line in `cat h-workers  | awk '{ print $3 }'`; do
       IN_hosts[$c]=$line
       ((c++))
done


## Turn-on the computational resources monitor on distributed platform
for ip_host in "${IP_hosts[@]}"; do
	echo $ip_host
	ssh   mpiuser@$ip_host 'bash -s' < /home/mpiuser/cloud/ECG/enerGyPU/dataCapture/enerGyPU_record-jetson.sh $Dir_remote $ARGV &
        ssh   mpiuser@$ip_host 'bash -s' < /home/mpiuser/cloud/ECG/enerGyPU/dataCapture/enerGyPU_bandwidth.sh $Dir_remote $ARGV ${IP_hosts[0]} &
       
done

## Turn-on the computational resources monitor on local machine
#./enerGyPU/dataCapture/enerGyPU_record-jetson.sh $Dir $ARGV &


echo "--- Launched ---"
#sleep 40s
## Aplication execution 

mkdir ../input/train
mkdir ../input/test
mkdir ../input/val

python3.6 ../Keras/splitdata.py $NUM_HOST

c=1;for ip_host in "${IP_hosts[@]:1}"; do
        ssh mpiuser@$ip_host  "cd /home/mpiuser/cloud/ECG/tensorflow/  && python3.6 distributed_execution.py"  ${SP_hosts[c]} ${IN_hosts[c]} $Dir $ARGV &
        ((c++))    
done

python3.6 distributed_execution.py ${SP_hosts[0]} ${IN_hosts[0]} $Dir $ARGV 

## Move memroy profiler to experiment tracks
#mv output $Dir/$ARGV/

rm -r  ../input/train
rm -r  ../input/test
rm -r  ../input/val


echo "--- Finished  ---"


## Turn-off the computational resources monitor for each host
for ip_host in "${IP_hosts[@]}"; do
	ssh -t mpiuser@$ip_host "sudo killall -9 tegrastats"
	ssh -t mpiuser@$ip_host "pkill -f 'grep'"
	ssh -t mpiuser@$ip_host "pkill -f 'bash -s'"
done

echo "Experiment directory: '$Dir/$ARGV'"

#exit
kill %1
