#!/bin/bash
#################################################################################
## Launcher: 
## Declare the execution steps for the application, libraries and the monitor
#################################################################################

## Application workload parameters
APP="cnn"
MPI_NPROC=12

## Location of the power consumption tracks 
Dir=../enerGyPU/testbed/
DATA=`date +%Y%m%d%H%M`
ARGV=$APP-$MPI_NPROC-$DATA
mkdir $Dir/$ARGV
mkdir output
mkdir output/worker
## Global parameters for distributed computing
Dir_remote=cloud/ECG/enerGyPU/testbed/
#IP_hosts=("134.59.132.111" "134.59.132.116" "134.59.132.23")

c=0
for line in `cat h-workers  | awk '{ print $1 }'`
do
       IP_hosts[$c]=$line
       ((c++))
done


## Turn-on the computational resources monitor on distributed platform
for ip_host in "${IP_hosts[@]}"; do
	ssh  mpiuser@$ip_host 'bash -s' < /home/mpiuser/cloud/ECG/enerGyPU/dataCapture/enerGyPU_record-jetson.sh $Dir_remote $ARGV &
        ssh  mpiuser@$ip_host 'bash -s' < /home/mpiuser/cloud/ECG/enerGyPU/dataCapture/enerGyPU_bandwidth.sh $Dir_remote $ARGV ${IP_hosts[0]} &
       
done


## Turn-on the computational resources monitor on local machine
#./enerGyPU/dataCapture/enerGyPU_record-jetson.sh $Dir $ARGV &


#sleep 40s
echo "--- Launched ---"

#sudo nohup iftop -t > $Dir/$ARGV/iflog.txt & 

## Aplication execution 
#mpiexec -n 12 --hostfile h-workers python3.6  mpirun.py $Dir $ARGV
#mpiexec -n 12 --hostfile h-workers python3.6 -m memory_profiler  mpirun.py $Dir $ARGV
#mpiexec -n 3 python3.6 mpirun.py $Dir $ARGV

## Move memroy profiler to experiment tracks
mv output $Dir/$ARGV/

echo "--- Finished  ---"


## Turn-off the computational resources monitor for each host
for ip_host in "${IP_hosts[@]}"; do
	ssh -t mpiuser@$ip_host "sudo killall -9 tegrastats"
	ssh -t mpiuser@$ip_host "pkill -f 'grep'"
	ssh -t mpiuser@$ip_host "pkill -f 'bash -s'"
done

#exit
#sudo killall -9 iftop
kill %1
