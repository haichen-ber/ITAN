while true
do
  # 改动项1 relation_train_net 这里改成你命令中包含的字符串
  # 比如我要运行的代码如下：
  # CUDA_VISIBLE_DEVICES=0,1 python ... --nproc_per_node=2 ./relation_train_net.py ...
  # 这个主要是控制不要同时占用太多显卡，免得被打
  count=$(ps -ef | grep python | grep -c baselines/attack_scripts/Untarget_arpre_attack)
  if [ $count -lt 6 ]  # 改动项2， 根据之前被占用的显卡数调整
    then
     # 改动项3 查询第1块gpu的容量2p 第2块3p  第3块4p  第四块5p 依次类推
     # 这里我有4张卡，获取四块GPU的显存占用情况
     stat0=$(gpustat | awk '{print $11}' | sed -n '0p')
     stat1=$(gpustat | awk '{print $11}' | sed -n '1p')
     stat2=$(gpustat | awk '{print $11}' | sed -n '2p')
     stat3=$(gpustat | awk '{print $11}' | sed -n '3p')
     stat4=$(gpustat | awk '{print $11}' | sed -n '4p')
     stat5=$(gpustat | awk '{print $11}' | sed -n '5p')
     stat6=$(gpustat | awk '{print $11}' | sed -n '6p')
     stat7=$(gpustat | awk '{print $11}' | sed -n '7p')
     stat_arr=($stat0 $stat1 $stat2 $stat3 $stat4 $stat5, $stat6, $stat7)
     gpu_available=0
     gpu_available_index_arr=()
     # 得到空闲GPU的数量和对应的序号
     for i in ${!stat_arr[@]}
     do
       # 如果显存占用小于100M，继续
       if [ "${stat_arr[$i]}" -lt 10000 ]
       then
         gpu_available=$[gpu_available+1]
         gpu_available_index_arr[${#gpu_available_index_arr[@]}]=$i
       fi
     done
     echo '-可用GPU数:'$gpu_available', 第'${gpu_available_index_arr[@]}'块GPU可用'
     # 如果GPU数大于指定数量，取指定数量GPU开始训练
     if [ $gpu_available -ge 2 ]
     then
       echo 'start running my code...'
       # 传值操作，这是我们上面创建的另一个shell脚本
       sh ./run.sh ${gpu_available_index_arr[1]} ${gpu_available_index_arr[3]} ${gpu_available_index_arr[4]} ${gpu_available_index_arr[5]} ${gpu_available_index_arr[7]}
       break # 防止下一次循环又重复运行上一行命令
     fi
  fi
  sleep 10
done
