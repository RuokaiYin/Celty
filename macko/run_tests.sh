bits_per_value=$1
[[ -z "$bits_per_value" ]] && { echo "Parameter bits_per_value is empty" ; exit 1; }
echo "Running with ${bits_per_value} bits per value"
algos=$2
[[ -z "$algos" ]] && { echo "Parameter algos is empty" ; exit 1; }
echo "Benchmarking algorithms: ${algos}"


gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n1 | sed -E 's/.*: (.*)/\1/' | tr ' ' '_')

base_results=results_${bits_per_value}bit/${gpu_name}

mkdir -p ${base_results}/partial
complete_result_file="${base_results}/results.txt"

echo ${gpu_name} | tee ${complete_result_file}

Rs=(4096 4096 5120 5120 8192 8192)
Cs=(4096 11008 5120 13824 8192 28672)


for ((i=0; i<${#Rs[@]}; i++)); do
    R=${Rs[i]}
    C=${Cs[i]}

    for algo in ${algos};
    do
        for d in 0.7 0.5 0.3;
        do
            partial_result_file=${base_results}/partial/${R}_${C}_${algo}_${d}.txt
            ./benchmark 16 $R $C 100 1000 1 0 $algo 47 $d > $partial_result_file
            if grep -q "Final results:" "$partial_result_file"; then
                grep "Final results:" "$partial_result_file"
            else
                # Probably OOM
                echo "Final results:	$bits_per_value	$R	$C	$algo	$d	 	 	 "
            fi

        done
    done
done | tee -a ${complete_result_file}
