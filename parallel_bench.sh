echo Benchmarking With Parallelism With $2 Threads
echo Connecting to server $1
(time mpirun --allow-run-as-root -np $2 python3 -u train.py --env-name='FetchReach-v1' --server-name=$1 --n-epochs=10 --n-cycles=10 2>&1 | tee reach.log) 2> parallel_bench.txt

