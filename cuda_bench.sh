echo Benchmarking With Cuda
echo Connecting to server $1
(time python3 -u train.py --cuda --env-name='FetchReach-v1' --server-name=$1 --n-epochs=10 --n-cycles=10) 2> cuda_bench.txt

