cnt=0

while (( "${cnt}" < 1 )); do
    python3 train.py --verbose=False
    python3 train.py --gpu=True --verbose=False
    python3 train.py --cpp=True --verbose=False
    python3 train.py --cpp=True --gpu=True --verbose=False
    python3 train.py --cuda=True --gpu=True --verbose=False
    (( cnt = "${cnt}" + 1 ))
done