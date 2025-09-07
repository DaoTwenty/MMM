module load gcc
module load cuda torch

rm -rf build/
mkdir build && cd build
cmake ..
make -j
./causallm
