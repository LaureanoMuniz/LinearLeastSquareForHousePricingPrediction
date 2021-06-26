mkdir build
cd build
rm -rf *
cmake -DPYTHON_EXECUTABLE="$(which python)" -DCMAKE_BUILD_TYPE=Release ..
make install
cd ..
cp build/metnum.cpython-38-x86_64-linux-gnu.so experiments/
