g++ -shared -Wl,-soname,wrapper -o wrapper.so -fPIC wrapper.cpp -I /usr/include/x86_64-linux-gnu -lopenblas
