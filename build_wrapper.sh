g++ -shared -Wl,-soname,wrapper -o cpp/wrapper.so -fPIC cpp/wrapper.cpp -I /usr/include/x86_64-linux-gnu -lopenblas
