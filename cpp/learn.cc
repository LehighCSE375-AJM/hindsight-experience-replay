#include <iostream>
#include "linear.h"
#include "matrix.h"
#include "models.h"

using namespace std;

int main() {    
    Matrix *in = new Matrix(1, 2, [](int row, int column){return 1;});
    cout << "IN" << endl << *in;
    Actor a = Actor();
    Matrix *out = a.forward(in);
    cout << "OUT\n" << *out;
    return 0;
}