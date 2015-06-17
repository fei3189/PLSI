#include <iostream>
#include <fstream>
#include <cstdlib>
#include "plsi.hpp"
using namespace std;


int main(int argc, char **argv) {
    if (argc < 4) {
        cerr << "usage: ./plsa train stopwords num_topics" << endl;
        exit(-1);
    }
    vector<string> docs, w_stop;
    string line;

    ifstream fin;
    fin.open(argv[1], ios_base::in);
    if (!fin) {
        cerr << "Train file: " << argv[1] << " does not exist" << endl;
        exit(-2);
    }
    getline(fin, line); // Ignore the first line
    for (; getline(fin, line); ) {
        docs.push_back(line);
    }
    fin.close();
    
    fin.open(argv[2], ios_base::in);
    if (!fin) {
        cerr << "Stop words file: " << argv[2] << " does not exist" << endl;
        exit(-3);
    }
    for (; getline(fin, line); ) {
        w_stop.push_back(line);
    }
    fin.close();

    PLSI plsa = PLSI::train(docs, atoi(argv[3]), w_stop);
    plsa.output();
    return 0;
}
