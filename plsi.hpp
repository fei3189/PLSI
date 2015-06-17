#ifndef _PLSI_HPP_
#define _PLSI_HPP_

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <set>
#include <cmath>
#include <cstdlib>
#include <ctime>
using namespace std;

class PLSI {
    struct Index {
        int idx;
        int value;
        Index(int i, int v): idx(i), value(v) {}
    };
private:
    vector<vector<float> > p_wz;
    vector<vector<float> > p_dz;
    vector<float> p_z;
    vector<vector<Index> > n_wd;
    int c_word;
    int c_doc;
    int k_z;
    map<string, int> word2id;
    map<int, string> id2word;
    
    int _all_words;

private:
    PLSI() { 
        srand((unsigned)time(0));
        _all_words = 0;
    }
    
    void _split(vector<string> &tokens, const string &str) {
        istringstream sin(str);
        string t;
        while (sin >> t)
            tokens.push_back(t);
    }

    void _resize1d(vector<float> &array, int d) {
        array.resize(d, 0);
        float sum = 0;
        for (int i = 0; i < array.size(); ++i) {
            array[i] = ((float)rand()) / RAND_MAX;
            sum += array[i];
        }

        for (int i = 0; i < array.size(); ++i) {
            array[i] /= sum;
        }
    }

    void _resize2d(vector<vector<float> > &array, int d1, int d2) {
        array.resize(d1);
        for (int i = 0; i < d1; ++i) {
            _resize1d(array[i], d2);
        }
    }

    void _resize2d_n(vector<vector<int> > &array, int d1, int d2) {
        array.resize(d1);
        for (int i = 0; i < d1; ++i)
            array[i].resize(d2, 0);
    }
    
    void _preprocess(const vector<string> &docs, int k, const vector<string> &_w_stop) {
        set<string> w_stop;
        n_wd.resize(docs.size());
        vector<vector<int> > _tmp_n_wd;
        for (int i = 0; i < _w_stop.size(); ++i) {
            w_stop.insert(_w_stop[i]);
        }
        for (int i = 0; i < docs.size(); ++i) {
            vector<string> tokens;
            _split(tokens, docs[i]);
            map<int, int> id2freq;
            for (int j = 0; j < tokens.size(); ++j) {
                if (w_stop.find(tokens[j]) != w_stop.end()) {
                    continue;
                }
                if (word2id.find(tokens[j]) == word2id.end()) {
                    int size = word2id.size();
                    word2id[tokens[j]] = size;
                    id2word[size] = tokens[j];
                }
                int id = word2id[tokens[j]];
                id2freq[id]++;
                _all_words++;
            }
            for (map<int, int>::iterator it = id2freq.begin(); it != id2freq.end(); ++it)
                n_wd[i].push_back(Index(it->first, it->second));
        }

        c_doc = docs.size();
        c_word = word2id.size();
        k_z = k;
        _resize1d(p_z, k);
        _resize2d(p_wz, c_word, k);
        _resize2d(p_dz, c_doc, k);
    }

    float _likelihood() {
        float lh = 0;
        for (int i = 0; i < c_doc; ++i) {
            const vector<Index> &ns = n_wd[i];
            const int size = ns.size();
            for (int j = 0; j < size; ++j) {
                float n = ns[j].value;
                float z = 0;
                for (int k = 0; k < k_z; ++k) {
                    z += p_wz[ns[j].idx][k] * p_dz[i][k] * p_z[k];
                }
                lh += n * log(z);
            }
        }
        return lh;
    }

    void _train(const vector<string> &docs, int z_k, const vector<string> &w_stop) {
        float eps = 0.01;
        float last = 0, current = -1000;
        int iter_num = 0;
        vector<float> p_zwd(z_k, 0);

        _preprocess(docs, z_k, w_stop);
        clock_t t = clock();
        while (fabs(current - last) > eps) {
            ++iter_num;
            vector<vector<float> > p_wz_current = p_wz;
            vector<vector<float> > p_dz_current = p_dz;
            vector<float> p_z_current = p_z;
            for (int i = 0; i < c_doc; ++i)
                for (int k = 0; k < z_k; ++k)
                    p_dz[i][k] = 0;
            for (int i = 0; i < c_word; ++i)
                for (int k = 0; k < z_k; ++k)
                    p_wz[i][k] = 0;
            for (int k = 0; k < z_k; ++k)
                p_z[k] = 0;
            for (int i = 0; i < c_doc; ++i) {
                for (int _j = 0; _j < n_wd[i].size(); ++_j) {
                    Index adj = n_wd[i][_j];
                    int j = adj.idx;
                    int v = adj.value;
                    float sum_zwd = 0;
                    for (int k = 0; k < z_k; ++k) {
                        float tmp = p_wz_current[j][k] * p_dz_current[i][k] * p_z_current[k];
                        p_zwd[k] = tmp;
                        sum_zwd += tmp;
                    }
                    for (int k = 0; k < z_k; ++k)
                        p_zwd[k] /= sum_zwd;
                    for (int k = 0; k < z_k; ++k) {
                        p_wz[j][k] += v * p_zwd[k];
                        p_dz[i][k] += v * p_zwd[k];
                        p_z[k] += v * p_zwd[k];
                    }
                }
            }

            float sum_z = 0;
            for (int k = 0; k < z_k; ++k) {
                float sum_w = 0, sum_d = 0;
                for (int i = 0; i < c_word; ++i)
                    sum_w += p_wz[i][k];
                for (int i = 0; i < c_word; ++i)
                    p_wz[i][k] /= sum_w;
                
                for (int i = 0; i < c_doc; ++i)
                    sum_d += p_dz[i][k];
                for (int i = 0; i < c_doc; ++i)
                    p_dz[i][k] /= sum_d;
                sum_z += p_z[k];
            }
            for (int k = 0; k < z_k; ++k)
                p_z[k] /= _all_words;

            last = current;
            current = _likelihood();
        }
        cout << "Train finished. " << endl;
        cout << "Likelihood = " << current << endl;
        cout << "eps = " << eps << ", iteration times = " << iter_num << endl;
        cout << "Time spent " << 1.0 * (clock() - t) / CLOCKS_PER_SEC << "s" << endl;
        cout << "Number of topics = " << z_k << endl;
        cout << endl;
    }
public:
    static PLSI train(const vector<string> &docs, int k, const vector<string> &w_stop = vector<string>()) {
        PLSI plsa;
        plsa._train(docs, k, w_stop);
        return plsa;
    }

    void output() {
        vector<vector<float> > _p_wz = this->p_wz;
        for (int k = 0; k < k_z; ++k) {
            cout << "********************* topic " << k << endl;
            for (int i = 0; i < 10; ++i) {
                int idx = -1;
                float max = 0;
                for (int j = 0; j < c_word; ++j) {
                    if (_p_wz[j][k] > max) {
                        max = _p_wz[j][k];
                        idx = j;
                    }
                }
                cout << id2word[idx] << " " << max << endl;
                if (idx == -1)
                    break;
                _p_wz[idx][k] = 0;
            }
        }
    }
};

#endif
