#include <Python.h>
#include <vector>
#include <cmath>
#include <assert.h>
#include <stdexcept>
#include <iostream>
using namespace std;

//  *********** List to vectorvector ************
// PyObject -> Vector
vector<float> listTupleToVector_Float(PyObject* incoming) {
    vector<float> data;
    if (PyTuple_Check(incoming)) {
        for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
            PyObject *value = PyTuple_GetItem(incoming, i);
            data.push_back( PyFloat_AsDouble(value) );
        }
    } else {
        if (PyList_Check(incoming)) {
            for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
                PyObject *value = PyList_GetItem(incoming, i);
                data.push_back( PyFloat_AsDouble(value) );
            }
        } else {
            throw logic_error("Passed PyObject pointer was not a list or tuple!");
        }
    }
    return data;
}
vector<vector<float> > List2Vecvec(PyObject* l)
{
    vector<vector<float> > V;

    if (PyList_Check(l)) {
        float lengthL = PyList_Size(l);
        for(Py_ssize_t i = 0; i < lengthL; ++i)
        {
            PyObject* tempL = PyList_GetItem(l, i);
            vector<float> tempV;
            
            if (PyList_Check(tempL)) {
                float len = PyList_Size(tempL);
                for(Py_ssize_t j = 0; j < len; ++j)
                {
                    PyObject* temp = PyList_GetItem(tempL, j);
                    float t = PyFloat_AsDouble(temp);
                    tempV.push_back(t);
                }
                V.push_back(tempV);
            } else {
                // cout << "1:Passed PyObject pointer was not a list!" << endl;
                throw logic_error("1:Passed PyObject pointer was not a list!");
                }
        }
    } else 
    {
        throw logic_error("0:Passed PyObject pointer was not a list!");
    }

    return V;
}

// ************ vectorVcotor2List ***********
PyObject* vectorToList_Float(const vector<float> &data) {
  PyObject* listObj = PyList_New( data.size() );
    if (!listObj) throw logic_error("Unable to allocate memory for Python list");
    for (unsigned int i = 0; i < data.size(); i++) {
        PyObject *num = PyFloat_FromDouble(double(data[i]));
        if (!num) {
            Py_DECREF(listObj);
            throw logic_error("Unable to allocate memory for Python list");
        }
        PyList_SET_ITEM(listObj, i, num);
    }
    return listObj;
}
PyObject* vectorVectorToList_Float(const vector<vector<float> > &data) {
    PyObject* list = PyList_New( data.size() );
    if (!list) throw logic_error("Unable to allocate memory for Python list");
    for (unsigned int i = 0; i < data.size(); i++) {
        PyObject* subList = NULL;
        try {
            subList = vectorToList_Float(data[i]);
        } catch (logic_error &e) {
            throw e;
        }
        if (!subList) {
            Py_DECREF(list);
            throw logic_error("Unable to allocate memory for Python list of lists");
        }
        PyList_SET_ITEM(list, i, subList);
    }

    return list;
}

// ********** Cal Similaruty ************
float Modular(const vector<float>& vec){   //求向量的模长
        int n = vec.size();
        float sum = 0.0;
        for (int i = 0; i<n; ++i)
            sum += vec[i] * vec[i];
        return sqrt(sum);
}

float cosSimilarity(const vector<float>& lhs, const vector<float>& rhs){
    int n = lhs.size();
    assert(n == rhs.size());
    float tmp = 0.0;  //内积
    for (int i = 0; i<n; ++i)
        tmp += lhs[i] * rhs[i];
    return 1. - tmp / (Modular(lhs)*Modular(rhs));
}

vector<vector<float> > matCosSimilarity(const vector<vector<float> >& A, const vector<vector<float> >& B){
    int n = A.size();
    int m = B.size();
    vector<vector<float> > sim;
    for(int i = 0; i < n; ++i)
    {
        vector<float> temp;
        for(int j = 0; j < m; ++j)
        {
            temp.push_back(cosSimilarity(A[i], B[j]));
        }
        sim.push_back(temp);
    }
    return sim;
}

// ********** Wrapper function **********
static PyObject* matSimilarity(PyObject* self, PyObject* args)
{
    PyObject *listA, *listB;
    if (! PyArg_ParseTuple(args, "OO", &listA, &listB))
        return NULL;
    
    vector<vector<float> > A, B;
    A = List2Vecvec(listA);
    B = List2Vecvec(listB);

    vector<vector<float> > SimVV;
    SimVV = matCosSimilarity(A, B);
    
    PyObject* SimList;
    SimList = vectorVectorToList_Float(SimVV);

    return SimList;
}

static PyMethodDef matSimilarity_func[] = {
    {"cal", (PyCFunction)matSimilarity, METH_VARARGS, " "},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matSimModule = {
    PyModuleDef_HEAD_INIT,
    "matSimilarity", //name of module
    NULL, //module documentation
    -1,
    matSimilarity_func
};

PyMODINIT_FUNC PyInit_matSimilarity(void){
    return PyModule_Create(&matSimModule);
}