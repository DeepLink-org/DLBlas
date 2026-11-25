#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstdint>

// The device library is built by build.sh and uses this bridge ABI.
extern "C" void task03_run(
    const uint16_t* residual_bf16,
    const float* mhc_fn,
    float* out);

namespace {

constexpr Py_ssize_t kForwardArgs = 4;
constexpr double kEps = 1.0e-6;

struct RunnerState {
    at::Tensor out;
    PyObject* out_py = nullptr;
};

void free_runner(PyObject* capsule)
{
    void* raw = PyCapsule_GetPointer(capsule, "task03.native-forward.runner");
    if (!raw) {
        // PyCapsule_GetPointer sets an exception for a malformed capsule.  A
        // destructor must not leave that exception behind.
        PyErr_Clear();
        return;
    }
    auto* state = static_cast<RunnerState*>(raw);
    Py_XDECREF(state->out_py);
    delete state;
}

bool check_forward_contract(PyObject* const* args, Py_ssize_t nargs)
{
    if (nargs != kForwardArgs) {
        PyErr_Format(
            PyExc_TypeError,
            "forward expects residual, mhc_fn, normw, eps (got %zd arguments)",
            nargs);
        return false;
    }
    if (!THPVariable_Check(args[0]) || !THPVariable_Check(args[1])) {
        PyErr_SetString(
            PyExc_TypeError,
            "forward residual and mhc_fn must be torch.Tensor objects");
        return false;
    }
    if (args[2] != Py_None) {
        PyErr_SetString(
            PyExc_ValueError,
            "task03 native forward supports normw=None only");
        return false;
    }
    // The official caller supplies a Python float.  Avoid coercion or a
    // temporary allocation on the hot path while still checking the contract
    // on every call.
    if (!PyFloat_Check(args[3]) || PyFloat_AS_DOUBLE(args[3]) != kEps) {
        PyErr_SetString(
            PyExc_ValueError,
            "task03 native forward supports eps=1e-6 only");
        return false;
    }
    return true;
}

PyObject* runner_call(PyObject* capsule, PyObject* const* args, Py_ssize_t nargs)
{
    HANDLE_TH_ERRORS

    if (!check_forward_contract(args, nargs))
        return nullptr;

    auto* state = static_cast<RunnerState*>(PyCapsule_GetPointer(
        capsule, "task03.native-forward.runner"));
    if (!state)
        return nullptr;
    if (!state->out.defined() || state->out_py == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "native forward output state is invalid");
        return nullptr;
    }

    // Every invocation unpacks both current Tensor objects and reads their
    // current storage addresses.  Only the output Tensor object is retained;
    // no input Tensor, input pointer, answer, or derived value is cached.
    const at::Tensor& residual = THPVariable_Unpack(args[0]);
    const at::Tensor& mhc_fn = THPVariable_Unpack(args[1]);
    if (!residual.defined() || !mhc_fn.defined()) {
        PyErr_SetString(PyExc_ValueError, "forward input Tensors must be defined");
        return nullptr;
    }

    const uint16_t* residual_ptr =
        static_cast<const uint16_t*>(residual.data_ptr());
    const float* mhc_fn_ptr = static_cast<const float*>(mhc_fn.data_ptr());
    float* out_ptr = static_cast<float*>(state->out.data_ptr());

    // Keep the returned Tensor alive while an asynchronous device launch is
    // in flight.  The launch ABI is non-Python code, so the GIL can be released
    // for parity with ctypes.CDLL; every exceptional path restores it before
    // touching Python objects or setting a Python exception.
    Py_INCREF(state->out_py);
    PyThreadState* thread_state = PyEval_SaveThread();
    try {
        task03_run(residual_ptr, mhc_fn_ptr, out_ptr);
    } catch (...) {
        PyEval_RestoreThread(thread_state);
        Py_DECREF(state->out_py);
        PyErr_SetString(PyExc_RuntimeError, "task03_run raised a C++ exception");
        return nullptr;
    }
    PyEval_RestoreThread(thread_state);
    return state->out_py;

    END_HANDLE_TH_ERRORS
}

PyObject* make_runner(PyObject*, PyObject* output)
{
    HANDLE_TH_ERRORS

    if (!THPVariable_Check(output)) {
        PyErr_SetString(PyExc_TypeError, "output must be a torch.Tensor");
        return nullptr;
    }
    const at::Tensor& out = THPVariable_Unpack(output);
    if (!out.defined()) {
        PyErr_SetString(PyExc_ValueError, "output Tensor must be defined");
        return nullptr;
    }

    auto* state = new RunnerState{out, output};
    Py_INCREF(output);
    PyObject* capsule = PyCapsule_New(
        state, "task03.native-forward.runner", free_runner);
    if (!capsule) {
        Py_DECREF(output);
        delete state;
        return nullptr;
    }

    static PyMethodDef runner_method = {
        "forward",
        reinterpret_cast<PyCFunction>(runner_call),
        METH_FASTCALL,
        "Run the Task03 device library.",
    };
    PyObject* callable = PyCFunction_NewEx(&runner_method, capsule, nullptr);
    Py_DECREF(capsule);
    return callable;

    END_HANDLE_TH_ERRORS
}

PyMethodDef methods[] = {
    {"make_runner", make_runner, METH_O, "Create a stateful native forward runner."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "task03_native_forward",
    "Task03 native CPython forward bridge.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit_task03_native_forward()
{
    return PyModule_Create(&module);
}
