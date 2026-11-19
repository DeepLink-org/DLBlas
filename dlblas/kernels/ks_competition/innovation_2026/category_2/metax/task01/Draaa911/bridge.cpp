#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstdint>

extern "C" void engram_run_fixed(
    int32_t* tokens,
    int64_t* mults,
    int32_t* vsizes,
    int32_t* offs,
    int32_t* out);

namespace {

struct RunnerState {
    at::Tensor out;
    PyObject* out_py;
};

void free_runner(PyObject* capsule)
{
    void* raw = PyCapsule_GetPointer(capsule, "task01.native_forward.runner");
    if (!raw) {
        PyErr_Clear();
        return;
    }
    auto* state = static_cast<RunnerState*>(raw);
    Py_DECREF(state->out_py);
    delete state;
}

PyObject* runner_call(PyObject* capsule, PyObject* const* args, Py_ssize_t nargs)
{
    HANDLE_TH_ERRORS
    if (nargs != 4) {
        PyErr_Format(PyExc_TypeError, "forward expects 4 Tensor arguments (got %zd)", nargs);
        return nullptr;
    }
    for (Py_ssize_t i = 0; i < nargs; ++i) {
        if (!THPVariable_Check(args[i])) {
            PyErr_Format(PyExc_TypeError, "forward argument %zd must be a torch.Tensor", i);
            return nullptr;
        }
    }

    auto* state = static_cast<RunnerState*>(
        PyCapsule_GetPointer(capsule, "task01.native_forward.runner"));
    if (!state)
        return nullptr;

    const at::Tensor& tokens = THPVariable_Unpack(args[0]);
    const at::Tensor& mults = THPVariable_Unpack(args[1]);
    const at::Tensor& vsizes = THPVariable_Unpack(args[2]);
    const at::Tensor& offs = THPVariable_Unpack(args[3]);
    if (!tokens.defined() || !mults.defined() || !vsizes.defined() ||
        !offs.defined() || !state->out.defined()) {
        PyErr_SetString(PyExc_ValueError, "all Tensor arguments must be defined");
        return nullptr;
    }

    int32_t* tokens_ptr = static_cast<int32_t*>(tokens.data_ptr());
    int64_t* mults_ptr = static_cast<int64_t*>(mults.data_ptr());
    int32_t* vsizes_ptr = static_cast<int32_t*>(vsizes.data_ptr());
    int32_t* offs_ptr = static_cast<int32_t*>(offs.data_ptr());
    int32_t* out_ptr = static_cast<int32_t*>(state->out.data_ptr());

    PyThreadState* thread_state = PyEval_SaveThread();
    engram_run_fixed(tokens_ptr, mults_ptr, vsizes_ptr, offs_ptr, out_ptr);
    PyEval_RestoreThread(thread_state);

    Py_INCREF(state->out_py);
    return state->out_py;
    END_HANDLE_TH_ERRORS
}

PyObject* make_runner(PyObject*, PyObject* out_py)
{
    HANDLE_TH_ERRORS
    if (!THPVariable_Check(out_py)) {
        PyErr_SetString(PyExc_TypeError, "output must be a torch.Tensor");
        return nullptr;
    }
    const at::Tensor& out = THPVariable_Unpack(out_py);
    if (!out.defined()) {
        PyErr_SetString(PyExc_ValueError, "output Tensor must be defined");
        return nullptr;
    }

    auto* state = new RunnerState{out, out_py};
    Py_INCREF(out_py);
    PyObject* capsule = PyCapsule_New(
        state, "task01.native_forward.runner", free_runner);
    if (!capsule) {
        Py_DECREF(out_py);
        delete state;
        return nullptr;
    }
    static PyMethodDef method = {
        "forward", reinterpret_cast<PyCFunction>(runner_call), METH_FASTCALL, nullptr};
    PyObject* callable = PyCFunction_NewEx(&method, capsule, nullptr);
    Py_DECREF(capsule);
    return callable;
    END_HANDLE_TH_ERRORS
}

PyMethodDef methods[] = {
    {"make_runner", make_runner, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "engram_bridge_native", nullptr, -1, methods};

}  // namespace

PyMODINIT_FUNC PyInit_engram_bridge_native() { return PyModule_Create(&module); }
