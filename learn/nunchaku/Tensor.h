#pragma once
// 精简版 Tensor.h — 仅提供 dispatch_utils.h 需要的 ScalarType 枚举
// 实际 torch 桥接在 nunchaku_bridge.cu 中

#include <cstddef>
#include <vector>
#include <cassert>

namespace nunchaku::kernels {

struct Tensor {
    enum ScalarType {
        FP32 = 0,
        FP16 = 1,
        BF16 = 2,
        INT8  = 3,
        INT32 = 4,
        INT64 = 5,
        INVALID_SCALAR_TYPE = -1,
    };

    ScalarType dtype() const { return dtype_; }
    bool valid() const { return data_ptr_ != nullptr; }
    int numel() const { return numel_; }

    template<typename T>
    T* data_ptr() const { return reinterpret_cast<T*>(data_ptr_); }

    // 占位: 实际不调用
    const int* shape = nullptr;
    int ndims() const { return 0; }

    // 内部
    void* data_ptr_ = nullptr;
    ScalarType dtype_ = INVALID_SCALAR_TYPE;
    int numel_ = 0;
    int shape_[8] = {0};
};

// 用于 dispatch 的 ScalarType 切换模板
template<typename F>
inline auto dispatchFloat(Tensor::ScalarType scalarType, F &&func) {
    switch (scalarType) {
    case Tensor::BF16: return func.template operator()<__nv_bfloat16>();
    case Tensor::FP16: return func.template operator()<half>();
    case Tensor::FP32: return func.template operator()<float>();
    default: assert(false); throw std::invalid_argument("not float");
    }
}

template<typename F>
inline auto dispatchBool(bool val, F &&func) {
    if (val) { func.template operator()<true>(); }
    else     { func.template operator()<false>(); }
}

}  // namespace nunchaku::kernels
