// PyBind11 绑定: nunchaku GEMM + Attention kernel → Python
#include <torch/extension.h>

namespace py = pybind11;

// 前向声明
namespace nunchaku_bridge {
void gemm_w4a4(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
               torch::Tensor ascales, torch::Tensor wscales, torch::Tensor bias);
void gemm_w4a4_lora(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
                    torch::Tensor ascales, torch::Tensor wscales,
                    torch::Tensor lora_act_out, torch::Tensor lora_up,
                    torch::Tensor bias);
void gemm_w8a8(torch::Tensor act, torch::Tensor wgt, torch::Tensor out,
               torch::Tensor ascales, torch::Tensor wscales, torch::Tensor bias);
void attention_fp16(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                    torch::Tensor o, float scale);
void quantize_w4a4_act(torch::Tensor input, torch::Tensor output, torch::Tensor oscales);
void quantize_w4a4_wgt(torch::Tensor input, torch::Tensor output, torch::Tensor oscales);
void quantize_w8a8_act(torch::Tensor input, torch::Tensor output, torch::Tensor oscales, bool fuse_glu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm_w4a4", &nunchaku_bridge::gemm_w4a4,
          "W4A4 GEMM: INT4 weights x INT4 activations -> FP16/BF16",
          py::arg("act"), py::arg("wgt"), py::arg("out"),
          py::arg("ascales"), py::arg("wscales"),
          py::arg("bias") = torch::Tensor());

    m.def("gemm_w4a4_lora", &nunchaku_bridge::gemm_w4a4_lora,
          "W4A4 GEMM with fused LoRA-up epilogue",
          py::arg("act"), py::arg("wgt"), py::arg("out"),
          py::arg("ascales"), py::arg("wscales"),
          py::arg("lora_act_out"), py::arg("lora_up"),
          py::arg("bias") = torch::Tensor());

    m.def("gemm_w8a8", &nunchaku_bridge::gemm_w8a8,
          "W8A8 GEMM: INT8 weights x INT8 activations -> BF16",
          py::arg("act"), py::arg("wgt"), py::arg("out"),
          py::arg("ascales"), py::arg("wscales"),
          py::arg("bias") = torch::Tensor());

    m.def("attention_fp16", &nunchaku_bridge::attention_fp16,
          "Flash Attention (FP16) with online softmax",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("o"), py::arg("scale"));

    m.def("quantize_w4a4_act", &nunchaku_bridge::quantize_w4a4_act,
          "Quantize activations to 4-bit packed format",
          py::arg("input"), py::arg("output"), py::arg("oscales"));

    m.def("quantize_w4a4_wgt", &nunchaku_bridge::quantize_w4a4_wgt,
          "Quantize weights to 4-bit packed format",
          py::arg("input"), py::arg("output"), py::arg("oscales"));

    m.def("quantize_w8a8_act", &nunchaku_bridge::quantize_w8a8_act,
          "Quantize activations to 8-bit packed format",
          py::arg("input"), py::arg("output"), py::arg("oscales"),
          py::arg("fuse_glu") = false);
}
