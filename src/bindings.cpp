#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "bungee/Bungee.h"
#include "bungee/Stream.h"
#include <vector>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <string>
#include <iostream>

namespace py = pybind11;

class BungeePy
{
public:
    BungeePy(int sample_rate, int channels)
        : sample_rate_(sample_rate), channels_(channels),
          speed_(1.0), pitch_(1.0), first_call_(true)
    {
        stretcher_ = std::make_unique<Bungee::Stretcher<Bungee::Basic>>(
            Bungee::SampleRates{sample_rate, sample_rate}, 
            channels
        );
        stretcher_->enableInstrumentation(false);
    }

    void set_speed(double speed)
    {
        speed_ = std::max(std::abs(speed), 1e-9);
    }
    
    void set_pitch(double pitch)
    {
        pitch_ = pitch;
    }
    
    void set_instrumentation(bool enable)
    {
        stretcher_->enableInstrumentation(enable);
    }

    py::array_t<float> process(py::array_t<float, py::array::c_style | py::array::forcecast> input_py_array)
    {
        // 验证输入数据
        if (input_py_array.ndim() != 2)
            throw std::runtime_error("输入必须是2D数组 (frames, channels)");
        if (input_py_array.shape(1) != channels_)
            throw std::runtime_error("输入通道数不匹配");

        // 获取输入框架数
        ssize_t input_frames = input_py_array.shape(0);
        if (input_frames == 0) {
            return py::array_t<float>(std::vector<ssize_t>{0, static_cast<ssize_t>(channels_)});
        }

        // 计算期望的输出样本数
        double output_sample_count_ideal = (input_frames * sample_rate_) / (speed_ * sample_rate_);
        int max_output_frames = static_cast<int>(std::ceil(output_sample_count_ideal)) + 10;

        // 初始化或重新初始化Stream对象
        if (!stream_ || max_buffer_size_ < static_cast<int>(input_frames)) {
            max_buffer_size_ = std::max(4096, static_cast<int>(input_frames));
            stream_ = std::make_unique<Bungee::Stream<Bungee::Basic>>(
                *stretcher_,
                max_buffer_size_,
                channels_
            );
            first_call_ = true;  // 重置标志
        }

        // 提取输入数据的指针
        auto input_buffer = input_py_array.unchecked<2>();
        
        // 创建输入指针数组 (保持与main.cpp相同的方式)
        std::vector<const float*> input_ptrs(channels_);
        
        // 转换输入数据格式 (按通道分离)
        std::vector<std::vector<float>> channel_buffers(channels_);
        for (int c = 0; c < channels_; ++c) {
            channel_buffers[c].resize(input_frames);
            for (ssize_t i = 0; i < input_frames; ++i) {
                channel_buffers[c][i] = input_buffer(i, c);
            }
            input_ptrs[c] = channel_buffers[c].data();
        }

        // 为输出分配内存
        std::vector<ssize_t> shape = {static_cast<ssize_t>(max_output_frames), static_cast<ssize_t>(channels_)};
        py::array_t<float> output_array(shape);
        py::buffer_info output_buf = output_array.request();
        float* output_ptr = static_cast<float*>(output_buf.ptr);
        
        // 设置输出指针数组
        std::vector<float*> output_ptrs(channels_);
        for (int c = 0; c < channels_; ++c) {
            output_ptrs[c] = output_ptr + c * max_output_frames;
        }
        
        // 处理音频
        int output_frames = stream_->process(
            input_ptrs.data(),
            output_ptrs.data(),
            static_cast<int>(input_frames),
            output_sample_count_ideal,
            pitch_
        );
        
        // 创建正确大小的返回数组
        py::array_t<float> final_output = py::array_t<float>({static_cast<ssize_t>(output_frames), static_cast<ssize_t>(channels_)});
        auto final_output_buf = final_output.request();
        float* final_output_ptr = static_cast<float*>(final_output_buf.ptr);
        
        // 按行优先顺序填充输出数组
        for (int i = 0; i < output_frames; ++i) {
            for (int c = 0; c < channels_; ++c) {
                final_output_ptr[i * channels_ + c] = output_ptrs[c][i];
            }
        }
        
        return final_output;
    }

    double get_speed() const { return speed_; }
    double get_pitch() const { return pitch_; }
    int get_latency() const { 
        return stream_ ? stream_->latency() : 0; 
    }

private:
    int sample_rate_;
    int channels_;
    int max_buffer_size_ = 4096;
    double speed_;
    double pitch_;
    bool first_call_;
    std::unique_ptr<Bungee::Stretcher<Bungee::Basic>> stretcher_;
    std::unique_ptr<Bungee::Stream<Bungee::Basic>> stream_;
};

PYBIND11_MODULE(bungee, m)
{
    py::class_<BungeePy>(m, "Bungee")
        .def(py::init<int, int>(), py::arg("sample_rate"), py::arg("channels"))
        .def_property("speed", &BungeePy::get_speed, &BungeePy::set_speed)
        .def_property("pitch", &BungeePy::get_pitch, &BungeePy::set_pitch)
        .def("set_speed", &BungeePy::set_speed)
        .def("set_pitch", &BungeePy::set_pitch)
        .def("set_instrumentation", &BungeePy::set_instrumentation)
        .def("get_latency", &BungeePy::get_latency)
        .def("process", &BungeePy::process, py::arg("input"), R"pbdoc(
            处理音频: 输入float32二维数组 (frames, channels), 返回处理后音频数组
        )pbdoc");
}
