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
    BungeePy(int sample_rate, int channels, double speed = 1.0, double pitch = 1.0, int log2_synthesis_hop_adjust = -1)
        : sample_rate_(sample_rate), channels_(channels),
          speed_(std::max(std::abs(speed), 1e-9)), pitch_(pitch), max_buffer_size_(2048)
    {
        stretcher_ = std::make_unique<Bungee::Stretcher<Bungee::Basic>>(
            Bungee::SampleRates{sample_rate, sample_rate}, 
            channels,
            log2_synthesis_hop_adjust
        );
        stretcher_->enableInstrumentation(false);
        
        // 预分配缓冲区
        input_ptrs_.resize(channels_);
        output_ptrs_.resize(channels_);
        channel_buffers_.resize(channels_);
        
        // 初始化 Stream
        initializeStream();
    }

    // 保留 property getter/setter 但标记为弃用（仍可通过属性访问）
    [[deprecated("Use constructor parameter instead")]]
    void set_speed(double speed)
    {
        speed_ = std::max(std::abs(speed), 1e-9);
    }
    
    [[deprecated("Use constructor parameter instead")]]
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

        // 获取输入帧数
        ssize_t input_frames = input_py_array.shape(0);
        if (input_frames == 0) {
            return py::array_t<float>(std::vector<ssize_t>{0, static_cast<ssize_t>(channels_)});
        }

        // 如果需要，重新初始化 Stream
        if (!stream_ || max_buffer_size_ < static_cast<int>(input_frames)) {
            max_buffer_size_ = std::max(max_buffer_size_ * 2, static_cast<int>(input_frames));
            initializeStream();
        }

        // 计算期望的输出样本数（参考 main.cpp）
        double output_sample_count_ideal = (input_frames * sample_rate_) / (speed_ * sample_rate_);
        int max_output_frames = static_cast<int>(std::ceil(output_sample_count_ideal)) + 10;

        // 提取输入数据指针
        auto input_buffer = input_py_array.unchecked<2>();
        
        // 准备通道缓冲区（优化分配策略）
        if (channel_buffers_.size() < channels_) {
            channel_buffers_.resize(channels_);
        }
        
        // 按通道分离数据（参考 main.cpp 的数据访问方式）
        for (int c = 0; c < channels_; ++c) {
            if (channel_buffers_[c].size() < input_frames) {
                channel_buffers_[c].resize(input_frames);
            }
            for (ssize_t i = 0; i < input_frames; ++i) {
                channel_buffers_[c][i] = input_buffer(i, c);
            }
            input_ptrs_[c] = channel_buffers_[c].data();
        }

        // 为输出分配内存（优化输出内存管理）
        py::array_t<float> output_array({static_cast<ssize_t>(max_output_frames), static_cast<ssize_t>(channels_)});
        py::buffer_info output_buf = output_array.request();
        float* output_ptr = static_cast<float*>(output_buf.ptr);
        
        // 设置输出指针（参考 main.cpp 的处理方式）
        for (int c = 0; c < channels_; ++c) {
            output_ptrs_[c] = output_ptr + c * max_output_frames;
        }
        
        // 处理音频
        int output_frames = stream_->process(
            input_ptrs_.data(),
            output_ptrs_.data(),
            static_cast<int>(input_frames),
            output_sample_count_ideal,
            pitch_
        );
        
        // 创建正确大小的返回数组（与原代码保持一致）
        py::array_t<float> final_output({static_cast<ssize_t>(output_frames), static_cast<ssize_t>(channels_)});
        auto final_output_buf = final_output.request();
        float* final_output_ptr = static_cast<float*>(final_output_buf.ptr);
        
        // 按行优先顺序填充输出数组
        for (int i = 0; i < output_frames; ++i) {
            for (int c = 0; c < channels_; ++c) {
                final_output_ptr[i * channels_ + c] = output_ptrs_[c][i];
            }
        }
        
        return final_output;
    }

    double get_speed() const { return speed_; }
    double get_pitch() const { return pitch_; }
    int get_latency() const { 
        return stream_ ? static_cast<int>(stream_->latency()) : 0; 
    }

private:
    void initializeStream() {
        stream_ = std::make_unique<Bungee::Stream<Bungee::Basic>>(
            *stretcher_,
            max_buffer_size_,
            channels_
        );
        
        Bungee::Request request;
        request.position = 0;
        request.speed = speed_;
        request.pitch = pitch_;
        
        stretcher_->preroll(request);
    }

    int sample_rate_;
    int channels_;
    int max_buffer_size_;
    double speed_;
    double pitch_;
    std::unique_ptr<Bungee::Stretcher<Bungee::Basic>> stretcher_;
    std::unique_ptr<Bungee::Stream<Bungee::Basic>> stream_;
    Bungee::InputChunk inputChunk_;  // 保存处理状态
    
    // 预分配缓冲区，避免重复创建
    std::vector<std::vector<float>> channel_buffers_;
    std::vector<const float*> input_ptrs_;
    std::vector<float*> output_ptrs_;
};

PYBIND11_MODULE(bungee, m)
{
    py::class_<BungeePy>(m, "Bungee")
        .def(py::init<int, int, double, double, int>(), py::arg("sample_rate"), py::arg("channels"), py::arg("speed") = 1.0, py::arg("pitch") = 1.0, py::arg("log2_synthesis_hop_adjust") = -1)
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