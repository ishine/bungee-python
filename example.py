import numpy as np
from bungee_python import bungee
import matplotlib.pyplot as plt
import soundfile as sf
import os


def generate_test_audio(sample_rate, channels, duration_seconds, frequency=440, amplitude=0.5):
    """生成测试音频数据

    Args:
        sample_rate: 采样率 (Hz)
        channels: 音频通道数
        duration_seconds: 音频时长 (秒)
        frequency: 音频频率 (Hz)，默认440Hz (A4音符)
        amplitude: 波形振幅 (0~1)

    Returns:
        shape为(frames, channels)的numpy数组，dtype=float32
    """
    # 生成时间序列
    t = np.linspace(0., duration_seconds, int(sample_rate * duration_seconds))
    
    # 生成正弦波
    audio = amplitude * np.sin(2. * np.pi * frequency * t)
    audio = audio.astype(np.float32)
    
    # 调整通道数
    if channels == 1:
        audio = audio[:, np.newaxis]  # 单声道
    elif channels == 2:
        # 双声道，第二个通道稍微相位偏移以产生立体声效果
        audio_right = amplitude * np.sin(2. * np.pi * frequency * t + 0.2)
        audio = np.stack([audio, audio_right.astype(np.float32)], axis=-1)
    
    return audio


def process_audio(input_audio, sample_rate, speed=1.0, pitch=1.0):
    """使用Bungee处理音频

    Args:
        input_audio: 输入音频，shape=(frames, channels)
        sample_rate: 采样率 (Hz)
        speed: 速度因子 (默认1.0)
        pitch: 音高因子 (默认1.0)

    Returns:
        处理后的音频，shape=(frames, channels)
    """
    channels = input_audio.shape[1]
    
    # 创建处理器实例
    stretcher = bungee.Bungee(sample_rate=sample_rate, channels=channels)
    stretcher.instrumentation = True
    # 设置处理参数
    stretcher.set_speed(speed)
    stretcher.set_pitch(pitch)
    
    # 处理音频
    output_audio = stretcher.process(input_audio)
    
    return output_audio


def plot_waveforms(original, processed, sample_rate, title="音频波形对比"):
    """绘制原始和处理后的波形对比图

    Args:
        original: 原始音频数组
        processed: 处理后的音频数组
        sample_rate: 采样率
        title: 图表标题
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    
    # 计算时间轴
    t_orig = np.arange(original.shape[0]) / sample_rate
    t_proc = np.arange(processed.shape[0]) / sample_rate
    
    # Plot original audio waveform (show only the first channel)
    ax1.plot(t_orig, original[:, 0])
    ax1.set_title('Original Audio')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, max(t_orig[-1], t_proc[-1]))
    
    # Plot processed audio waveform (show only the first channel)
    ax2.plot(t_proc, processed[:, 0])
    ax2.set_title('Processed Audio')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(0, max(t_orig[-1], t_proc[-1]))
    
    fig.tight_layout()
    plt.suptitle(title)
    
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/{title.replace(' ', '_')}.png")
    

def save_audio(audio, sample_rate, filename):
    """保存音频到文件

    Args:
        audio: 音频数据，shape=(frames, channels)
        sample_rate: 采样率
        filename: 输出文件名
    """
    os.makedirs("output", exist_ok=True)
    sf.write(f"output/{filename}", audio, sample_rate)
    

def main():
    # 音频参数
    sample_rate = 44100
    channels = 2  # 使用立体声以展示多通道处理
    duration_seconds = 5
    frequency = 440  # A4音符
    
    print(f"生成测试音频: {frequency}Hz, {duration_seconds}秒, {channels}通道")
    input_audio = generate_test_audio(sample_rate, channels, duration_seconds, frequency)
    print(f"输入音频形状: {input_audio.shape}")
    
    # 测试不同参数组合
    test_cases = [
        {"speed": 1.0, "pitch": 1.0, "name": "原速_原音高"},
        {"speed": 0.5, "pitch": 1.0, "name": "半速_原音高"},
        {"speed": 2.0, "pitch": 1.0, "name": "倍速_原音高"},
        {"speed": 1.0, "pitch": 0.5, "name": "原速_降低八度"},
        {"speed": 1.0, "pitch": 2.0, "name": "原速_提高八度"},
        {"speed": 0.8, "pitch": 1.2, "name": "减速_提高音高"},
    ]
    
    # 处理并保存所有测试用例
    for case in test_cases:
        print(f"\n处理测试用例: {case['name']}")
        print(f"速度: {case['speed']}, 音高: {case['pitch']}")
        
        # 处理音频
        output_audio = process_audio(
            input_audio,
            sample_rate,
            speed=case['speed'],
            pitch=case['pitch']
        )
        
        print(f"输出音频形状: {output_audio.shape}")
        
        # 绘制波形对比图
        title = f"波形比较 - {case['name']}"
        plot_waveforms(input_audio, output_audio, sample_rate, title)
        
        # 保存处理后的音频
        save_audio(output_audio, sample_rate, f"{case['name']}.wav")
    
    # 保存原始音频作为参考
    save_audio(input_audio, sample_rate, "original.wav")
    print("\n处理完成! 结果保存在 'output' 目录中")


if __name__ == "__main__":
    main()