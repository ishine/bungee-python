import numpy as np
import re
from bungee_python import bungee
import matplotlib.pyplot as plt
import soundfile as sf
import os


def generate_test_audio(
    sample_rate, channels, duration_seconds, frequency=440, amplitude=0.5
):
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
    t = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds))
    # 生成随时间线性增长的振幅包络
    amp_env = np.linspace(0.0, amplitude, t.shape[0])

    # 生成正弦波，振幅随时间增长
    audio = amp_env * np.sin(2.0 * np.pi * frequency * t)
    audio = audio.astype(np.float32)

    # 调整通道数
    if channels == 1:
        audio = audio[:, np.newaxis]  # 单声道
    elif channels == 2:
        # 双声道，第二个通道稍微相位偏移以产生立体声效果
        audio_right = amp_env * np.sin(2.0 * np.pi * frequency * t + 0.2)
        audio = np.stack([audio, audio_right.astype(np.float32)], axis=-1)

    return audio


def process_audio(input_audio, sample_rate, speed=1.0, pitch=1.0):
    """使用Bungee处理音频"""
    channels = input_audio.shape[1]
    # 创建处理器实例，直接设置速度和音高参数
    processor = bungee.Bungee(
        sample_rate=sample_rate,
        channels=channels,
        speed=speed,
        pitch=pitch,
    )
    processor.set_debug(True)
    # processor.process(input_audio.copy()[:5000, :])
    processor.preroll()  # 预处理以准备音频流
    audio = processor.process(input_audio)
    print(f"延迟: {processor.get_latency() / sample_rate}秒")
    return audio


def plot_waveforms(original, processed, sample_rate, title="Waveform Comparison"):
    """Plot comparison of original and processed audio waveforms

    Args:
        original: Original audio array
        processed: Processed audio array
        sample_rate: Sample rate
        title: Chart title
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 6), sharex=False)
    print(f"Original audio max: {np.max(original)}, Processed audio max: {np.max(processed)}")
    print(f"Original audio min: {np.min(original)}, Processed audio min: {np.min(processed)}")
    # 添加均值和方差信息
    print(f"Original audio mean: {np.mean(original)}, Processed audio mean: {np.mean(processed)}")
    print(f"Original audio std: {np.std(original)}, Processed audio std: {np.std(processed)}")

    # Calculate time axis
    t_orig = np.arange(original.shape[0]) / sample_rate
    t_proc = np.arange(processed.shape[0]) / sample_rate

    # Show only the first channel (if multi-channel)
    orig_ch = original[:, 0] if original.ndim > 1 else original
    proc_ch = processed[:, 0] if processed.ndim > 1 else processed

    # Plot original audio waveform
    ax1.plot(t_orig, orig_ch, color="tab:blue", linewidth=1)
    ax1.set_title("Original Audio (First Channel)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(0, max(t_orig[-1], t_proc[-1]))

    # Plot processed audio waveform
    ax2.plot(t_proc, proc_ch, color="tab:orange", linewidth=1)
    ax2.set_title("Processed Audio (First Channel)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim(0, max(t_orig[-1], t_proc[-1]))

    # Optimize layout to avoid title overlap
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(title, fontsize=16)

    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    # Remove special characters from filename
    safe_title = re.sub(r'[\\/:*?"<>|]', "_", title.replace(" ", "_"))
    plt.savefig(f"output/{safe_title}.png", dpi=120)
    plt.close(fig)


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
    duration_seconds = 1.11
    frequency = 110  #

    print(f"生成测试音频: {frequency}Hz, {duration_seconds}秒, {channels}通道")
    input_audio = generate_test_audio(
        sample_rate, channels, duration_seconds, frequency
    )
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
            speed=case["speed"],
            pitch=case["pitch"],
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
