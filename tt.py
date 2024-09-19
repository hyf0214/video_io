import math
import time

import imageio
import json
import subprocess
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip,CompositeVideoClip, concatenate_videoclips
import asyncio
from ffmpeg import FFmpeg
from ffmpeg import Progress
from tencentcloud.common import credential
from tencentcloud.vod.v20180717 import vod_client, models
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
import base64

httpProfile = HttpProfile(endpoint="vod.tencentcloudapi.com")
clientProfile = ClientProfile(httpProfile=httpProfile)
cred = credential.Credential("AKIDsrihIyjZOBsjimt8TsN8yvv1AMh5dB44", "CPZcxdk6W39Jd4cGY95wvupoyMd0YFqW")
client_vod = vod_client.VodClient(cred, "", clientProfile)


def split_video(video_url, output_prefix):
    clip_list = []
    clip = VideoFileClip(video_url)
    total_duration_sec = clip.duration
    last_num = total_duration_sec // 60
    split_num = math.ceil(total_duration_sec // 60)
    for i in range(split_num):
        # Calculate start and end time for each clip
        start_time = i * 60
        end_time = min((i + 1) * 60, total_duration_sec)

        video_path = f"{output_prefix}/live{i + 1}.mp4"
        audio_path = f"{output_prefix}/live{i + 1}.mp3"
        image_path = f"{output_prefix}/live{i + 1}.png"

        # Ensure end time doesn't exceed the video duration
        if start_time < total_duration_sec:
            # Generate the subclip
            subclip = clip.subclip(start_time, end_time)
        else:
            start_time = last_num * 60
            end_time = total_duration_sec - start_time
            subclip = clip.subclip(start_time, end_time)

        frame = subclip.get_frame(0)  # 获取第一帧
        # 保存第一帧为图片
        imageio.imwrite(image_path, frame)
        # Save the subclip to a file
        subclip.write_videofile(video_path, codec='libx264')
        subclip.audio.write_audiofile(audio_path)

        clip_list.append({
            "audio_path": audio_path,
            "video_path": video_path
        })

    return clip_list


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        # 读取图片文件内容
        image_data = image_file.read()
        # 使用base64进行编码
        base64_data = base64.b64encode(image_data)
        # 将字节序列转换为字符串
        base64_string = base64_data.decode('utf-8')
        return base64_string


def vod_wallpaper(file_id, image_path):
    image_b64 = image_to_base64(image_path)

    # 实例化一个请求对象,每个接口都会对应一个request对象
    req = models.ModifyMediaInfoRequest()
    params = {
        "FileId": file_id,
        "CoverData": image_b64,
    }
    req.from_json_string(json.dumps(params))

    # 返回的resp是一个ModifyMediaInfoResponse的实例，与请求对象对应
    resp = client_vod.ModifyMediaInfo(req)
    print(resp)
    return True


def process_video_ffmpeg(video_url, output_prefix, slice_duration=300):
    ffmpeg = (
        FFmpeg()
        .input(video_url)
        .output(f"{output_prefix}/live_%01d.mp4",
                vcodec="copy", acodec="aac", segment_time=120, f="segment",
                reset_timestamps="1", sc_threshold="0", g="1",
                force_key_frames="expr:gte(t, n_forced * 1)")
    )

    ffmpeg.execute()

def test(input_file, output_file):
    video_codec = 'libx264'  # H.264 编码
    video_bitrate = '6000k'  # 码率 4000Kbps
    video_resolution = '1080x1920'  # 分辨率 1920x1080
    video_fps = '30'  # 帧率 30fps
    audio_codec = 'libmp3lame'  # MP3 音频编码
    audio_sample_rate = '48000'  # 采样率 48000Hz
    audio_bitrate = '128k'  # 音频码率 48Kbps
    audio_channels = '2'  # 单声道

    # 构造ffmpeg命令
    cmd = [
        'ffmpeg',
        '-i', input_file,  # 输入文件
        '-c:v', video_codec,  # 视频编码器
        '-b:v', video_bitrate,  # 视频码率
        '-s', video_resolution,  # 分辨率
        '-r', video_fps,  # 帧率
        '-c:a', audio_codec,  # 音频编码器
        '-ar', audio_sample_rate,  # 音频采样率
        '-b:a', audio_bitrate,  # 音频码率
        '-ac', audio_channels,  # 音频通道数
        output_file  # 输出文件
    ]
    # 执行命令
    subprocess.run(cmd, check=True)

def merge_last_video():
    video1 = r"D:\video_test\split\live_coding_10.mp4"
    video2 = r"D:\video_test\split\live_coding_9.mp4"
    ffmpeg1 = (
        FFmpeg()
        .input(video1)
        .input(video2)
        .concat()
        .output(f"D:/video_test/split/live_coding.mp4",
                vcodec="copy", acodec="aac", v=1, a=1)
    )
    ffmpeg1.execute()


import subprocess

from datetime import datetime
def video_conding(input_file, output_file):

    # 如果你想要更具体的格式，可以使用strftime()方法
    start_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("开始日期和时间:", start_now)

    cmd = [
        'ffmpeg',
        '-i', input_file,  # 输入文件
        '-c:v', 'libx264',  # 视频编码器
        '-b:v', '6000k',  # 视频码率
        '-s', '1080x1920',  # 分辨率
        '-r', '30',  # 帧率
        '-c:a', 'libmp3lame',  # 音频编码器
        '-ar', '48000',  # 音频采样率
        '-b:a', '128k',  # 音频码率
        '-ac', '2',  # 音频通道数
        output_file  # 输出文件
    ]
    # 执行命令
    subprocess.run(cmd, check=True)
    end_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("结束日期和时间:", end_now)
    return output_file

def process_large_video(file_path, output_path, chunk_size=1024 * 1024 * 10):  # 默认每块10MB
    # 定义ffmpeg命令行，设置为流式处理模式
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'mpegts',  # 输入格式为mpegts，这对流式数据很合适
        '-i', 'pipe:0',  # 从标准输入读取
        '-c:v', 'libx264',  # 视频编码器
        '-c:a', 'aac',  # 音频编码器
        # '-strict', 'experimental',  # 支持实验功能
        output_path
    ]

    # 创建ffmpeg子进程
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # 创建一个临时文件来存储每块的转码输出
    temp_output_file = 'temp_output.mp4'

    # 创建ffmpeg命令行来合并转码后的块
    merge_cmd = [
        'ffmpeg',
        '-y',  # 覆盖输出文件
        '-f', 'concat',  # 合并模式
        '-safe', '0',
        '-i', 'file_list.txt',  # 合并文件列表
        '-c', 'copy',  # 直接复制流
        output_path
    ]

    # 创建合并文件列表
    with open('file_list.txt', 'w') as file_list:
        # 读取文件并分块处理
        with open(file_path, 'rb') as file:
            while chunk := file.read(chunk_size):
                # 将数据块写入ffmpeg的标准输入
                ffmpeg_process.stdin.write(chunk)
                ffmpeg_process.stdin.flush()

                # 暂停并等待ffmpeg处理
                ffmpeg_process.stdin.close()
                ffmpeg_process.wait()

                # 将转码后的块写入临时文件
                with open(temp_output_file, 'ab') as temp_file:
                    temp_file.write(ffmpeg_process.stdout.read())

                # 记录合并列表
                file_list.write(f"file '{temp_output_file}'\n")

                # 重置ffmpeg进程以处理下一块
                ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                                  stderr=subprocess.PIPE)

    # 关闭ffmpeg进程
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    # 合并所有块
    subprocess.run(merge_cmd, check=True)

import io
import wave
from pathlib import Path
def main():
    ffmpeg1 = (
        FFmpeg()
        .input("D:/video_demo_mp/oohlabag-20240819-1203.mp4")
        .output(f"D:/video_demo_mp/live_%01d.mp4",
                vcodec="libx264", acodec="libmp3lame", segment_time=1800, f="segment", r="30",
                s="1080x1920", ac="2", ar="48000",
                video_bitrate="6000k",  # 使用 '-' 而不是 '_'
                audio_bitrate="128k",  # 使用 '-' 而不是 '_'
                reset_timestamps="1", sc_threshold="0", g="1",
                force_key_frames="expr:gte(t, n_forced * 1)")
    )
    ffmpeg1.execute()

    @ffmpeg1.on("progress")
    def on_progress(progress: Progress):
        print(progress)



import os
if __name__ == '__main__':
    video1 = "C:/Users/hhu30/Downloads/口感（1）.mp4"
    video2 = "C:/Users/hhu30/Downloads/口感（1） - 副本.mp4"
    ffmpeg = (
        FFmpeg()
        .option("y")
        .input(video1)
        .input(video2)
        .output("D:/video_demo_mp4/1.mp4", codec="copy")
    )


    # Execute the FFmpeg command
    ffmpeg.execute()

    @ffmpeg.on("progress")
    def on_progress(progress: Progress):
        print(progress)
