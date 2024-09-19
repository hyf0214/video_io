import os
import math
import base64
import re
from typing import Dict, Any, List
import time
import sentry_sdk
from ffmpeg.asyncio import FFmpeg
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
import logging
import json
import uvicorn
import subprocess
from pydantic import BaseModel
from moviepy.editor import VideoFileClip, concatenate_videoclips
from fastapi import FastAPI, HTTPException, Request
from fastapi_healthcheck import HealthCheckFactory, healthCheckRoute

from qcloud_vod.vod_upload_client import VodUploadClient
from qcloud_vod.model import VodUploadRequest

from tencentcloud.common import credential
from tencentcloud.vod.v20180717 import vod_client, models
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException


class VideoIo(BaseModel):
    file_id: str
    file_name: str
    # record_video_io: List = None
    metadata: Dict[Any, Any]


class Video(FastAPI):
    httpProfile = HttpProfile(endpoint="vod.tencentcloudapi.com")
    clientProfile = ClientProfile(httpProfile=httpProfile)

    def __init__(self):
        super().__init__()
        self.logger = self.init_logger()
        self.client_down = VodUploadClient("AKIDCR4fQDyonkfUME8AKTVZZWK2kBXfhgfX", "vnjnsu14425FNYr5RsMpNyibcsEglwdV")
        self.concurrent_upload_number = 10
        self.split_class_id = int(os.environ.get("VOD_ SPLIT_MATERIAL", 1196410))
        self.SubAppId = int(os.environ.get("VOD_SUB_APP_ID", 1500032969))
        self.raw_class_id = int(os.environ.get(" VOD_ORIGINAL_MATERIAL", 1196409))
        self.cred = credential.Credential("AKIDsrihIyjZOBsjimt8TsN8yvv1AMh5dB44", "CPZcxdk6W39Jd4cGY95wvupoyMd0YFqW")
        self.client_vod = vod_client.VodClient(self.cred, "", self.clientProfile)
        self.video_codec = 'libx264'  # H.264 编码
        self.video_bitrate = '6000k'  # 码率 4000Kbps
        self.video_resolution = '1080x1920'  # 分辨率 1920x1080
        self.video_fps = '30'  # 帧率 30fps
        self.audio_codec = 'libmp3lame'  # MP3 音频编码
        self.audio_sample_rate = '48000'  # 采样率 48000Hz
        self.audio_bitrate = '128k'  # 音频码率 48Kbps
        self.audio_channels = '2'  # 单声道

    @staticmethod
    def get_log_level_from_env(env_key: str):
        log_level = os.environ.get(env_key, "DEBUG")
        log_level = log_level.upper()
        if log_level == "DEBUG":
            log_level = logging.DEBUG
        elif log_level == "WARNING":
            log_level = logging.WARNING
        elif log_level == "ERROR":
            log_level = logging.ERROR
        else:  # fallback to Info level
            log_level = logging.INFO
        return log_level

    def init_logger(self):
        logger = logging.getLogger("uvicorn")
        logger.handlers.clear()
        log_level = self.get_log_level_from_env("LOGGER_LOG_LEVEL")
        logger.setLevel(log_level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(filename)s#%(funcName)s:%(lineno)d] %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        return logger

    def video_conding(self, input_file, output_file):
        cmd = [
            'ffmpeg',
            '-i', input_file,  # 输入文件
            '-c:v', self.video_codec,  # 视频编码器
            '-b:v', self.video_bitrate,  # 视频码率
            '-s', self.video_resolution,  # 分辨率
            '-r', self.video_fps,  # 帧率
            '-c:a', self.audio_codec,  # 音频编码器
            '-ar', self.audio_sample_rate,  # 音频采样率
            '-b:a', self.audio_bitrate,  # 音频码率
            '-ac', self.audio_channels,  # 音频通道数
            output_file  # 输出文件
        ]
        # 执行命令
        subprocess.run(cmd, check=True)
        return output_file

    async def image_to_base64(self, image_path: str) -> base64:
        try:
            with open(image_path, "rb") as image_file:
                # 读取图片文件内容
                image_data = image_file.read()
                # 使用base64进行编码
                base64_data = base64.b64encode(image_data)
                # 将字节序列转换为字符串
                base64_string = base64_data.decode('utf-8')
                return base64_string
        except FileNotFoundError:
            self.logger.error(f"img2b64 Error: 文件 '{image_path}' 未找到。")
        except Exception as e:
            self.logger.error(f"img2b64 Error: 发生异常 '{e}'.")

    async def upload_vod(self, video_path: str) -> str:
        client = self.client_down
        request = VodUploadRequest()
        request.MediaFilePath = video_path
        request.ConcurrentUploadNumber = self.concurrent_upload_number
        request.ClassId = self.split_class_id
        request.SubAppId = self.SubAppId
        try:
            response = client.upload("ap-shanghai", request)
            self.logger.info(f"文件:{video_path}上传vod成功，fileID：{response.FileId}")
            return response.FileId
        except Exception as e:
            self.logger.error(f"上传vod失败{e}")
            raise HTTPException(status_code=500, detail={
                "code": 500,
                "level": "RED",
                "message": "服务器上传vod失败",
                "data": str(e)
            })

    async def get_vod_client(self, fileid: str) -> tuple[str, str, str, float, str]:
        try:
            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = models.DescribeMediaInfosRequest()
            # req.SubAppId = self.SubAppId
            params = {
                "SubAppId": self.SubAppId,
                "FileIds": [fileid],
                "ClassIds": [self.raw_class_id]
            }
            req.from_json_string(json.dumps(params))

            # 返回的resp是一个DescribeMediaInfosResponse的实例，与请求对象对应
            resp = self.client_vod.DescribeMediaInfos(req)
            # 输出json格式的字符串回包
            data = json.loads(resp.to_json_string())
            # transcoding_url = data["MediaInfoSet"][0]["TranscodeInfo"]["TranscodeSet"][1]["Url"]
            get_intranet_media_url = data["MediaInfoSet"][0]["BasicInfo"]["IntranetMediaUrl"]
            transcoding_url = data["MediaInfoSet"][0]["BasicInfo"]["MediaUrl"]
            self.logger.info(f"data:{data} get_intranet_media_url:{get_intranet_media_url} transcoding_url:{transcoding_url}")
            type = data["MediaInfoSet"][0]["BasicInfo"]["Type"]
            name = data["MediaInfoSet"][0]["BasicInfo"]["Name"]
            video_duration = data["MediaInfoSet"][0]["MetaData"]["Duration"]
            return get_intranet_media_url, type, name, video_duration, transcoding_url
        except TencentCloudSDKException as err:
            self.logger.error(f"获取文件信息失败:{err}")
            raise HTTPException(status_code=500, detail={
                "code": 500,
                "level": "RED",
                "message": "服务器获取vod失败",
                "data": str(err)
            })

    async def download_file(self, url: str, save_path: str, video_path: str, transcoding_url: str) -> bool:
        attempts = 0
        current_url = url
        while attempts < 5:
            try:
                if os.path.exists(save_path):
                    resume_header = {'Range': f"bytes={os.path.getsize(save_path)}-"}
                else:
                    resume_header = {}
                response = requests.get(current_url, headers=resume_header, stream=True)
                response.raise_for_status()
                with open(save_path, "ab") as f:  # Use "ab" to append in binary mode
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
            except Exception as e:
                attempts += 1
                self.logger.warning(f"Download attempt {attempts}. Retrying...")
                if attempts != 0:
                    self.logger.info(f"Switching to transcoding URL: {transcoding_url}")
                    current_url = transcoding_url  # 切换到备用 URL
        self.logger.error(f"保存vod文件出错")
        await self.delete_folder(video_path)
        raise HTTPException(status_code=500, detail={
            "code": 500,
            "level": "RED",
            "message": "保存vod文件出错",
            "data": ""
        })

    async def download_vod(self, file_id: str, video_path: str) -> tuple[str, str, float]:
        video_url, file_type, name, video_duration, transcoding_url = await self.get_vod_client(file_id)
        down_video_path = f"{video_path}/{name}.{file_type}"
        self.logger.info("开始下载视频：:%s", down_video_path)
        await self.download_file(video_url, down_video_path, video_path, transcoding_url)
        # # 先转码再切割
        # coding_video_path = f"{video_path}/{name}_coding.{file_type}"
        # coding_video_path = self.video_conding(down_video_path, coding_video_path)
        # os.remove(down_video_path)
        # return coding_video_path, name, video_duration

        # 先切割再转码
        return down_video_path, name, video_duration

    async def vod_wallpaper(self, file_id: str, image_path: str):
        image_b64 = await self.image_to_base64(image_path)
        try:
            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = models.ModifyMediaInfoRequest()
            params = {
                "SubAppId": self.SubAppId,
                "FileId": file_id,
                "ClassIds": [self.split_class_id],
                "CoverData": image_b64,
            }
            req.from_json_string(json.dumps(params))

            # 返回的resp是一个ModifyMediaInfoResponse的实例，与请求对象对应
            resp = self.client_vod.ModifyMediaInfo(req)
            data = json.loads(resp.to_json_string())
            cover_url = data.get("CoverUrl", "封面None")
            self.logger.info(f"视频封面地址：{cover_url}")
            return True
        except TencentCloudSDKException as err:
            self.logger.error(f"vod绑定壁纸失败 {err}")

    async def delete_vod(self, file_id: str):
        try:
            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = models.DeleteMediaRequest()
            params = {
                "FileId": file_id,
                "SubAppId": self.SubAppId
            }
            req.from_json_string(json.dumps(params))
            resp = self.client_vod.DeleteMedia(req)
        except TencentCloudSDKException as err:
            self.logger.error(f"vod删除切片失败 {err}")

    async def vod_add_frame_descs(self, file_id: str, frame_list: list[dict]) -> bool:
        try:
            # 实例化一个请求对象,每个接口都会对应一个request对象
            req = models.ModifyMediaInfoRequest()
            params = {
                "SubAppId": self.SubAppId,
                "ClassIds": [self.raw_class_id],
                "FileId": file_id,
                "AddKeyFrameDescs": frame_list,
            }
            req.from_json_string(json.dumps(params))

            # 返回的resp是一个ModifyMediaInfoResponse的实例，与请求对象对应
            resp = self.client_vod.ModifyMediaInfo(req)
            self.logger.info(f"AddKeyFrameDescs绑定成功！")
            return True
        except TencentCloudSDKException as err:
            self.logger.error(f"vod信息绑定失败 {err}")

    # def split_video_memory(self,video_path):
    #     video_memory = os.path.getsize(video_path)
    #     if video_memory >= 1024 * 1024 * 1024:
    #
    #         video_time = self.get_video_time(video_path)
    #         # 创建两个子剪辑
    #         clip1 = video.subclip(0, split_time)
    #         clip2 = video.subclip(split_time, video.duration)


    @staticmethod
    def numerical_sort(value):
        parts = re.split(r'(\d+)', value)
        parts[1::2] = map(int, parts[1::2])  # 将数字部分转换为整数
        return parts

    @staticmethod
    async def get_video_time(video_path: str) -> float:
        ffprobe = FFmpeg(executable="ffprobe").input(
            video_path,
            print_format="json",  # ffprobe will output the results in JSON format
            show_streams=None,
        )
        media = json.loads(await ffprobe.execute())
        time = media['streams'][0]['duration']
        return float(time)

    async def split_image(self, video_url: str, img_path: str) -> str:
        ffmpeg = (
            FFmpeg()
            .input(video_url)
            .output(img_path,
                    vframes=1, vf='select=eq(n\\,0)')
        )

        @ffmpeg.on("start")
        def on_start(arguments: list[str]):
            self.logger.info(f"arguments:{arguments}")

        @ffmpeg.on("stderr")
        def on_stderr(line):
            self.logger.info(f"stderr:{line}")

        @ffmpeg.on("completed")
        def on_completed():
            self.logger.info("completed")

        await ffmpeg.execute()
        return img_path

    async def run_info(self, output_prefix: str, file_id: str, name: str, split_duration: float, video_duration: float,
                       file_name: str, video_path: str) -> List[dict]:
        clip_list = []
        split_time = []
        start_path = output_prefix.split("temp")[1]
        files = sorted(os.listdir(output_prefix), key=self.numerical_sort)
        self.logger.info(files)
        if len(files) <= 3:
            video_path = f"{output_prefix}/{name}_0.mp4"
            out_video_path = f"{output_prefix}/{name}_0_conding.mp4"
            self.video_conding(video_path, out_video_path)
            os.remove(video_path)
            split_file_id = await self.upload_vod(out_video_path)
            clip_list.append({
                "file_id": file_id,
                "file_name": file_name,
                "split_file_id": split_file_id,
                "audio_path": f"{start_path}/{name}_0.mkv",
                "video_path": f"{start_path}/{name}_0_conding.mp4",
                "split_time": split_duration,
                "video_time": video_duration,
            })
        else:
            try:
                mp4_files = [
                    file
                    for file in os.listdir(output_prefix)
                    if file.endswith('.mp4')
                ]
                mp4_files_sorted = sorted(mp4_files,
                                          key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(0)))
                for index, file in enumerate(mp4_files_sorted, 1):
                    video_path = f"{output_prefix}/{file}"
                    img_path = video_path.replace("mp4", "png")
                    out_video_path = self.video_conding(video_path, video_path.replace(".", "_conding."))
                    os.remove(video_path)
                    await self.split_image(out_video_path, img_path)
                    split_file_id = await self.upload_vod(out_video_path)
                    await self.vod_wallpaper(split_file_id, img_path)
                    duration_calc = split_duration * index
                    split_time.append({
                        "TimeOffset": duration_calc - video_duration if duration_calc > video_duration else duration_calc,
                        "Content": split_file_id
                    })
                    clip_list.append({
                        "file_id": file_id,
                        "file_name": file_name,
                        "split_file_id": split_file_id,
                        "audio_path": video_path.split("temp")[1].replace(".mp4", ".mkv"),
                        "video_path": out_video_path.split("temp")[1],
                        "split_time": split_duration,
                        "video_time": video_duration - (split_duration * (index - 1)) if len(
                            mp4_files_sorted) == index else split_duration,
                    })
                    os.remove(img_path)
                await self.vod_add_frame_descs(file_id, split_time)
            except Exception as err:
                for data in clip_list:
                    split_file_id = data["split_file_id"]
                    await self.delete_vod(split_file_id)
                self.logger.error(f"处理返回数据失败：{err}")
                raise HTTPException(status_code=500, detail={
                    "code": 500,
                    "level": "RED",
                    "message": "保存vod文件出错",
                    "data": f"处理返回数据失败：{err}"
                })
        return clip_list

    async def delete_folder(self, folder_path: str, retries=3, delay=5):
        """
        尝试删除文件夹及其内容，失败时重试指定次数。
        :param folder_path: 要删除的文件夹路径
        :param retries: 失败时重试的次数
        :param delay: 重试前的等待时间（秒）
        """
        for attempt in range(retries):
            try:
                if os.path.exists(folder_path):
                    # 逐个文件删除
                    for root, dirs, files in os.walk(folder_path, topdown=False):
                        for file in files:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                    # 删除空文件夹
                    os.rmdir(folder_path)
                    self.logger.info(f"文件夹{folder_path}删除成功！", )
                    return
            except Exception as e:
                self.logger.error(f"删除文件夹 '{folder_path}' 失败（尝试 {attempt + 1}/{retries}）：{e}")
                time.sleep(delay)
        return

    def merge_last_video(self, video_duration, split_duration, output_prefix, name):
        if video_duration / split_duration != int(video_duration / split_duration) and int(video_duration / split_duration) != 0:
            video1 = f"{output_prefix}/{name}_{int(video_duration / split_duration)-1}.mp4"
            video2 = f"{output_prefix}/{name}_{int(video_duration / split_duration)}.mp4"
            ffmpeg = (
                FFmpeg()
                .option("y")
                .input(video1)
                .input(video2)
                .output(f"{output_prefix}/{name}_{int(video_duration / split_duration)-1}.mp4", codec="copy")
            )

            # clip1 = VideoFileClip(video1)
            # clip2 = VideoFileClip(video2)
            # concatenate_videoclips([clip1, clip2]).write_videofile(f"{output_prefix}/{name}_{int(video_duration / split_duration)-1}.mp4")
            # os.remove(video2)
            # clip1.close()
            # clip2.close()
        return


    @staticmethod
    def create_usercount(output_prefix: str, count: int) -> bool:
        for file in os.listdir(output_prefix):
            if file.endswith('.mp4'):
                info_path = f"{output_prefix}/{file.replace('.mp4', '.metadata')}"
                with open(info_path, "w", encoding="utf-8") as f:
                    f.write(str(count))
        return True

    async def split_video(self, video_url: str, output_prefix: str, file_id: str, name: str,
                          split_duration: float, file_name: str, video_duration: float, user_count: int, video_path: str) -> List[dict]:
        ffmpeg1 = (
            FFmpeg()
            .input(video_url)
            .output(f"{output_prefix}/{name}_%01d.mp4",
                    vcodec="copy", acodec="aac", segment_time=split_duration, f="segment",
                    reset_timestamps="1", sc_threshold="0", g="1",
                    force_key_frames="expr:gte(t, n_forced * 1)")
        )
        await ffmpeg1.execute()
        for file in os.listdir(output_prefix):
            if file.endswith(".mp4"):
                file_type = file.replace('.mp4', ".mkv")
                ffmpeg2 = (
                    FFmpeg()
                    # .input(f"{output_prefix}/{file}")
                    .input(f"{output_prefix}/{file}")
                    .output(f"{output_prefix}/{file_type}", map="0:a", acodec="copy")
                )
                await ffmpeg2.execute()

        @ffmpeg1.on("completed")
        def on_completed():
            self.logger.info("completed")

        # clip_time = await self.get_video_time(video_url)
        self.logger.info(f"素材视频时长：{video_duration}")
        self.create_usercount(output_prefix, user_count)
        self.merge_last_video(video_duration, split_duration, output_prefix, name)
        clip_list = await self.run_info(output_prefix, file_id, name, split_duration, video_duration, file_name,
                                        video_path)
        return clip_list

    async def main(self, task_id: str, file_id: str, split_duration: float, file_name: str, user_count: int) -> List[
        dict]:
        video_path = f"/app/temp/{task_id}"
        split_video_path = f"{video_path}/{task_id}_split"
        os.makedirs(split_video_path, exist_ok=True)
        app.logger.info(f"创建切割文件夹:{split_video_path}")
        sever_path, name, video_duration = await self.download_vod(file_id, video_path)

        clip_list = await app.split_video(sever_path, split_video_path, file_id, name, split_duration, file_name,
                                          video_duration, user_count, video_path)
        return clip_list

sentry_sdk.init(
    dsn="http://5ad84a97b5095c09a506b531bcf1c4f6@122.51.245.126:80/6",
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    traces_sample_rate=1.0,
)

app = Video()
_healthChecks = HealthCheckFactory()
app.add_api_route('/health', endpoint=healthCheckRoute(factory=_healthChecks))


@app.post("/api/record_video_io")
async def record_video_io(video_io: VideoIo, request: Request):
    app.logger.info("开始视频切割任务！")
    file_id = video_io.file_id
    file_name = video_io.file_name
    task_id = request.headers.get("X-Task-Id")
    split_duration = video_io.metadata.get("slice_duration")
    user_count = int(video_io.metadata.get("batch_count"))
    clip_list = await app.main(task_id, file_id, split_duration, file_name, user_count)

    app.logger.info(f"返回数据：{clip_list}")
    return clip_list


@app.post("/api/retry_video_io")
async def record_video_io(video_io: VideoIo, request: Request):
    app.logger.info(f"重试任务！")
    file_id = video_io.fileId
    task_id = request.headers.get("X-Task-Id")
    retry_io = video_io.record_video_io
    file_name = video_io.file_name
    split_duration = video_io.metadata.get("slice_duration")
    user_count = int(video_io.metadata.get("batch_count"))
    split_time = retry_io[0].get("split_time") if retry_io else None

    clip_list = []
    if split_duration == split_time:
        app.logger.info("split_duration等于split_time")
        for data in retry_io:
            split_file_id = data.get("split_file_id")
            split_time = data.get("split_time")
            clip_list.append({
                "split_file_id": split_file_id,
                "split_time": split_time
            })
    else:
        clip_list = await app.main(task_id, file_id, split_duration, file_name, user_count)
    app.logger.info(f"重试返回数据：{clip_list}")
    return clip_list


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
