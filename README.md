# 视觉抠像客户端

这是一个基于 C# WinForms 的普通 RGB 摄像头抠像客户端，面向 4K 摄像头采集、实时预览和当前帧 PNG 保存。

## 功能

- 普通 RGB 摄像头实时预览
- RVM ONNX 模型抠像
- ONNX Runtime DirectML 推理，优先使用 Windows 独立显卡
- 支持纯色背景、背景图片、背景视频
- 支持低分辨率实时预览和当前帧 4K PNG 保存
- 摄像头列表刷新和切换

## 环境要求

- Windows 10/11
- .NET 8 SDK
- 支持 DirectML 的显卡和驱动
- 4K USB 摄像头或采集卡

## 模型

请下载 RVM ONNX 模型，并放到：

```text
models\rvm_mobilenetv3_fp32.onnx
```

推荐模型：

```text
https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx
```

模型文件较大，默认不提交到 GitHub。

## 启动

在项目目录运行：

```powershell
.\start_native_client.cmd
```

或者：

```powershell
dotnet run --project .\NativeMattingClient
```

## 使用说明

1. 点击刷新摄像头，选择需要使用的摄像头。
2. 设置采集分辨率，4K 摄像头建议使用 `3840 x 2160`。
3. 选择背景模式：纯色、图片或视频。
4. 点击开始预览。
5. 调整抠像参数，观察实时预览效果。
6. 点击保存当前帧，输出带背景合成后的 PNG。

## 参数建议

- 实时预览边长：`720` 或 `960`
- 预览 downsample：`0.25`
- 4K 保存 downsample：`0.125`
- 如果边缘质量不足，可尝试 `0.15` 或 `0.20`，速度会下降

## 开源协议

本项目使用 MIT License。第三方依赖、RVM 模型和相关算法请遵守其原始项目许可证。
