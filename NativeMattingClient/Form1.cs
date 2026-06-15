using System.Diagnostics;
using System.Management;
using DirectShowLib;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using CvSize = OpenCvSharp.Size;

namespace NativeMattingClient;

public partial class Form1 : Form
{
    private readonly string _rootDir;
    private readonly string _modelDir;
    private readonly string _captureDir;

    private readonly PictureBox _preview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(22, 22, 22), SizeMode = PictureBoxSizeMode.Zoom };
    private readonly ComboBox _modeList = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 190, Height = 42, Enabled = false };
    private readonly ComboBox _cameraList = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 460, Height = 42 };
    private readonly ComboBox _backgroundMode = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 190, Height = 42 };
    private readonly Button _refreshButton = new() { Text = "Refresh Camera", Width = 160, Height = 42 };
    private readonly Button _startButton = new() { Text = "Start Preview", Width = 150, Height = 42 };
    private readonly Button _stopButton = new() { Text = "Stop", Width = 105, Height = 42, Enabled = false };
    private readonly Button _saveButton = new() { Text = "Save 4K PNG", Width = 150, Height = 42, Enabled = false };
    private readonly Button _pickBgImageButton = new() { Text = "BG Image", Width = 120, Height = 42 };
    private readonly Button _pickBgVideoButton = new() { Text = "BG Video", Width = 120, Height = 42 };
    private readonly NumericUpDown _widthInput = new() { Minimum = 320, Maximum = 7680, Value = 3840, Increment = 160, Width = 115, Height = 42 };
    private readonly NumericUpDown _heightInput = new() { Minimum = 240, Maximum = 4320, Value = 2160, Increment = 90, Width = 115, Height = 42 };
    private readonly NumericUpDown _previewMaxInput = new() { Minimum = 360, Maximum = 2160, Value = 960, Increment = 120, Width = 105, Height = 42 };
    private readonly NumericUpDown _previewDownsampleInput = new() { Minimum = 0.05M, Maximum = 1.0M, DecimalPlaces = 3, Increment = 0.025M, Value = 0.25M, Width = 105, Height = 42 };
    private readonly NumericUpDown _saveDownsampleInput = new() { Minimum = 0.05M, Maximum = 1.0M, DecimalPlaces = 3, Increment = 0.025M, Value = 0.125M, Width = 105, Height = 42 };
    private readonly CheckBox _useGpuBox = new() { Text = "DirectML GPU", Checked = true, AutoSize = false, Width = 145, Height = 42, TextAlign = ContentAlignment.MiddleLeft };
    private readonly Label _status = new() { AutoSize = false, Dock = DockStyle.Bottom, Height = 36, TextAlign = ContentAlignment.MiddleLeft };

    private readonly object _latestLock = new();
    private readonly object _backgroundLock = new();
    private Mat? _latestFrame;
    private Mat? _latestComposited;
    private CancellationTokenSource? _cts;
    private Task? _captureTask;
    private Task? _inferenceTask;
    private VideoCapture? _backgroundVideo;
    private Mat? _backgroundImage;
    private Color _backgroundColor = Color.LimeGreen;
    private RvmOnnxMatting? _matting;
    private volatile bool _running;
    private int _captureFrames;
    private int _inferFrames;
    private readonly Stopwatch _statsClock = Stopwatch.StartNew();

    public Form1()
    {
        InitializeComponent();

        _rootDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        _modelDir = Path.Combine(_rootDir, "models");
        _captureDir = Path.Combine(_rootDir, "outputs", "captures");
        Directory.CreateDirectory(_modelDir);
        Directory.CreateDirectory(_captureDir);

        Text = "Native RVM Matting Client - RGB DirectML";
        Width = 1360;
        Height = 860;
        MinimumSize = new System.Drawing.Size(1040, 680);

        BuildUi();
        WireEvents();
        _ = RefreshCamerasAsync();
        TryLoadModel();
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        StopPipeline();
        lock (_backgroundLock)
        {
            _backgroundImage?.Dispose();
            _backgroundVideo?.Dispose();
            _backgroundImage = null;
            _backgroundVideo = null;
        }

        _matting?.Dispose();
        _matting = null;
        base.OnFormClosing(e);
    }

    private void BuildUi()
    {
        Font = new Font("Segoe UI", 11F, FontStyle.Regular, GraphicsUnit.Point);
        _modeList.Items.Add("RGB Matting");
        _modeList.SelectedIndex = 0;
        _backgroundMode.Items.AddRange(["Solid Color", "Image", "Video"]);
        _backgroundMode.SelectedIndex = 0;

        var top = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 178,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            Padding = new Padding(10),
            BackColor = Color.FromArgb(245, 245, 245)
        };

        top.Controls.AddRange([
            LabelOf("Mode"), _modeList,
            LabelOf("Camera"), _cameraList, _refreshButton,
            LabelOf("Width"), _widthInput,
            LabelOf("Height"), _heightInput,
            LabelOf("Preview Max"), _previewMaxInput,
            LabelOf("Preview S"), _previewDownsampleInput,
            LabelOf("Save S"), _saveDownsampleInput,
            _useGpuBox,
            _startButton, _stopButton, _saveButton,
            LabelOf("Background"), _backgroundMode, _pickBgImageButton, _pickBgVideoButton
        ]);

        Controls.Add(_preview);
        Controls.Add(_status);
        Controls.Add(top);
    }

    private static Label LabelOf(string text) => new()
    {
        Text = text,
        AutoSize = false,
        Width = Math.Max(82, text.Length * 12),
        Height = 42,
        TextAlign = ContentAlignment.MiddleCenter,
        Margin = new Padding(10, 3, 2, 10)
    };

    private void WireEvents()
    {
        _refreshButton.Click += async (_, _) => await RefreshCamerasAsync();
        _startButton.Click += (_, _) => StartPipeline();
        _stopButton.Click += (_, _) => StopPipeline();
        _saveButton.Click += async (_, _) => await SaveCurrentFrameAsync();
        _pickBgImageButton.Click += (_, _) => PickBackgroundImage();
        _pickBgVideoButton.Click += (_, _) => PickBackgroundVideo();
        _useGpuBox.CheckedChanged += (_, _) => TryLoadModel(forceReload: true);
    }

    private async Task RefreshCamerasAsync()
    {
        _refreshButton.Enabled = false;
        SetStatus("Refreshing cameras...");
        try
        {
            var cameras = await Task.Run(CameraDiscovery.FindAvailableCameras);
            _cameraList.Items.Clear();
            foreach (var camera in cameras)
            {
                _cameraList.Items.Add(camera);
            }

            if (_cameraList.Items.Count > 0)
            {
                _cameraList.SelectedIndex = 0;
                SetStatus($"Camera candidates ready: {_cameraList.Items.Count}. Select one and click Start Preview.");
            }
            else
            {
                SetStatus("No camera candidates found.");
            }
        }
        catch (Exception ex)
        {
            SetStatus($"Camera refresh failed: {ex.Message}");
        }
        finally
        {
            _refreshButton.Enabled = true;
        }
    }

    private void TryLoadModel(bool forceReload = false)
    {
        if (!forceReload && _matting != null)
        {
            return;
        }

        _matting?.Dispose();
        _matting = null;

        var modelPath = Directory.GetFiles(_modelDir, "*.onnx").OrderBy(p => p.Contains("mobilenet", StringComparison.OrdinalIgnoreCase) ? 0 : 1).FirstOrDefault();
        if (modelPath == null)
        {
            SetStatus($"ONNX model not found. Put it in {Path.GetFullPath(_modelDir)}. Camera preview still works.");
            return;
        }

        try
        {
            _matting = new RvmOnnxMatting(modelPath, _useGpuBox.Checked);
            SetStatus($"Model loaded: {Path.GetFileName(modelPath)} | Provider: {_matting.ProviderName}");
        }
        catch (Exception ex)
        {
            SetStatus($"Model load failed: {ex.Message}");
        }
    }

    private void StartPipeline()
    {
        if (_running || _cameraList.SelectedItem is not CameraInfo camera)
        {
            return;
        }

        TryLoadModel();
        _running = true;
        _cts = new CancellationTokenSource();
        _captureTask = Task.Run(() => CaptureLoop(camera, _cts.Token));
        _inferenceTask = Task.Run(() => InferenceLoop(_cts.Token));
        _startButton.Enabled = false;
        _stopButton.Enabled = true;
        _saveButton.Enabled = true;
    }

    private void StopPipeline()
    {
        if (!_running)
        {
            return;
        }

        _running = false;
        _cts?.Cancel();
        try
        {
            Task.WaitAll(new[] { _captureTask, _inferenceTask }.Where(t => t != null).Cast<Task>().ToArray(), TimeSpan.FromSeconds(2));
        }
        catch
        {
            // Stopping camera devices can throw when the driver is already gone.
        }

        _cts?.Dispose();
        _cts = null;
        _captureTask = null;
        _inferenceTask = null;

        lock (_latestLock)
        {
            _latestFrame?.Dispose();
            _latestFrame = null;
            _latestComposited?.Dispose();
            _latestComposited = null;
        }

        _startButton.Enabled = true;
        _stopButton.Enabled = false;
        _saveButton.Enabled = false;
        SetStatus("Stopped.");
    }

    private void CaptureLoop(CameraInfo camera, CancellationToken token)
    {
        using var cap = OpenCamera(camera, out var backendName);
        BeginInvoke(() => SetStatus($"Opening camera {camera.Index} via {backendName}..."));
        if (cap == null || !cap.IsOpened())
        {
            BeginInvoke(() => SetStatus($"Camera open failed: {camera.Name}. Try another index or close other camera apps."));
            BeginInvoke(() =>
            {
                _startButton.Enabled = true;
                _stopButton.Enabled = false;
                _saveButton.Enabled = false;
            });
            _running = false;
            return;
        }

        cap.Set(VideoCaptureProperties.FourCC, FourCC.FromFourChars('M', 'J', 'P', 'G'));
        cap.Set(VideoCaptureProperties.FrameWidth, (double)_widthInput.Value);
        cap.Set(VideoCaptureProperties.FrameHeight, (double)_heightInput.Value);
        cap.Set(VideoCaptureProperties.Fps, 30);
        cap.Set(VideoCaptureProperties.BufferSize, 1);
        BeginInvoke(() => SetStatus($"Camera opened: {camera.Name} | Backend {backendName}"));

        using var frame = new Mat();
        while (!token.IsCancellationRequested)
        {
            if (!cap.Read(frame) || frame.Empty())
            {
                Thread.Sleep(5);
                continue;
            }

            lock (_latestLock)
            {
                _latestFrame?.Dispose();
                _latestFrame = frame.Clone();
            }

            Interlocked.Increment(ref _captureFrames);
        }
    }

    private static VideoCapture? OpenCamera(CameraInfo camera, out string backendName)
    {
        foreach (var candidate in new[] { VideoCaptureAPIs.DSHOW, VideoCaptureAPIs.MSMF, VideoCaptureAPIs.ANY })
        {
            VideoCapture? cap = null;
            try
            {
                cap = new VideoCapture(camera.Index, candidate);
                if (cap.IsOpened())
                {
                    backendName = candidate.ToString();
                    return cap;
                }
            }
            catch
            {
                cap?.Dispose();
                continue;
            }

            cap.Dispose();
        }

        backendName = "None";
        return null;
    }

    private void InferenceLoop(CancellationToken token)
    {
        var localClock = Stopwatch.StartNew();
        Mat? lastSeen = null;
        while (!token.IsCancellationRequested)
        {
            Mat? frame = null;
            lock (_latestLock)
            {
                if (_latestFrame != null && !ReferenceEquals(_latestFrame, lastSeen))
                {
                    frame = _latestFrame.Clone();
                    lastSeen = _latestFrame;
                }
            }

            if (frame == null)
            {
                Thread.Sleep(2);
                continue;
            }

            using (frame)
            {
                var previewFrame = ResizeLongest(frame, (int)_previewMaxInput.Value);
                using (previewFrame)
                {
                    Mat output;
                    if (_matting != null)
                    {
                        output = _matting.Matte(previewFrame, (float)_previewDownsampleInput.Value, bg => PrepareBackground(bg, previewFrame.Width, previewFrame.Height));
                    }
                    else
                    {
                        output = previewFrame.Clone();
                    }

                    lock (_latestLock)
                    {
                        _latestComposited?.Dispose();
                        _latestComposited = output.Clone();
                    }

                    PostPreview(output);
                    output.Dispose();
                    Interlocked.Increment(ref _inferFrames);
                }
            }

            if (localClock.ElapsedMilliseconds > 500)
            {
                PostStats();
                localClock.Restart();
            }
        }
    }

    private void PostPreview(Mat bgr)
    {
        try
        {
            var bitmap = BitmapConverter.ToBitmap(bgr);
            BeginInvoke(() =>
            {
                var old = _preview.Image;
                _preview.Image = bitmap;
                old?.Dispose();
            });
        }
        catch
        {
            // Preview must not kill the capture loop.
        }
    }

    private void PostStats()
    {
        if (_statsClock.Elapsed.TotalSeconds < 1)
        {
            return;
        }

        var seconds = _statsClock.Elapsed.TotalSeconds;
        var captureFps = Interlocked.Exchange(ref _captureFrames, 0) / seconds;
        var inferFps = Interlocked.Exchange(ref _inferFrames, 0) / seconds;
        _statsClock.Restart();

        Mat? frame;
        lock (_latestLock)
        {
            frame = _latestFrame;
        }

        var actual = frame == null ? "No frame" : $"{frame.Width}x{frame.Height}";
        BeginInvoke(() => SetStatus($"Input {actual} | Capture {captureFps:F1} FPS | Preview Infer {inferFps:F1} FPS | Provider {(_matting?.ProviderName ?? "No model")}"));
    }

    private async Task SaveCurrentFrameAsync()
    {
        _saveButton.Enabled = false;
        SetStatus("Saving 4K PNG. Full-resolution inference is slower than preview...");
        try
        {
            Mat? frame;
            lock (_latestLock)
            {
                frame = _latestFrame?.Clone();
            }

            if (frame == null || frame.Empty())
            {
                SetStatus("No camera frame to save.");
                return;
            }

            using (frame)
            {
                var output = await Task.Run(() =>
                {
                    if (_matting == null)
                    {
                        return frame.Clone();
                    }

                    using var saver = _matting.CreateIndependentSession();
                    return saver.Matte(frame, (float)_saveDownsampleInput.Value, bg => PrepareBackground(bg, frame.Width, frame.Height));
                });

                using (output)
                {
                    var path = Path.Combine(_captureDir, $"native_rvm_4k_{DateTime.Now:yyyyMMdd_HHmmss}.png");
                    Cv2.ImWrite(path, output);
                    SetStatus($"Saved: {path}");
                }
            }
        }
        catch (Exception ex)
        {
            SetStatus($"Save failed: {ex.Message}");
        }
        finally
        {
            _saveButton.Enabled = _running;
        }
    }

    private void PickBackgroundImage()
    {
        using var dialog = new OpenFileDialog { Filter = "Image|*.jpg;*.jpeg;*.png;*.bmp|All files|*.*" };
        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        var img = Cv2.ImRead(dialog.FileName, ImreadModes.Color);
        if (img.Empty())
        {
            SetStatus("Background image load failed.");
            return;
        }

        lock (_backgroundLock)
        {
            _backgroundImage?.Dispose();
            _backgroundImage = img;
        }

        _backgroundMode.SelectedIndex = 1;
        SetStatus($"Background image loaded: {dialog.FileName}");
    }

    private void PickBackgroundVideo()
    {
        using var dialog = new OpenFileDialog { Filter = "Video|*.mp4;*.mov;*.avi;*.mkv|All files|*.*" };
        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        var video = new VideoCapture(dialog.FileName);
        if (!video.IsOpened())
        {
            video.Dispose();
            SetStatus("Background video open failed.");
            return;
        }

        lock (_backgroundLock)
        {
            _backgroundVideo?.Dispose();
            _backgroundVideo = video;
        }

        _backgroundMode.SelectedIndex = 2;
        SetStatus($"Background video loaded: {dialog.FileName}");
    }

    private Mat PrepareBackground(Mat reusable, int width, int height)
    {
        var mode = _backgroundMode.SelectedIndex;
        lock (_backgroundLock)
        {
            if (mode == 1 && _backgroundImage != null)
            {
                Cv2.Resize(_backgroundImage, reusable, new CvSize(width, height), 0, 0, InterpolationFlags.Linear);
                return reusable;
            }

            if (mode == 2 && _backgroundVideo != null)
            {
                using var frame = new Mat();
                if (!_backgroundVideo.Read(frame) || frame.Empty())
                {
                    _backgroundVideo.Set(VideoCaptureProperties.PosFrames, 0);
                    _backgroundVideo.Read(frame);
                }

                if (!frame.Empty())
                {
                    Cv2.Resize(frame, reusable, new CvSize(width, height), 0, 0, InterpolationFlags.Linear);
                    return reusable;
                }
            }
        }

        reusable.Create(height, width, MatType.CV_8UC3);
        reusable.SetTo(new Scalar(_backgroundColor.B, _backgroundColor.G, _backgroundColor.R));
        return reusable;
    }

    private static Mat ResizeLongest(Mat src, int maxSide)
    {
        var longest = Math.Max(src.Width, src.Height);
        if (longest <= maxSide)
        {
            return src.Clone();
        }

        var scale = maxSide / (double)longest;
        var dst = new Mat();
        Cv2.Resize(src, dst, new CvSize((int)Math.Round(src.Width * scale), (int)Math.Round(src.Height * scale)), 0, 0, InterpolationFlags.Area);
        return dst;
    }

    private void SetStatus(string text)
    {
        _status.Text = "  " + text;
    }
}

public sealed record CameraInfo(int Index, string Name)
{
    public override string ToString() => $"{Name}  [Index {Index}]";
}

public static class CameraDiscovery
{
    public static List<CameraInfo> FindAvailableCameras()
    {
        var names = FindDirectShowCameraNames();
        if (names.Count == 0)
        {
            names = FindCameraNames();
        }

        var result = new List<CameraInfo>();
        var count = Math.Max(10, names.Count);
        for (var i = 0; i < count; i++)
        {
            var name = i < names.Count ? names[i] : $"Camera {i}";
            result.Add(new CameraInfo(i, name));
        }

        return result;
    }

    private static List<string> FindDirectShowCameraNames()
    {
        try
        {
            return DsDevice
                .GetDevicesOfCat(FilterCategory.VideoInputDevice)
                .Select(d => d.Name)
                .Where(n => !string.IsNullOrWhiteSpace(n))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToList();
        }
        catch
        {
            return [];
        }
    }

    private static List<string> FindCameraNames()
    {
        var names = new List<string>();
        try
        {
            using var searcher = new ManagementObjectSearcher("SELECT Name FROM Win32_PnPEntity WHERE PNPClass='Camera' OR PNPClass='Image'");
            foreach (var item in searcher.Get())
            {
                var name = item["Name"]?.ToString();
                if (!string.IsNullOrWhiteSpace(name))
                {
                    names.Add(name);
                }
            }
        }
        catch
        {
            // WMI can be disabled; OpenCV index probing still works.
        }

        return names.Distinct(StringComparer.OrdinalIgnoreCase).ToList();
    }
}

public sealed class RvmOnnxMatting : IDisposable
{
    private readonly string _modelPath;
    private readonly bool _useGpu;
    private readonly InferenceSession _session;
    private DenseTensor<float>[] _rec = CreateInitialRec();
    private readonly string[] _inputNames;

    public string ProviderName { get; }

    public RvmOnnxMatting(string modelPath, bool useGpu)
    {
        _modelPath = modelPath;
        _useGpu = useGpu;
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern = false
        };

        if (useGpu)
        {
            options.AppendExecutionProvider_DML(0);
            ProviderName = "DirectML";
        }
        else
        {
            ProviderName = "CPU";
        }

        _session = new InferenceSession(modelPath, options);
        _inputNames = _session.InputMetadata.Keys.ToArray();
    }

    public RvmOnnxMatting CreateIndependentSession() => new(_modelPath, _useGpu);

    public Mat Matte(Mat bgr, float downsampleRatio, Func<Mat, Mat> backgroundFactory)
    {
        using var bg = new Mat();
        backgroundFactory(bg);

        var srcTensor = ToTensor(bgr);
        var downsample = new DenseTensor<float>(new[] { Math.Clamp(downsampleRatio, 0.05f, 1f) }, [1]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName(0, "src"), srcTensor),
            NamedOnnxValue.CreateFromTensor(InputName(1, "r1i"), _rec[0]),
            NamedOnnxValue.CreateFromTensor(InputName(2, "r2i"), _rec[1]),
            NamedOnnxValue.CreateFromTensor(InputName(3, "r3i"), _rec[2]),
            NamedOnnxValue.CreateFromTensor(InputName(4, "r4i"), _rec[3]),
            NamedOnnxValue.CreateFromTensor(InputName(5, "downsample_ratio"), downsample)
        };

        using var results = _session.Run(inputs);
        var resultList = results.ToList();
        var fgr = resultList[0].AsTensor<float>();
        var pha = resultList[1].AsTensor<float>();
        _rec = resultList.Skip(2).Take(4).Select(v => CloneTensor(v.AsTensor<float>())).ToArray();
        return Composite(fgr, pha, bg);
    }

    private string InputName(int index, string fallback)
    {
        if (_inputNames.Length > index)
        {
            return _inputNames[index];
        }

        return fallback;
    }

    private static DenseTensor<float>[] CreateInitialRec()
    {
        return
        [
            new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
            new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
            new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
            new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1])
        ];
    }

    private static DenseTensor<float> CloneTensor(Tensor<float> tensor)
    {
        var dims = tensor.Dimensions.ToArray();
        var clone = new DenseTensor<float>(dims);
        var src = tensor.ToArray();
        src.CopyTo(clone.Buffer.Span);
        return clone;
    }

    private static DenseTensor<float> ToTensor(Mat bgr)
    {
        var height = bgr.Height;
        var width = bgr.Width;
        var tensor = new DenseTensor<float>([1, 3, height, width]);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var pixel = bgr.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = pixel.Item2 / 255f;
                tensor[0, 1, y, x] = pixel.Item1 / 255f;
                tensor[0, 2, y, x] = pixel.Item0 / 255f;
            }
        }

        return tensor;
    }

    private static Mat Composite(Tensor<float> fgr, Tensor<float> pha, Mat bg)
    {
        var height = bg.Height;
        var width = bg.Width;
        var output = new Mat(height, width, MatType.CV_8UC3);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var alpha = Math.Clamp(pha[0, 0, y, x], 0f, 1f);
                var bgPixel = bg.At<Vec3b>(y, x);
                var fr = Math.Clamp(fgr[0, 0, y, x], 0f, 1f) * 255f;
                var fg = Math.Clamp(fgr[0, 1, y, x], 0f, 1f) * 255f;
                var fb = Math.Clamp(fgr[0, 2, y, x], 0f, 1f) * 255f;
                var b = (byte)Math.Clamp(fb * alpha + bgPixel.Item0 * (1f - alpha), 0f, 255f);
                var g = (byte)Math.Clamp(fg * alpha + bgPixel.Item1 * (1f - alpha), 0f, 255f);
                var r = (byte)Math.Clamp(fr * alpha + bgPixel.Item2 * (1f - alpha), 0f, 255f);
                output.Set(y, x, new Vec3b(b, g, r));
            }
        }

        return output;
    }

    public void Dispose()
    {
        _session.Dispose();
    }
}
