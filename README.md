
<h1 align="center">
  <br>
   <a href="https://etiennelorthoy.com">Interop CUDA vs C vs C# Benchmark in C# Runtime</a>
  <br>
</h1>

At which point is it usefull to leverage CUDA cores to accelerate processing from a managed C# runtime ? This project try to answer this question:

```bash
Start C#
C# (ms): 461.4153
Data 32 set:60.400001525878906 test:60.400001525878906 validated:True

Start CUDA GPU
        Start CUDA 0 ms
        Finit CUDA 300 ms
CUDA (ms): 300.4521
Data 32 set:60.400000000000006 test:60.400001525878906 validated:True

Start C CPU
        Start unmanaged C +0 ms
        Unmanaged C stopped +312 ms
C (ms): 612.7176
Data 32 set:60.400000000000006 test:60.400001525878906 validated:True

Ctrl + C to exit...
```

<p align="center">
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/) then [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64) installed on your computer.

```bash
# Clone this repository
$ git clone https://github.com/EtienneLorthoy/interop-cuda-csharp-cpp-benchmark

# Go into the repository
$ ./InteropCudaCSharpBenchmark.sln

# Run F5 or Ctrl + F5
```

## Credits

This software uses the following open dependencies:

- [Visual Studio 2022 Community](https://visualstudio.microsoft.com/downloads/)
- [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
- [gitingore.io](https://www.toptal.com/developers/gitignore)

## License

MIT

---

> [etiennelorthoy.com](https://etiennelorthoy.com) &nbsp;&middot;&nbsp;
> LinkedIn [@etiennelorthoy](https://www.linkedin.com/in/etienne-lorthoy/) &nbsp;&middot;&nbsp;
> GitHub [@etiennelorthoy](https://github.com/EtienneLorthoy) &nbsp;&middot;&nbsp;
> Twitter [@ELorthoy](https://twitter.com/ELorthoy)
