# Convsharp
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

The Convsharp library is simple library written in C# which provides simple tool for creating and using basic convolutional neural networks. Convsharp is aimed on readable code which is especially for the simple tasks where user want to see how CNN works due to low performance and architecture possibilities in comparison with libraries as Tensorflow, PyTorch, CNTK, etc. Library contains following features:

* 1D and 2D convolutional layer
* max-pooling and average pooling 2D layer and 1d max-pooling layer
* regularization techniques like dropout or L2-normalization
* optimalization techniques like mini-batch gradient descent or Adam
* activation functions like ReLU, Tanh or Sigmoid

For more information visit page with documentation on github pages: https://mmedek.github.io/convsharp/index.html

## Installing

To use Convsharp in your project, please use NuGet package. For the installation go through following steps:

1. In Microsoft Visual Studio open your project
2. Right-click on the 'Dependencies' in Solution Explorer
3. Select item 'Manage NuGet packages...'
4. Click on the 'Browse' tab
5. Type 'Consharp' and hit ENTER
6. Select Convsharp library
7. Click on 'Install' button

Next way how to install library is open 'Package Manager Console' in Microsoft Visual Studio and type 'Install-Package Convsharp -Version 1.0.0'

### Examples

For quick start you can check examples placed in repository. There are several examples how to use library.

## Building

In Microsoft Visual Studio 2017 open '/src/Convsharp.csproj'.

## Contributing

Every new contributer is welcome. In the future I would like to speed up library a lot with using vector instructions and replacing implementation of convolution with im2col approach. There is still plenty of work :)
