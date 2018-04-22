using System;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Logger;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Class which represents max pooling layer
    /// </summary>
    [Serializable]
    public class AveragePooling2DLayer : AbstractLayer
    {
        #region Constants
        /// <summary>
        /// Default filter size
        /// </summary>
        private const int DEFAULT_FILTER_SIZE = 2;
        /// <summary>
        /// Default stride of filter same as in conv layer
        /// </summary>
        private const int DEFAULT_STRIDE = 2;
        #endregion

        #region Local variables
        /// <summary>
        /// Default stride of filter in conv layer
        /// </summary>
        private int filterSize;
        /// <summary>
        /// Size of stride which is used
        /// </summary>
        private int stride;
        /// <summary>
        /// Output weight and height
        /// </summary>
        private int outputSize;
        /// <summary>
        /// Value which holding size of side original image
        /// which was used during forward step
        /// </summary>
        private int origImageSize;
        #endregion

        /// <summary>
        /// Constructor for init
        /// </summary>
        /// <param name="filterSize">size of filter which will be used</param>
        /// <param name="stride">stride which will be used</param>
        public AveragePooling2DLayer(Dimension inputDimension = null, int filterSize = DEFAULT_FILTER_SIZE, int stride = DEFAULT_STRIDE)
        {
            this.filterSize = filterSize;
            this.stride = stride;
            this.inputDimension = inputDimension;
            layerName = "Average Pooling";
        }

        public override double[][][][] ForwardPropagation(double[][][][] input, int startIndex, int endIndex, bool predict = false)
        {
            #region Test hyperparameters
            int currImageCount = input.Length;
            int currDepth = input[0].Length;
            int currWidth = input[0][0].Length;
            int currHeight = input[0][0][0].Length;

            if (!IsInputDataSameAsDim(currDepth, currWidth, currHeight))
            {
                string msg = "Input data have not same dimension as initialized dimension in convolution layer.";
                Utils.ThrowException(msg);
            }

            if (startIndex == DEFAULT_INDEX && endIndex == startIndex)
            {
                startIndex = 0;
                endIndex = currImageCount;
            }
            else
            {
                currImageCount = endIndex - startIndex;
            }

            TestHyperParametersAndSetOutputSize(currWidth);
            #endregion

            // save original image size
            origImageSize = input[0][0].Length;

            // init output arrays with 0's
            activations = Utils.Init4dArr(currImageCount, currDepth, outputSize, outputSize);

            // indexes used in output array
            int xIndex, yIndex;
            // temp for storing current maximum
            double sum;
            for (int imageIndex = 0; imageIndex < currImageCount; imageIndex++)
            {
                // iterate through all filters
                for (int filterIndex = 0; filterIndex < currDepth; filterIndex++)
                {
                    yIndex = 0;
                    // iterate through height of current image according to stride
                    for (int y = 0; y < currHeight - filterSize + 1; y += stride)
                    {
                        xIndex = 0;
                        // iterate through width of current image according to stride
                        for (int x = 0; x < currWidth - filterSize + 1; x += stride)
                        {
                            // iterate through all channels of current pixel
                            for (int channel = 0; channel < currDepth; channel++)
                            {
                                sum = 0;
                                for (int yCurr = y; yCurr < y + filterSize; yCurr++)
                                {
                                    for (int xCurr = x; xCurr < x + filterSize; xCurr++)
                                    {
                                        sum += input[imageIndex][channel][xCurr][yCurr];

                                    }
                                }

                                activations[imageIndex][channel][xIndex][yIndex] = sum / (filterSize * filterSize);
                            }
                            xIndex++;
                        }
                        yIndex++;
                    }
                }
            }

            return activations;
        }

        public override double[][][][] BackwardPropagation(double[][][][] input, int startIndex, int endIndex)
        {
            int currImageCount = input.Length;
            int currDepth = input[0].Length;
            int currWidth = input[0][0].Length;
            int currHeight = input[0][0][0].Length;

            double[][][][] outputGradient = Utils.Init4dArr(currImageCount, currDepth, origImageSize, origImageSize);

            for (int i = 0; i < currImageCount; i++)
            {
                for (int j = 0; j < currDepth; j++)
                {
                    for (int y = 0; y < currHeight; y++)
                    {
                        for (int x = 0; x < currWidth; x++)
                        {
                            for (int currY = 0; currY < filterSize; currY++)
                            {
                                for (int currX = 0; currX < filterSize; currX++)
                                {
                                    outputGradient[i][j][x * filterSize + currX][y * filterSize + currY] = input[i][j][x][y];
                                }
                            }
                        }
                    }
                }
            }
            return outputGradient;
        }

        public override long Summary()
        {
            int depth = activations[0].GetLength(0);
            int width = activations[0][0].GetLength(0);
            int height = activations[0][0].GetLength(0);

            string output = String.Format("{0,-15} {1, -30} {2, -45}", layerName.ToString(),
                "(" + "None" + ", " + width + ", " + height
                + ", " + depth + ")", "0");
            Log.Info(output);

            return 0;
        }

        #region Test input parameters
        /// <summary>
        /// Test if is possible to process maxpooling
        /// </summary>
        /// <param name="size">size of W and H in input data</param>
        private void TestHyperParametersAndSetOutputSize(int size)
        {
            double diff = (double)(size - filterSize) / (double)stride + 1;

            if (diff % Math.Floor(diff) != 0)
            {
                string msg = "Invalid hyperparameters in pooling layer.";
                Utils.ThrowException(msg);
            }

            outputSize = (int)diff;

            // dimension is same for the max pooling layer
        }
        #endregion

        public override AbstractLayer Compile(AbstractLayer previousLayer)
        {
            if (previousLayer == null && this.inputDimension == null)
            {
                string msg = "Dimension of first layer of network not set!";
                Utils.ThrowException(msg);
            }
            else if (inputDimension == null)
            {
                inputDimension = previousLayer.outputDimension;
                index = 0;
            }
            else if (previousLayer != null)
            {
                inputDimension = previousLayer.outputDimension;
                index = previousLayer.Index + 1;
            }

            TestHyperParametersAndSetOutputSize(inputDimension.width);

            outputDimension = new Dimension(inputDimension.imageCount, inputDimension.depth,
                 outputSize, outputSize);

            compiled = true;

            return this;
        }
    }
}