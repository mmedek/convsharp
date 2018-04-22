﻿using System;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Logger;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Class which represents max pooling layer
    /// for 1D data
    /// </summary>
    [Serializable]
    public class MaxPooling1DLayer : AbstractLayer
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
        /// Multidimensional array which holds information
        /// about item which was selected as maximum
        /// on that item is value equal to 1 on others to 0
        /// </summary>
        private int[][][][][] lastSwitches;
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
        public MaxPooling1DLayer(Dimension inputDimension = null, int filterSize = DEFAULT_FILTER_SIZE, int stride = DEFAULT_STRIDE)
        {
            this.filterSize = filterSize;
            this.stride = stride;
            this.inputDimension = inputDimension;
            layerName = "Max Pooling";
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
                string msg = "Input data have not same dimension as initialized dimension in pooling layer.";
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

            TestHyperParametersAndSetOutputSize(currHeight);
            #endregion

            // save original image size
            origImageSize = input[0][0][0].Length;

            // init output arrays with 0's
            activations = Utils.Init4dArr(currImageCount, currDepth, 1, outputSize);
            // init switches arrays with 0's
            lastSwitches = Utils.InitInt5dArr(currImageCount, currDepth, 1, outputSize, 1);
            // indexes used in output array
            int yIndex;
            // temp for storing current maximum
            double max;
            int yMax;
            for (int imageIndex = 0; imageIndex < currImageCount; imageIndex++)
            {
                // iterate through all filters
                for (int filterIndex = 0; filterIndex < currDepth; filterIndex++)
                {
                    yIndex = 0;
                    // iterate through height of current image according to stride
                    for (int y = 0; y < currHeight - filterSize + 1; y += stride)
                    {
                        // iterate through all channels of current pixel
                        for (int channel = 0; channel < currDepth; channel++)
                        {
                            max = Double.MinValue;
                            yMax = -1;
                            for (int yCurr = y; yCurr < y + filterSize; yCurr++)
                            {
                                if (max < input[imageIndex][channel][0][yCurr])
                                {
                                    max = input[imageIndex][channel][0][yCurr];
                                    yMax = yCurr;
                                }
                            }
                            activations[imageIndex][channel][0][yIndex] = max;
                            lastSwitches[imageIndex][channel][0][yIndex][0] = yMax;
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

            double[][][][] outputGradient = Utils.Init4dArr(currImageCount, currDepth, 1, origImageSize);

            int coorY;
            for (int i = 0; i < currImageCount; i++)
            {
                for (int j = 0; j < currDepth; j++)
                {
                    for (int y = 0; y < currHeight; y++)
                    {
                        coorY = lastSwitches[i][j][0][y][0];
                        outputGradient[i][j][0][coorY] = input[i][j][0][y];
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
                "(" + "None" + ", " + width + ", " + height + ", " + depth + ")", "0");
            Log.Info(output);

            return 0;
        }

        public override AbstractLayer Compile(AbstractLayer previousLayer)
        {
            if (previousLayer == null && this.inputDimension == null)
            {
                string msg = "Dimension of first layer of network not set!";
                Utils.ThrowException(msg);
            }
            else if (previousLayer != null)
            {
                inputDimension = previousLayer.outputDimension;
                index = previousLayer.Index + 1;
            }

            TestHyperParametersAndSetOutputSize(inputDimension.height);
            outputDimension = new Dimension(inputDimension.imageCount, inputDimension.depth,
                 inputDimension.width, outputSize);

            compiled = true;

            return this;
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
                string msg = "Invalid hyperparameters in pooling layer. " +
                    "Only odd filter size is supported.";
                Utils.ThrowException(msg);
            }

            outputSize = (int)diff;

            // height is same for the max pooling layer
        }
        #endregion
    }
}