using System;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Logger;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Layer which flatten input during the forward step
    /// (e.g. (10, 3, 5, 5) -> (10, 1, 1, 75)) and unflatten
    /// input during backward step
    /// </summary>
    [Serializable]
    public class FlattenLayer : AbstractLayer
    {
        /// <summary>
        /// Variable which contains number of neurons in dense 
        /// layer we re using local variable because we want 
        /// to use it in summary.
        /// </summary>
        private int size;
        /// <summary>
        /// Last number of samples in forward step.
        /// </summary>
        private int lastNumSamples;
        /// <summary>
        /// Last number of filters/depth in forward step.
        /// </summary>
        private int lastNumFilters;
        /// <summary>
        /// Last number of rows in forward step.
        /// </summary>
        private int lastNumRows;
        /// <summary>
        /// Last number of cols in forward step.
        /// </summary>
        private int lastNumCols;

        public FlattenLayer(Dimension inputDimension = null)
        {
            this.inputDimension = inputDimension;
        }

        /// <summary>
        /// From input 4D array creates 2D array where first
        /// dimension is number of samples, so rest of the
        /// values is included behind values - we need use
        /// dense layers as ordinary neural network with 1D
        /// input, respectively 2D (samples, input)
        /// </summary>
        /// <param name="input">Input into layer</param>
        /// <returns>2D array (samples, input)</returns>
        public override double[][][][] ForwardPropagation(double[][][][] input, int startIndex, int endIndex, bool predict = false)
        {
            int numSamples = input.GetLength(0);
            lastNumSamples = numSamples;
            lastNumFilters = inputDimension.depth;
            lastNumRows = inputDimension.height;
            lastNumCols = inputDimension.width;
            if (!IsInputDataSameAsDim(lastNumFilters, lastNumCols, lastNumRows))
            {
                string msg = "Input data have not same dimension as initialized dimension in flatten layer.";
                Utils.ThrowException(msg);
            }
            double[][][][] flatten = Utils.Init4dArr(numSamples, 1, 1, size);
            int index;
            for (int sample = 0; sample < numSamples; sample++)
            {
                index = 0;
                for (int i = 0; i < inputDimension.depth; i++)
                {
                    for (int j = 0; j < inputDimension.width; j++)
                    {
                        for (int k = 0; k < inputDimension.height; k++)
                        {
                            flatten[sample][0][0][index++] = input[sample][i][j][k];
                        }
                    }
                }
            }

            return flatten;
        }

        public override double[][][][] BackwardPropagation(double[][][][] input, int startIndex, int endIndex)
        {
            double[][][][] unflatten = Utils.Init4dArr(lastNumSamples, inputDimension.depth, 
                inputDimension.width, inputDimension.height);
            int index;
            for (int sample = 0; sample < input.Length; sample++)
            {
                index = 0;
                for (int i = 0; i < lastNumFilters; i++)
                {
                    for (int j = 0; j < lastNumCols; j++)
                    {
                        for (int k = 0; k < lastNumRows; k++)
                            unflatten[sample][i][j][k] = input[sample][0][0][index++];
                    }
                }
            }
        
            return unflatten;
        }

        public override long Summary()
        {
            string output = String.Format("{0,-15} {1, -30} {2, -45}", "Flatten",
                "(" + "None" + ", 1, 1, " + size + ")", "0");
            Log.Info(output);
            // flatten layer only flatten input and have not any weights
            return 0L;
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


            size = inputDimension.depth * inputDimension.height * inputDimension.width;
            outputDimension = new Dimension(inputDimension.imageCount, 1, 1, size);

            compiled = true;

            return this;
        }
    }
}