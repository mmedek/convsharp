using System;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Logger;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Dropout layer which can be used for flatten layers as
    /// regularization technique
    /// </summary>
    [Serializable]
    public class DropoutLayer : AbstractLayer
    {
        /// <summary>
        /// Probability of truning off neuron in layer.
        /// </summary>
        private double probability;
        /// <summary>
        /// Mask of truned off neurons.
        /// </summary>
        private double[][][] mask;
        /// <summary>
        /// Number of neurons in this layer.
        /// </summary>
        private int numNeurons;

        public DropoutLayer(double probability)
        {
            this.probability = probability;
            layerName = "Dropout";
        }

        public override double[][][][] ForwardPropagation(double[][][][] input, int startIndex = 0, int endIndex = 0, bool predict = false)
        {
            int currImageCount = input.Length;
            int currDepth = input[0].Length;
            int currWidth = input[0][0].Length;
            int currHeight = input[0][0][0].Length;

            numNeurons = currDepth * currWidth * currHeight;
            double zeros = numNeurons * probability;

            mask = GenerateFlattenMask(numNeurons, zeros, probability);

            for (int i = 0; i < currImageCount; i++)
            {
                for (int l = 0; l < currHeight; l++)
                    input[i][0][0][l] *= mask[0][0][l];
            }

            return input;
        }

        public override double[][][][] BackwardPropagation(double[][][][] input, int startIndex = 0, int endIndex = 0)
        {
            int currImageCount = input.Length;
            int currDepth = input[0].Length;
            int currWidth = input[0][0].Length;
            int currHeight = input[0][0][0].Length;

            for (int i = 0; i < currImageCount; i++)
            {
                for (int j = 0; j < currDepth; j++)
                {
                    for (int k = 0; k < currWidth; k++)
                    {
                        for (int l = 0; l < currHeight; l++)
                            input[i][0][0][l] *= mask[0][0][l];
                    }
                }
            }

            return input;
        }

        /// <summary>
        /// Build the mask for dropout which is used during backpropagation.
        /// </summary>
        /// <param name="values">Number of values in layer</param>
        /// <param name="zeros">Number of turn off neurons</param>
        /// <param name="probability">Probability of turning off neurons</param>
        /// <returns>Dropout mask</returns>
        private double[][][] GenerateFlattenMask(int values, double zeros, double probability)
        {
            double[][][] arr = new double[1][][];
            arr[0] = new double[1][];
            arr[0][0] = new double[values];
            for (int x = 0; x < values; x++)
                arr[0][0][x] = 1 / probability;

            int i = 0;
            int currzeros = 0;
            while (zeros < currzeros)
            {
                i = Utils.GetRandomInt(0, values);
                if (arr[0][0][i] != 0)
                {
                    arr[0][0][i] = 0;
                    currzeros++;
                }
            }
            return arr;
        }

        public override long Summary()
        {
            string output = String.Format("{0,-15} {1, -30} {2, -45}", layerName.ToString(),
                "(" + "None" + ", 1, 1, " + numNeurons + ")",
                "0");
            Log.Info(output);
            return 0L;
        }

        public override AbstractLayer Compile(AbstractLayer previousLayer)
        {
            if (previousLayer == null)
            {
                string msg = "Dropout layer cannot be first layer!";
                Utils.ThrowException(msg);
            }

            if (previousLayer.outputDimension.depth != 1
                || previousLayer.outputDimension.width != 1)
            {
                string msg = "Dropout layer is adapted only for flatten layers!";
                Utils.ThrowException(msg);
            }

            index = previousLayer.Index + 1;
            compiled = true;
            outputDimension = previousLayer.outputDimension;
            return this;
        }
    }
}