using System;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Layers.ActivationFunctions;
using Zcu.Convsharp.Logger;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Abstract layer for all activation functions. This layer
    /// will be usually used after liner layer, but can be used
    /// after convolutional layer or max pooling as well.
    /// </summary>
    [Serializable]
    public class ActivationLayer : AbstractLayer
    {
        #region Local variables
        /// <summary>
        /// Instance of activation functions which
        /// will be used for computing activation
        /// of neurons
        /// </summary>
        ActivationFunction activationFunction;
        /// <summary>
        /// Depth of this layer.
        /// </summary>
        private int depth;
        /// <summary>
        /// Height of the multidimension array for
        /// this layer.
        /// </summary>
        private int height;
        /// <summary>
        /// Width of the multidimension array for
        /// this layer.
        /// </summary>
        private int width;
        #endregion Local variables

        /// <summary>
        /// Constructor for creating new dense layer
        /// </summary>
        /// <param name="activationFunction">type of activation function</param>
        /// <param name="numNeurons">number of neurons which will be used</param>
        public ActivationLayer(ActivationFunction activationFunction, Dimension inputDimension = null)
        {
            this.activationFunction = activationFunction;
            this.inputDimension = inputDimension;
            layerName = "Activation";
        }

        public override double[][][][] ForwardPropagation(double[][][][] input, int startIndex = 0, int endIndex = 0, bool predict = false)
        {
            int numSamples = input.Length;
            depth = input[0].Length;
            width = input[0][0].Length;
            height = input[0][0][0].Length;
            if (!IsInputDataSameAsDim(depth, width, height))
            {
                string msg = "Input data have not same dimension as initialized dimension in activation layer.";
                Utils.ThrowException(msg);
            }
            Dimension dim = new Dimension(numSamples, depth, width, height);
            activations = Utils.Init4dArr(numSamples, depth, width, height);
            activations = activationFunction.Compute(input);

            return activations;
        }

        public override double[][][][] BackwardPropagation(double[][][][] input, int startIndex = 0, int endIndex = 0)
        {
            // set indexes for propagating batch
            int numSamples = 0;
            if (startIndex == DEFAULT_INDEX && endIndex == startIndex)
            {
                startIndex = 0;
                endIndex = numSamples;
            }
            else
            {
                numSamples = endIndex - startIndex;
            }

            // softmax has prettier derivation so we need one step less
            if (activationFunction.GetType() == typeof(Softmax))
            {
                return input;
            }

            // value activations is overriden so be careful about it
            double[][][][] currGrad = MatOp.Cwise(input, activationFunction.Derivate(activations));

            return currGrad;
        }

        public override long Summary()
        {
            string output = String.Format("{0,-15} {1, -30} {2, -45}", layerName.ToString(),
                "(" + "None" + ", " + depth + ", " + width + ", " + height + ")", "0");
            Log.Info(output);

            return 0L;
        }

        public override AbstractLayer Compile(AbstractLayer previousLayer)
        {
            if (previousLayer == null && inputDimension == null)
            {
                string msg = "Dimension of first layer of network not set!";
                Utils.ThrowException(msg);
            }
            else if (previousLayer != null)
            {
                inputDimension = previousLayer.outputDimension;
                index = previousLayer.Index + 1;
            }

            outputDimension = inputDimension;
            compiled = true;

            return this;
        }
    }
}