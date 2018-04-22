using System;
using System.Collections.Generic;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Initializers;
using Zcu.Convsharp.Logger;
using Zcu.Convsharp.Regularizers;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Layer which proceed weight sum on each 
    /// neuron defined in this layer
    /// </summary>
    [Serializable]
    public class LinearLayer : AbstractLayer, ILearnable
    {
        #region Local variables
        /// <summary>
        /// Number of neurons in this layer
        /// </summary>
        private int numNeurons;
        /// <summary>
        /// Biases which will be used during summing 
        /// activations
        /// </summary>
        private double[] biases;
        /// <summary>
        /// Public property for biases which is used for
        /// biases updates during learning.
        /// </summary>
        public double[] Biases
        {
            get { return biases; }
            set { biases = value; }
        }
        /// <summary>
        /// Derivation of biases used for
        /// updating biases
        /// </summary>
        private double[] dBiases;
        /// <summary>
        /// Public property for biases gradient which is used
        /// for biases updates during learning.
        /// </summary>
        public double[] Dbiases
        {
            get { return dBiases; }
            set { }
        }
        /// <summary>
        /// Output from the layer
        /// </summary>
        private double[][][][] weights;
        /// <summary>
        /// Public property for weights/filters which is used during
        /// learning.
        /// </summary>
        public double[][][][] Weights
        {
            get { return weights; }
            set { weights = value; }
        }
        /// <summary>
        /// Derivation of weights which is used for
        /// updating weights
        /// </summary>
        private double[][][][] dWeights;
        /// <summary>
        /// Public property for weights which is used for
        /// weights updates during learning.
        /// </summary>
        public double[][][][] Dweights
        {
            get { return dWeights; }
            set { }
        }
        /// <summary>
        /// Number of neurons in flatten layer if exist
        /// otherwise is equal to zero
        /// </summary>
        private int flattenSize = 0;
        /// <summary>
        /// Variable which contains last input from forward
        /// propagation and is used during backpropagation
        /// step
        /// </summary>
        private double[][][][] lastInput;
        /// <summary>
        /// Instance of regularizer, which can be used according
        /// to setting of layer
        /// </summary>
        private AbstractRegularizer regularizer;
        /// <summary>
        /// Public property for regularizer.
        /// </summary>
        public AbstractRegularizer Regularizer
        {
            get { return regularizer; }
            set { }
        }
        /// <summary>
        /// Instance of initializer which is used for
        /// initialization of biases
        /// </summary>
        private AbstractBiasInitializer biasInitializer;
        /// <summary>
        /// Instance of initializer which is used for
        /// initialization of weights
        /// </summary>
        private AbstractWeightInitializer weightInitializer;
        #endregion Local variables

        /// <summary>
        /// Constructor for creating new dense layer
        /// </summary>
        /// <param name="activationFunction">type of activation function</param>
        /// <param name="numNeurons">number of neurons which will be used</param>
        public LinearLayer(int numNeurons, AbstractBiasInitializer biasInitializer = null,
            AbstractWeightInitializer weightInitializer = null, AbstractRegularizer regularizer = null)
        {
            this.numNeurons = numNeurons;
            this.regularizer = regularizer;
            layerName = "Linear";
        }

        public override double[][][][] ForwardPropagation(double[][][][] input, int startIndex = 0, int endIndex = 0, bool predict = false)
        {
            TestIsFlatten(input);

            int numSamples = input.Length;
            lastInput = input;

            if (!IsInputDataSameAsDim(input[0].Length, input[0][0].Length, input[0][0][0].Length))
            {
                string msg = "Input data have not same dimension as initialized dimension in flatten layer.";
                Utils.ThrowException(msg);
            }

            if (startIndex == DEFAULT_INDEX && endIndex == startIndex)
            {
                startIndex = 0;
                endIndex = numSamples;
            }
            else
            {
                numSamples = endIndex - startIndex;
            }
            flattenSize = input[0][0][0].Length;

            activations = Utils.Init4dArr(numSamples, 1, 1, outputDimension.height);

            // for the dense layer will be 'channel' everytime set
            // to 1, because we want to same interface for each layer
            int channel;
            // input into every neuron is summed
            double sum;
            // activation index - needed if we have batches
            int activationIndex = 0;
            for (int i = startIndex; i < endIndex; i++)
            {
                channel = 0;
                for (int j = 0; j < numNeurons; j++)
                {
                    sum = 0;
                    for (int k = 0; k < flattenSize; k++)
                    {
                        sum += input[activationIndex][0][0][k] * weights[k][0][0][j];
                    }
                    sum += biases[j];
                    // compute final sum of neuron for this
                    activations[activationIndex][channel][channel][j] = sum;
                }
                activationIndex++;
            }

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

            int n = input.Length;
            dWeights = MatOp.Dot(MatOp.Transpose(lastInput, startIndex, endIndex), input, divisor: n);
            // regularization
            if (regularizer != null)
            {
                // lambda/m * W
                double[][][][] regMatrix = regularizer.Regularize(weights, numSamples);
                // add regularization coef
                dWeights = MatOp.Add(dWeights, regMatrix);
            }
            dBiases = MatOp.Mean(input);
            double[][][][] result = MatOp.Dot(input, MatOp.Transpose(weights));
          
            return result;
        }

        /// <summary>
        /// Test if the input into dense layer has flatten
        /// shape
        /// </summary>
        /// <param name="input">input data into dense layer</param>
        private void TestIsFlatten(double[][][][] input)
        {
            int numFilters = input[0].GetLength(0);
            int numRows = input[0][0].GetLength(0);

            if (numFilters != 1 || numRows != 1)
            {
                string msg = "Input into dense layer is not flatten, please use " +
                    "the flatten layer before dense layer.";
                Utils.ThrowException(msg);
            }
        }

        /// <summary>
        /// Test if the dimension into linear layer has flatten
        /// shape
        /// </summary>
        /// <param name="dim">dimension of input data</param>
        private void TestIsDimensionFlatten(Dimension dim)
        {
            int numFilters = dim.depth;
            int numRows = dim.width;

            if (numFilters != 1 || numRows != 1)
            {
                string msg = "Input into linear layer is not flatten, please use " +
                    "the flatten layer before linear layer.";
                Utils.ThrowException(msg);
            }
        }

        public override long Summary()
        {
            List<string[]> summaries = new List<string[]>();
            string[] layerSummary = new string[3];

            long paramameters = (biases.GetLength(0) + weights.Length
                * weights[0][0][0].Length);

            string output = String.Format("{0,-15} {1, -30} {2, -45}", layerName.ToString(),
                "(" + "None" + ", 1, 1, " + numNeurons + ")", paramameters.ToString());
            Log.Info(output);

            return paramameters;
        }

        public override AbstractLayer Compile(AbstractLayer previousLayer)
        {
            if (previousLayer == null)
            {
                string msg = "Dimension of first layer of network not set!";
                Utils.ThrowException(msg);
            }
            else if (previousLayer != null)
            {
                inputDimension = previousLayer.outputDimension;
                index = previousLayer.Index + 1;
            }

            inputDimension = previousLayer.outputDimension;

            outputDimension = new Dimension(inputDimension.imageCount, 1, 1, numNeurons);

            activations = Utils.Init4dArr(inputDimension.imageCount, 1, 1, numNeurons);

            #region Init initializers
            if (biasInitializer == null)
            {
                biasInitializer = new ConstantBiasInitializer(0d);
            }
            biases = biasInitializer.Initialize(numNeurons);


            int fanIn = inputDimension.height;
            int fanOut = numNeurons;
            if (weightInitializer == null)
            {
                weightInitializer = new XavierWeightInitializer(fanIn, fanOut);
            }
            weights = weightInitializer.Initialize(new Dimension(inputDimension.height, 1, 1, numNeurons));
            #endregion Init initializers

            compiled = true;

            return this;
        }
    }
}