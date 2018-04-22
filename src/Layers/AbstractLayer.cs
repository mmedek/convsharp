using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Abstract class used as template for layers which are used in convolional neural network
    /// e.g. ConvolutionalLayer, MaxPoolingLayer, etc.
    /// </summary>
    [Serializable]
    public abstract class AbstractLayer
    {
        /// <summary>
        /// Default values for start of batches if we are using
        /// full batch training
        /// </summary>
        protected const int DEFAULT_INDEX = 0;
        /// <summary>
        /// Variable which contains name of layer for printing it in summary
        /// </summary>
        protected string layerName;
        /// <summary>
        /// Dimension of output
        /// [number of items, depth, width, height]
        /// </summary>
        public Dimension outputDimension;
        /// <summary>
        /// Output from the convolution layer
        /// </summary>
        protected double[][][][] activations;
        /// <summary>
        /// Variable which is set after successfull compilation
        /// of this layer, before that the network will be off
        /// </summary>
        protected bool compiled = false;
        /// <summary>
        /// Dimension of input layer.
        /// </summary>
        protected Dimension inputDimension;
        /// <summary>
        /// Public property which is set as true if layer was
        /// already compiled false otherwise
        /// </summary>
        public bool Compiled
        {
            get { return compiled; }
        }
        /// <summary>
        /// Index is identificator of the layer. Index tells
        /// the position in network and is used for the
        /// identification of layer.
        /// </summary>
        protected int index = 0;
        /// <summary>
        /// Public property for index which is identifier of
        /// each layer.
        /// </summary>
        public int Index
        {
            get { return index; }
        }

        /// <summary>
        /// Forward propagation which is computed during learning procedure
        /// helps compute error function
        /// </summary>
        /// <param name="input">input matrix</param>
        /// <returns>out matrix computed during forward propagation
        /// through the layer</returns>
        public abstract double[][][][] ForwardPropagation(double[][][][] input, int startIndex = DEFAULT_INDEX, int endIndex = DEFAULT_INDEX, bool predict = false);
        /// <summary>
        /// Method which check the dimension of the layer and initializes
        /// all possible values in the layer - e.g. weights and biases.
        /// </summary>
        /// <param name="previousLayer">previous layer in the model</param>
        /// <returns>current layer</returns>
        public abstract AbstractLayer Compile(AbstractLayer previousLayer);
        /// <summary>
        /// Backward propagation which is computed during learning procedure
        /// helps decrease error on each layer
        /// </summary>
        /// <param name="input">input matrix</param>
        /// <returns>output matrix computed during backward propagation</returns>
        public abstract double[][][][] BackwardPropagation(double[][][][] input, int startIndex = DEFAULT_INDEX, int endIndex = DEFAULT_INDEX);
        /// <summary>
        /// Returns summary about layer which contains layer type, output shape and number
        /// of parameters
        /// </summary>
        /// <returns>number of parameters</returns>
        public abstract long Summary();
        /// <summary>
        /// Check if is initialize dimension same like parameters of input data.
        /// </summary>
        /// <param name="currDepth">depth of input data</param>
        /// <param name="currWidth">width of input data</param>
        /// <param name="currHeight">height of input data</param>
        /// <returns>true if is dimension same, false otherwise</returns>
        protected bool IsInputDataSameAsDim(int currDepth, int currWidth, int currHeight)
        {
            return currDepth == inputDimension.depth
                && currWidth == inputDimension.width
                && currHeight == inputDimension.height;
        }
    }
}