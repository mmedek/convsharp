using System;
using Zcu.Convsharp.Layer;

namespace Zcu.Convsharp.Optimizers
{
    /// <summary>
    /// Abstract class for all optimizers which will
    /// be used for updating parameters during backpropagation.
    /// </summary>
    [Serializable]
    public abstract class AbstractOptimizer
    {
        /// <summary>
        /// Variable for storing current learning rate
        /// </summary>
        protected double learningRate;
        /// <summary>
        /// Method for updating weights and biases during
        /// backpropagation.
        /// </summary>
        /// <param name="learnableLayer">Layer with weights and biases which
        /// will be updated.</param>
        /// <param name="iteration">Number of iteration of learning call.</param>
        public abstract void UpdateWeights(ILearnable learnableLayer, int iteration = 1);
    }
}