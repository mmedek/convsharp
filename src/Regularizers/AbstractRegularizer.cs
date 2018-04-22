using System;

namespace Zcu.Convsharp.Regularizers
{
    /// <summary>
    /// Abstract class for regularizers which
    /// are used for regularizing network
    /// (for fighting with overfitting)
    /// </summary>
    [Serializable]
    public abstract class AbstractRegularizer
    {
        /// <summary>
        /// Parameter for regularization
        /// </summary>
        protected double regularizationParameter;
        /// <summary>
        /// Number of examples in current batch
        /// </summary>
        protected int numExamples;
        /// <summary>
        /// Compute addition to cost function which will
        /// be added because loss function will be higher
        /// if we will use regularization.
        /// </summary>
        /// <param name="matrix">input matrix</param>
        /// <param name="numExamples">number of samples in matrix</param>
        /// <returns>addition to cost function</returns>
        public abstract double ComputeAdditionToCostFunc(double[][][][] matrix, int numExamples);
        /// <summary>
        /// Regularization
        /// </summary>
        /// <param name="matrix">weight matrix</param>
        /// <param name="numExamples">number of samples</param>
        /// <returns>regularized weight matrix</returns>
        public abstract double[][][][] Regularize(double[][][][] matrix, int numExamples);
    }
}