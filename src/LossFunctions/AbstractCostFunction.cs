using System;

namespace Zcu.Convsharp.CostFunctions
{
    /// <summary>
    /// Class which is used as tempalte for all cost/loss functions
    /// used in this simple framework
    /// </summary>
    [Serializable]
    public abstract class AbstractCostFunction
    {
        /// <summary>
        /// Main method which compute loss/error in current iteration
        /// </summary>
        /// <param name="predicts">predictions according to iteration</param>
        /// <param name="targets">expected result according to train data</param>
        /// <returns>error value</returns>
        public abstract double[][][][] Derivate(double[][][][] predicts, double[][] targets, int batchStart = 0, int batchEnd = 0);
        /// <summary>
        /// Compute loss function for full dataset
        /// </summary>
        /// <param name="predicts">output from network</param>
        /// <param name="targets">expected targets in one hot encoding</param>
        /// <returns></returns>
        public abstract double Compute(double[][][][] predicts, double[][] targets);
        /// <summary>
        /// Compute accuracy and loss according to specific loss function
        /// </summary>
        /// <param name="currentInput">output from neural network</param>
        /// <param name="trainLabels">expected targets</param>
        /// <returns>accuracy</returns>
        public abstract double ComputeAccuracy(double[][][][] currentInput, double[][] trainLabels);
        /// <summary>
        /// Compute class of input
        /// </summary>
        /// <param name="input">current input</param>
        /// <returns>resulting class</returns>
        public abstract int GetResult(double[][][] input);
        /// <summary>
        /// Check dimension of input data and train/test labels
        /// </summary>
        /// <param name="input">input data</param>
        /// <param name="targets">train/test labels</param>
        /// <param name="batchStart">start of batch</param>
        /// <param name="batchEnd">end of batch</param>
        public abstract void TestDimension(double[][][][] input, double[][] targets, int batchStart = 0, int batchEnd = 0);
    }
}