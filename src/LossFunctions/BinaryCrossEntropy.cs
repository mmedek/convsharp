using System;

namespace Zcu.Convsharp.CostFunctions
{
    /// <summary>
    /// Class for binary cross entropy function
    /// which is similar to categorical cross entorpy
    /// but we expect only one input here (usually from
    /// sigmoid activation layer).
    /// </summary>
    [Serializable]
    public class BinaryCrossEntropy : AbstractNonCategoricalCostFunction
    {
        /// <summary>
        /// Base of logarithm used in cross-entropy, e.g in 
        /// Tensorflow is used as base for cross-entropy E
        /// we are using 2
        /// </summary>
        private int LOG_BASE = 2;

        /// <summary>
        /// Empty constructor only for creating new instance
        /// of binary cross entropy function
        /// </summary>
        public BinaryCrossEntropy()
        {
        }

        public override double Compute(double[][][][] input, double[][] targets)
        {
            TestDimension(input, targets);
            int labelsCount = input[0][0][0].GetLength(0);
            int size = input.Length;
            double sum = 0;
            int index = 0;
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < labelsCount; j++)
                {
                    sum += targets[i][j] * Math.Log(input[index][0][0][j], LOG_BASE)
                        + (1 - targets[i][j]) * Math.Log(1 - input[index][0][0][j], LOG_BASE);
                }
                index++;
            }
            return -sum / size;
        }

        public override double[][][][] Derivate(double[][][][] input, double[][] trainLabels, int batchStart, int batchEnd)
        {
            TestDimension(input, trainLabels, batchStart, batchEnd);
            int depth = 1;
            int depthIndex = 0;
            int labelsCount = input[0][0][0].GetLength(0);
            int batchSize = input.GetLength(0);
            double[][][][] diff = new double[batchSize][][][];
            int index = 0;
            for (int i = 0; i < input.Length; i++)
            {
                diff[index] = new double[depth][][];
                diff[index][depthIndex] = new double[depth][];
                diff[index][depthIndex][depthIndex] = new double[labelsCount];
                for (int j = 0; j < labelsCount; j++)
                {
                    diff[index][depthIndex][depthIndex][j] = input[index][0][0][j] - trainLabels[i][j];
                }
                index++;
            }
            return diff;
        }
    }
}