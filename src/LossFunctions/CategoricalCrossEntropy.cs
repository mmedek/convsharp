using System;

namespace Zcu.Convsharp.CostFunctions
{
    /// <summary>
    /// Implementation of categorical cross entropy function
    /// </summary>
    [Serializable]
    public class CategoricalCrossEntropy : AbstractCategoricalCostFunction
    {
        /// <summary>
        /// Base of logarithm used in cross-entropy, e.g in 
        /// Tensorflow is used as base for cross-entropy E
        /// we are using 2
        /// </summary>
        private int LOG_BASE = 2;

        /// <summary>
        /// Empty constructor only for creating new instance
        /// of cross entropy function
        /// </summary>
        public CategoricalCrossEntropy()
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
                    sum += targets[i][j] * Math.Log(input[index][0][0][j], LOG_BASE);
                }
                index++;
            }
            return -sum / size;
        }

        public override double[][][][] Derivate(double[][][][] input, double[][] trainLabels, int startIndex, int endIndex)
        {
            int batchSize = input.Length;

            if (startIndex == 0 && endIndex == startIndex)
            {
                startIndex = 0;
                endIndex = batchSize;
            }
            else
            {
                batchSize = endIndex - startIndex;
            }

            TestDimension(input, trainLabels, startIndex, endIndex);
            int depth = 1;
            int depthIndex = 0;
            int labelsCount = input[0][0][0].GetLength(0);
            double[][][][] diff = new double[batchSize][][][];
            int index = 0;
            for (int i = startIndex; i < endIndex; i++)
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