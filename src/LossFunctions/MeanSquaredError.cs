using System;

namespace Zcu.Convsharp.CostFunctions
{
    /// <summary>
    /// Class for mean squared error loss function
    /// </summary>
    [Serializable]
    public class MeanSquaredError : AbstractNonCategoricalCostFunction
    {
        /// <summary>
        /// Power which is used after substraction of
        /// target with input in absolute value
        /// </summary>
        private int POWER = 2;

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
                    sum += Math.Pow(targets[i][j] - input[index][0][0][j], POWER);
                }
                index++;
            }
            return 0.5d * sum / size;
        }

        public override double[][][][] Derivate(double[][][][] input, double[][] targets, int batchStart, int batchEnd)
        {
            TestDimension(input, targets, batchStart, batchEnd);
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
                    diff[index][depthIndex][depthIndex][j] = input[index][0][0][j] - targets[i][j];
                }
                index++;
            }
            return diff;
        }
    }
}