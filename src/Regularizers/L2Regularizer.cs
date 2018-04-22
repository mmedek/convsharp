using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.Regularizers
{
    /// <summary>
    /// Most common method for regularization in neural network.
    /// We are using it for fithting with overfitting.
    /// </summary>
    [Serializable]
    public class L2Regularizer : AbstractRegularizer
    {
        /// <summary>
        /// Constructor for creating instance of class L2Regularizer
        /// </summary>
        /// <param name="regularizationParameter">Regularization parameter</param>
        public L2Regularizer(double regularizationParameter)
        {
            this.regularizationParameter = regularizationParameter;
        }

        public override double ComputeAdditionToCostFunc(double[][][][] matrix, int numExamples)
        {
            double multiplicator = regularizationParameter / ( 2 * numExamples);
            double sum = 0;
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    for (int k = 0; k < matrix[0][0].Length; k++)
                    {
                        for (int l = 0; l < matrix[0][0][0].Length; l++)
                            sum += matrix[i][j][k][l] * matrix[i][j][k][l];
                    }
                }
            }
            sum *= multiplicator;
            return sum;
        }

        public override double[][][][] Regularize(double[][][][] matrix, int numExamples)
        {
            double multiplicator = regularizationParameter / numExamples;
            double[][][][] resMatrix = MatOp.MultiplyByConst(matrix, multiplicator);
            return resMatrix;
        }
    }
}