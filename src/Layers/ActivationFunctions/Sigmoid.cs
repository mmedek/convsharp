using System;

namespace Zcu.Convsharp.Layers.ActivationFunctions
{
    /// <summary>
    /// Class Sigmoid for implementation sigmoid activation function
    /// </summary>
    [Serializable]
    public class Sigmoid : ActivationFunction
    {
        /// <summary>
        /// Empty constructor for creating instance of Sigmoid class
        /// for later usage
        /// </summary>
        public Sigmoid()
        {
        }

        public override double[][][][] Derivate(double[][][][] values)
        {
            int l0 = values.Length;
            int l1 = values[0].Length;
            int l2 = values[0][0].Length;
            int l3 = values[0][0][0].Length;

            double[][][][] sigmoids = Compute(values);
            for (int i = 0; i < l0; i++)
            {
                for (int j = 0; j < l1; j++)
                {
                    for (int k = 0; k < l2; k++)
                    {
                        for (int l = 0; l < l3; l++)
                            values[i][j][k][l] = sigmoids[i][j][k][l] * (1 - sigmoids[i][j][k][l]);
                    }
                }
            }

            return values;
        }

        public override double[][][][] Compute(double[][][][] values)
        {
            int l0 = values.Length;
            int l1 = values[0].Length;
            int l2 = values[0][0].Length;
            int l3 = values[0][0][0].Length;

            for (int i = 0; i < l0; i++)
            {
                for (int j = 0; j < l1; j++)
                {
                    for (int k = 0; k < l2; k++)
                    {
                        for (int l = 0; l < l3; l++)
                            values[i][j][k][l] = 1d / (1d + Math.Pow(Math.E, -1 * values[i][j][k][l]));
                    }
                }
            }

            return values;
        }
    }
}