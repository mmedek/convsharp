using System;

namespace Zcu.Convsharp.Layers.ActivationFunctions
{
    /// <summary>
    /// Class Relu for implementation relu activation function
    /// </summary>
    [Serializable]
    public class Tanh : ActivationFunction
    {
        /// <summary>
        /// Empty constructor for creating instance of relu class
        /// for later usage
        /// </summary>
        public Tanh()
        {
        }

        public override double[][][][] Derivate(double[][][][] values)
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
                            values[i][j][k][l] = 1 - Math.Pow(Math.Tanh(values[i][j][k][l]), 2);
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

            double exp, negExp;
            for (int i = 0; i < l0; i++)
            {
                for (int j = 0; j < l1; j++)
                {
                    for (int k = 0; k < l2; k++)
                    {
                        for (int l = 0; l < l3; l++)
                        {
                            exp = Math.Pow(Math.E, values[i][j][k][l]);
                            negExp = Math.Pow(Math.E, -1 * values[i][j][k][l]);
                            values[i][j][k][l] = (exp - negExp) / (exp + negExp);
                        }
                    }
                }
            }

            return values;
        }
    }
}