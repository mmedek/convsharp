using System;

namespace Zcu.Convsharp.Layers.ActivationFunctions
{
    /// <summary>
    /// Class Relu for implementation relu activation function
    /// </summary>
    [Serializable]
    public class Relu : ActivationFunction
    {
        /// <summary>
        /// Empty constructor for creating instance of Relu class
        /// for later usage
        /// </summary>
        public Relu()
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
                            // value[value < 0] = 0
                            values[i][j][k][l] = (values[i][j][k][l] > 0) ? 1 : 0;
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
                            values[i][j][k][l] = (values[i][j][k][l] > 0) ? values[i][j][k][l] : 0;
                    }
                }
            }

            return values;
        }
    }
}