using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.Layers.ActivationFunctions
{
    /// <summary>
    /// Class Softmax for implementation softmax activation function
    /// </summary>
    [Serializable]
    public class Softmax : ActivationFunction
    {
        /// <summary>
        /// Empty constructor for creating instance of softmax class
        /// for later usage
        /// </summary>
        public Softmax()
        {
        }

        public override double[][][][] Derivate(double[][][][] value)
        {
            // value is derivated together with loss function
            // and we expected that softmax will be used only
            // as last layer because if we use softmax as hidden
            // layer we minimaze the nonlinearity
            return value;
        }

        public override double[][][][] Compute(double[][][][] values)
        {
            TestIsFlatten(values);

            int l0 = values.Length;
            int l1 = values[0].Length;
            int l2 = values[0][0].Length;
            int l3 = values[0][0][0].Length;

            double sum, value;
            for (int i = 0; i < l0; i++)
            {
                sum = 0;
                for (int l = 0; l < l3; l++)
                {
                    value = Math.Pow(Math.E, values[i][0][0][l]);
                    values[i][0][0][l] = value;
                    sum += value;
                }
                for (int l = 0; l < l3; l++)
                {
                    values[i][0][0][l] /= sum;
                }
            }

            return values;
        }

        /// <summary>
        /// Test if the input into dense layer has flatten
        /// shape
        /// </summary>
        /// <param name="input">input data into dense layer</param>
        private void TestIsFlatten(double[][][][] input)
        {
            int numFilters = input[0].GetLength(0);
            int numRows = input[0][0].GetLength(0);

            if (numFilters != 1 || numRows != 1)
            {
                string msg = "Input into dense layer is not flatten, please use " +
                    "the flatten layer before dense layer.";
                Utils.ThrowException(msg);
            }
        }

        /// <summary>
        /// Normalize softmax into [0-1] value
        /// </summary>
        /// <param name="value">activation x</param>
        /// <param name="sum">divisor which is equal to sum of all 
        /// activation for current vector</param>
        /// <returns>final activation for neuron</returns>
        public double Normalize(double value, double sum)
        {
            return value / sum;
        }
    }
}