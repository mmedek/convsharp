using System;
using Zcu.Convsharp.Layer;

namespace Zcu.Convsharp.Optimizers
{
    /// <summary>
    /// Stochastic gradient descent algorithm
    /// for updating weights during backrpopagation
    /// </summary>
    [Serializable]
    public class SGD : AbstractOptimizer
    {
        /// <summary>
        /// Constructor for creating instance of
        /// class SGD
        /// </summary>
        /// <param name="learningRate">Value of learning rate.</param>
        public SGD(double learningRate)
        {
            this.learningRate = learningRate;
        }

        public override void UpdateWeights(ILearnable learnableLayer, int iteration = 1)
        {
            // update all weights by stochastic gradient descent
            double[][][][] weights = learnableLayer.Weights;
            double[][][][] dWeights = learnableLayer.Dweights;

            int lx0 = weights.Length;
            int lx1 = weights[0].Length;
            int lx2 = weights[0][0].Length;
            int lx3 = weights[0][0][0].Length;
            for (int i = 0; i < lx0; i++)
            {
                for (int j = 0; j < lx1; j++)
                {
                    for (int k = 0; k < lx2; k++)
                    {
                        for (int l = 0; l < lx3; l++)
                            weights[i][j][k][l] -= learningRate * dWeights[i][j][k][l];
                    }
                }
            }

            // update biases
            double[] biases = learnableLayer.Biases;
            double[] dBiases = learnableLayer.Dbiases;

            for (int i = 0; i < biases.Length; i++)
            {
                biases[i] -= learningRate * dBiases[i];
            }
        }
    }
}