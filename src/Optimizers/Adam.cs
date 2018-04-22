using System;
using System.Collections.Generic;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Layer;

namespace Zcu.Convsharp.Optimizers
{
    /// <summary>
    /// ADAM algorithm for updatiting weights
    /// https://arxiv.org/abs/1412.6980v8
    /// </summary>
    [Serializable]
    public class Adam : AbstractOptimizer
    {
        /// <summary>
        /// Values for momentum, contains current value
        /// for each layer according to key - index of layer
        /// </summary>
        private Dictionary<int, double[][][][]> m = new Dictionary<int, double[][][][]>();
        /// <summary>
        /// Values for powered momentum, contains current value
        /// for each layer according to key - index of layer
        /// </summary>
        private Dictionary<int, double[][][][]> r = new Dictionary<int, double[][][][]>();
        /// <summary>
        /// Variable beta which is used as multiplicator with
        /// momentum value.
        /// </summary>
        private double beta1;
        /// <summary>
        /// Variable beta which is used as multiplicator with
        /// powered momentum value.
        /// </summary>
        private double beta2;
        /// <summary>
        /// Constant multiplicator
        /// </summary>
        private double epsilon;

        /// <summary>
        /// Constructor for creating instance of Adam class
        /// </summary>
        /// <param name="learningRate">Value of learning rate</param>
        /// <param name="beta1">Beta 1 parameter</param>
        /// <param name="beta2">Beta 2 parameter</param>
        /// <param name="epsilon">Constant variable epsilon</param>
        public Adam(double learningRate, double beta1 = 0.9d, double beta2 = 0.999d, double epsilon = 0.0001d)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
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

            int index = ((AbstractLayer) learnableLayer).Index;
            // if it is first iteration we will init fields
            if (!m.ContainsKey(index))
                m.Add(index, Utils.Init4dArr(lx0, lx1, lx2, lx3));
            if (!r.ContainsKey(index))
                r.Add(index, Utils.Init4dArr(lx0, lx1, lx2, lx3));

            m[index] = MatOp.Add(MatOp.MultiplyByConst(dWeights, 1d - beta1), MatOp.MultiplyByConst(m[index], beta1));
            r[index] = MatOp.Add(MatOp.MultiplyByConst(MatOp.Cwise(dWeights, dWeights), 1d - beta2), MatOp.MultiplyByConst(r[index], beta2));

            double[][][][] mExt = MatOp.DivideByConst(m[index], 1d - Math.Pow(beta1, iteration));
            double[][][][] rExt = MatOp.DivideByConst(r[index], 1d - Math.Pow(beta2, iteration));

            var a = MatOp.MultiplyByConst(mExt, learningRate);
            var b = MatOp.AddConst(MatOp.Sqrt(rExt), epsilon);
            var c = MatOp.Mwise(a, b);
            learnableLayer.Weights = MatOp.Substract(weights, c);

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