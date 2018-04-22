using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.Initializers
{
    /// <summary>
    /// Class for weight initializer which assign all
    /// weights value according to normal distribution
    /// </summary>
    [Serializable]
    public class NormalWeightInitializer : AbstractWeightInitializer
    {
        /// <summary>
        /// Mean of generated weights
        /// </summary>
        private double mean;
        /// <summary>
        /// Standard deviation of generated weights
        /// </summary>
        private double std;

        /// <summary>
        /// Constructor for creating new instances of NormalWeightInitializer
        /// class.
        /// </summary>
        /// <param name="mean">Mean of weights</param>
        /// <param name="std">Standard deviation of weights</param>
        public NormalWeightInitializer(double mean = 0d, double std = 0.5d)
        {
            this.mean = mean;
            this.std = std;
        }

        public override double[][][][] Initialize(Dimension dim)
        {
            double randStdNormal, randNormal;
            // init array for weights with zeros
            double[][][][] initializations = Utils.Init4dArr(dim.imageCount, dim.depth,
                dim.width, dim.height);
            for (int i = 0; i < dim.imageCount; i++)
            {
                for (int j = 0; j < dim.depth; j++)
                {
                    for (int k = 0; k < dim.width; k++)
                    {
                        for (int l = 0; l < dim.height; l++)
                        {
                            // generate random weights according to normal
                            // distribution
                            randStdNormal = Math.Sqrt(-2.0 * Math.Log(1 - Utils.GetRandomDouble())) *
                                Math.Sin(2.0 * Math.PI * (1 - Utils.GetRandomDouble()));
                            randNormal = mean + std * randStdNormal;
                            initializations[i][j][k][l] = randNormal;
                        }
                    }
                }
            }
            return initializations;
        }
    }

    /// <summary>
    /// Class for weight initializer which assign all
    /// weights value according to xavier algorithm    
    /// http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    /// </summary>
    [Serializable]
    public class XavierWeightInitializer : AbstractWeightInitializer
    {
        /// <summary>
        /// Mean of generated weights
        /// </summary>
        private double mean;
        /// <summary>
        /// Standard deviation of generated weights
        /// </summary>
        private double std;
        /// <summary>
        /// Number of input neurons into current layer
        /// </summary>
        private int fanIn;
        /// <summary>
        /// Number of output neurons into current layer
        /// </summary>
        private int fanOut;

        /// <summary>
        /// Constructor for creating new instances of XavierWeightInitializer
        /// class.
        /// </summary>
        /// <param name="fanIn">Number of input neurons into current layer</param>
        /// <param name="fanOut">Number of output neurons into current layer</param>
        /// <param name="mean">Mean of weights</param>
        public XavierWeightInitializer(int fanIn, int fanOut, double mean = 0d)
        {
            this.mean = 0d;
            this.fanIn = fanIn;
            this.fanOut = fanOut;
        }

        public override double[][][][] Initialize(Dimension dim)
        {
            double randStdNormal, randNormal;
            // init array for weights with zeros
            double[][][][] initializations = Utils.Init4dArr(dim.imageCount, dim.depth,
                dim.width, dim.height);
            // variance = 2d / (fanIn + fanOut), std = sqrt(variance)
            std = Math.Sqrt(2d / (fanIn + fanOut));
            for (int i = 0; i < dim.imageCount; i++)
            {
                for (int j = 0; j < dim.depth; j++)
                {
                    for (int k = 0; k < dim.width; k++)
                    {
                        for (int l = 0; l < dim.height; l++)
                        {
                            // generate random weights according to normal
                            // distribution but with different variance for each
                            // layer
                            randStdNormal = Math.Sqrt(-2.0 * Math.Log(1 - Utils.GetRandomDouble())) *
                                Math.Sin(2.0 * Math.PI * (1 - Utils.GetRandomDouble()));
                            randNormal = mean + std * randStdNormal;
                            initializations[i][j][k][l] = randNormal;
                        }
                    }
                }
            }
            return initializations;
        }
    }
}