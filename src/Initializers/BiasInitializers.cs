using System;

namespace Zcu.Convsharp.Initializers
{
    /// <summary>
    /// Bias initializer which set all weights on specified constant
    /// according to input.
    /// </summary>
    [Serializable]
    public class ConstantBiasInitializer : AbstractBiasInitializer
    {
        /// <summary>
        /// Constant variable which will be used for initialization
        /// of all bias values.
        /// </summary>
        private double value;

        /// <summary>
        /// Constructor which creates new instance
        /// of ConstantBiasInitializer class.
        /// </summary>
        /// <param name="value">constant value</param>
        public ConstantBiasInitializer(double value)
        {
            this.value = value;
        }

        public override double[] Initialize(int size)
        {
            double[] biases = new double[size];
            for (int i = 0; i < size; i++)
            {
                biases[i] = value;
            }
            return biases;
        }
    }

    // here expected other initializers in the future
}
