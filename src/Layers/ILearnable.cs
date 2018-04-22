using Zcu.Convsharp.Regularizers;

namespace Zcu.Convsharp.Layer
{
    /// <summary>
    /// Interface which implements only layers which have weights and
    /// using these weights during process of backpropagation when
    /// we are updating these weights
    /// </summary>
    public interface ILearnable
    {
        /// <summary>
        /// Property which enforce definiton of regularizer.
        /// </summary>
        AbstractRegularizer Regularizer { get; set; }
        /// <summary>
        /// Property which enforce definition of biases.
        /// </summary>
        double[] Biases { get; set; }
        /// <summary>
        /// Property which enforce definition of weights.
        /// </summary>
        double[][][][] Weights { get; set; }
        /// <summary>
        /// Property which enforce definition of gradient of weights.
        /// </summary>
        double[][][][] Dweights { get; set; }
        /// <summary>
        /// Property which enforce definition of gradient of biases.
        /// </summary>
        double[] Dbiases { get; set; }
    }
}
