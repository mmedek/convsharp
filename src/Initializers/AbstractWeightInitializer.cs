using System;
using Zcu.Convsharp.Common;

namespace Zcu.Convsharp.Initializers
{
    /// <summary>
    /// Abstract class for weight initializer.
    /// This intitializer init weights in each
    /// learnable layer.
    /// </summary>
    [Serializable]
    public abstract class AbstractWeightInitializer
    {
        /// <summary>
        /// Initialize 4D array of doubles according to input
        /// dimension.
        /// </summary>
        /// <param name="dim">input dimension</param>
        /// <returns>4D array of initialized weights</returns>
        public abstract double[][][][] Initialize(Dimension dim);
    }
}
