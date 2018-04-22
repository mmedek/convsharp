using System;

namespace Zcu.Convsharp.Initializers
{
    /// <summary>
    /// Abstract class for bias initializers
    /// </summary>
    [Serializable]
    public abstract class AbstractBiasInitializer
    {
        /// <summary>
        /// Initialize 1D array of doubles according to input
        /// size.
        /// </summary>
        /// <param name="size">number of items in double array</param>
        /// <returns></returns>
        public abstract double[] Initialize(int size);
    }
}