using System;

namespace Zcu.Convsharp.Layers.ActivationFunctions
{
    /// <summary>
    /// ActivationFunction is abstract class which defines 
    /// pattern for each activation function which will be
    /// used in CNN
    /// </summary>
    [Serializable]
    public abstract class ActivationFunction
    {
        /// <summary>
        /// Method which return value according to type
        /// of activation function, simply proceed func
        /// which is defined in implementation - e.g.
        /// sigmoid, relu, tanh, ...
        /// </summary>
        /// <param name="value">x parametr in function</param>
        /// <returns></returns>
        public abstract double[][][][] Compute(double[][][][] value);
        /// <summary>
        /// Derivation of activation function
        /// </summary>
        /// <param name="value">values which we
        /// need to derivate</param>
        /// <returns>derivated values</returns>
        public abstract double[][][][] Derivate(double[][][][] value);
    }
}
