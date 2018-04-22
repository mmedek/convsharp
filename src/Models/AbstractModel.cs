using System;
using System.Collections.Generic;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.CostFunctions;
using Zcu.Convsharp.Layer;
using Zcu.Convsharp.Loaders;
using Zcu.Convsharp.Optimizers;

namespace Zcu.Convsharp.Model
{
    /// <summary>
    /// Abstract class used as template for model of convolutional neural network
    /// </summary>
    [Serializable]
    public abstract class AbstractModel
    {
        /// <summary>
        /// Method which prints how looks current model
        /// of network if was network already trained
        /// </summary>
        public abstract void Summary();
        /// <summary>
        /// Compile model through all layers and check
        /// dimensions of networks
        /// </summary>
        /// <param name="costFunction">Type of cost function which will
        /// be used for training.</param>
        /// <param name="optimizer">Type of optimizer which will be used
        /// for training.</param>
        public abstract void Compile(AbstractCostFunction costFunction, AbstractOptimizer optimizer);
        /// <summary>
        /// Method for adding new layer into model of convolutional neural network
        /// </summary>
        /// <param name="layer">e.g. ConvolutionalLayer</param>
        public abstract void Add(AbstractLayer layer);
        /// <summary>
        /// Method for training neural network according to settings.
        /// </summary>
        /// <param name="loader">loader which implements loading
        /// from the current dataset</param>
        /// <param name="epochCount">Number of epochs for training.</param>
        /// <param name="useValidationSet">parametr which signal if we will
        /// use validation set or not</param>
        /// <returns>list of results</returns>
        public abstract List<EpochHistory> Fit(AbstractLoader loader, int epochCount, bool useValidationSet = false);
        /// <summary>
        /// Method which will save the model of neural network.
        /// </summary>
        /// <param name="pathToModel">path were model will saved</param>
        public abstract void Save(string pathToModel);
        /// <summary>
        /// Method for evaluating data
        /// </summary>
        /// <param name="testImages">Test data</param>
        /// <param name="testLabels">Test labels</param>
        /// <param name="print">Variable which prints results during
        /// returning results if is true, otherwise not</param>
        /// <returns>tuple (accuracy, loss)</returns>
        public abstract Tuple<double, double> Evaluate(double[][][][] testImages, double[][] testLabels, bool print = true);
        /// <summary>
        /// Method returns class according to trained model.
        /// </summary>
        /// <param name="item">Input data in same format like training data</param>
        /// <returns>Classified class</returns>
        public abstract int Predict(double[][][] item);
    }
}